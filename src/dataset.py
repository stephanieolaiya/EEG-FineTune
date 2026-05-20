"""
ThingsEEG2Dataset — PyTorch Dataset for the THINGS-EEG2 dataset.

Reference: Gifford et al. 2022, NeuroImage — https://osf.io/3jk45/

Expected data layout (under data_root):
    preprocessed_data/
        sub-01/
            preprocessed_eeg_training.npy   # dict, see below
            preprocessed_eeg_test.npy
        sub-02/
            ...
    image_metadata.npy                      # optional concept labels
    image_set/                              # JPEG stimulus images (for CLIP cache)

Each .npy file is a dict (loaded with allow_pickle=True) containing:
    'preprocessed_eeg_data' : float32, shape (n_concepts, n_reps, n_channels, n_times)
    'ch_names'              : list of channel name strings
    'times'                 : float array of time points in seconds

The data was already preprocessed by the dataset authors:
    - Bandpass filtered (0.1–100 Hz)
    - Downsampled to 250 Hz
    - Epoched from –0.2 to 0.8 s relative to stimulus onset
    - Baseline corrected (–0.2 to 0 s)
    - MVNN whitened

This dataset applies a further optional z-score and time-window crop before
returning (eeg_tensor, concept_idx, clip_embedding) tuples.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import open_clip
except ImportError as e:
    raise ImportError("open-clip-torch is required: pip install open-clip-torch") from e

from PIL import Image


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def zscore_normalize(eeg: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel over the time axis.
    Input shape: (n_channels, n_times)
    """
    mean = eeg.mean(axis=-1, keepdims=True)
    std  = eeg.std(axis=-1, keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    return (eeg - mean) / std


def crop_epoch(eeg: np.ndarray, times: np.ndarray, t_start: float = 0.0, duration: float = 0.5) -> np.ndarray:
    """Crop epoch to [t_start, t_start + duration] seconds.
    Input shape: (n_channels, n_times); times is the 1-D array of time points.
    """
    mask = (times >= t_start) & (times < t_start + duration)
    return eeg[:, mask]


# ---------------------------------------------------------------------------
# CLIP embedding cache
# ---------------------------------------------------------------------------

def build_clip_embedding_cache(
    image_paths: list[str],
    cache_path: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode all stimulus images with CLIP and save to cache_path.
    Returns float32 array of shape (n_images, embed_dim).
    """
    print(f"Building CLIP embedding cache for {len(image_paths)} images...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert("RGB"))
                except Exception:
                    img = torch.zeros(3, 224, 224)
                imgs.append(img)
            batch_tensor = torch.stack(imgs).to(device)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    cache = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(cache_path, cache)
    print(f"Cache saved → {cache_path}  shape={cache.shape}")
    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThingsEEG2Dataset(Dataset):
    """PyTorch Dataset for the THINGS-EEG2 preprocessed data.

    Args:
        data_root:      Path containing preprocessed_data/, image_set/, etc.
        subjects:       Zero-padded subject IDs, e.g. ["01", "02"].
        split:          "training" or "test".
        avg_reps:       If True, average across EEG repetitions per concept
                        (recommended for training). If False, each (concept, rep)
                        pair is a separate sample.
        t_start:        Start of post-stimulus window in seconds (default 0.0).
        duration:       Length of window in seconds (default 0.5).
        apply_zscore:   Apply per-channel z-score normalization (default True).
        clip_device:    Device for CLIP encoding ("cpu" or "cuda").
        rebuild_cache:  Force re-computation of CLIP embedding cache.
    """

    def __init__(
        self,
        data_root: str | Path,
        subjects: Sequence[str],
        split: str = "training",
        avg_reps: bool = True,
        t_start: float = 0.0,
        duration: float = 0.5,
        apply_zscore: bool = True,
        clip_device: str = "cpu",
        rebuild_cache: bool = False,
    ) -> None:
        self.data_root    = Path(data_root)
        self.subjects     = list(subjects)
        self.split        = split
        self.avg_reps     = avg_reps
        self.t_start      = t_start
        self.duration     = duration
        self.apply_zscore = apply_zscore

        self._load_eeg()
        self._load_or_build_clip_cache(clip_device, rebuild_cache)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_eeg(self) -> None:
        """Load and concatenate EEG data across subjects.

        After loading, self.eeg_data has shape:
            avg_reps=True  → (N, n_channels, n_times)  where N = total concepts
            avg_reps=False → (N, n_channels, n_times)  where N = concepts × reps
        self.concept_indices maps each sample to its concept index (0-based).
        """
        eeg_list:     list[np.ndarray] = []
        concept_list: list[np.ndarray] = []

        for sub in self.subjects:
            npy_path = (
                self.data_root
                / "preprocessed_data"
                / f"sub-{sub}"
                / f"preprocessed_eeg_{self.split}.npy"
            )
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"EEG file not found: {npy_path}\n"
                    "Run the download cell in notebooks/phase1_setup.ipynb first."
                )
            d = np.load(npy_path, allow_pickle=True).item()
            eeg = d["preprocessed_eeg_data"]   # (n_concepts, n_reps, n_channels, n_times)
            times = d["times"]

            if not hasattr(self, "times"):
                self.times    = times
                self.ch_names = d["ch_names"]
                self.sfreq    = round(1.0 / float(np.diff(times).mean()))
                self.n_channels = eeg.shape[2]

            if self.avg_reps:
                eeg = eeg.mean(axis=1)  # (n_concepts, n_channels, n_times)
            else:
                n_c, n_r, n_ch, n_t = eeg.shape
                eeg = eeg.reshape(n_c * n_r, n_ch, n_t)

            n_samples   = eeg.shape[0]
            n_concepts  = d["preprocessed_eeg_data"].shape[0]
            concept_idx = np.repeat(np.arange(n_concepts), 1 if self.avg_reps else d["preprocessed_eeg_data"].shape[1])

            eeg_list.append(eeg)
            concept_list.append(concept_idx)

        self.eeg_data       = np.concatenate(eeg_list,     axis=0)  # (N, C, T)
        self.concept_indices = np.concatenate(concept_list, axis=0)  # (N,)
        self.n_total_samples = len(self.eeg_data)
        self.n_concepts_per_sub = d["preprocessed_eeg_data"].shape[0]

    def _load_or_build_clip_cache(self, device: str, rebuild: bool) -> None:
        cache_path = self.data_root / "embeddings_cache.npy"
        if cache_path.exists() and not rebuild:
            self.clip_embeddings = np.load(cache_path)
            return

        image_set_dir = self.data_root / "image_set"
        if not image_set_dir.exists():
            # Fall back to zero embeddings so sanity check can still run
            print(
                f"[WARNING] image_set/ not found at {image_set_dir}.\n"
                "CLIP embeddings will be zero vectors until you download the images\n"
                "and run Step 6 in the Colab notebook."
            )
            n_concepts = self.n_concepts_per_sub
            self.clip_embeddings = np.zeros((n_concepts, 512), dtype=np.float32)
            return

        # Build from images
        meta_path = self.data_root / "image_metadata.npy"
        if meta_path.exists():
            meta = np.load(meta_path, allow_pickle=True).item()
            image_files = list(meta.get("image_path", []))
        else:
            image_files = sorted(
                str(p.name) for p in image_set_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
        full_paths = [str(image_set_dir / f) for f in image_files]
        self.clip_embeddings = build_clip_embedding_cache(
            full_paths, cache_path, device=device
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_total_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor]:
        eeg = self.eeg_data[idx].copy()   # (n_channels, n_times)

        if self.apply_zscore:
            eeg = zscore_normalize(eeg)
        eeg = crop_epoch(eeg, self.times, self.t_start, self.duration)

        concept_idx = int(self.concept_indices[idx])
        clip_emb    = self.clip_embeddings[concept_idx % len(self.clip_embeddings)]

        return (
            torch.from_numpy(eeg.astype(np.float32)),
            concept_idx,
            torch.from_numpy(clip_emb),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        n_crop = int(self.duration * self.sfreq)
        return {
            "n_subjects"         : len(self.subjects),
            "n_total_samples"    : self.n_total_samples,
            "n_concepts_per_sub" : self.n_concepts_per_sub,
            "n_channels"         : self.n_channels,
            "sampling_rate_hz"   : self.sfreq,
            "time_range_s"       : f"{self.times[0]:.2f} – {self.times[-1]:.2f}",
            "n_times_raw"        : len(self.times),
            "n_times_cropped"    : n_crop,
            "crop_window_s"      : f"{self.t_start} – {self.t_start + self.duration}",
            "avg_reps"           : self.avg_reps,
            "clip_embed_dim"     : self.clip_embeddings.shape[-1],
            "split"              : self.split,
        }

    def check_nan_inf(self) -> None:
        """Scan full raw EEG array for NaN/Inf. Raises ValueError if found."""
        if np.any(np.isnan(self.eeg_data)):
            raise ValueError("NaN values detected in EEG data.")
        if np.any(np.isinf(self.eeg_data)):
            raise ValueError("Inf values detected in EEG data.")
        print("check_nan_inf passed — no NaN or Inf values found.")
