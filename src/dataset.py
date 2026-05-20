"""
ThingsEEGDataset — PyTorch Dataset for the THINGS-EEG dataset.

Expected data layout (under data_root):
    preprocessed_data/
        sub-01/
            sub-01_eeg_training.npy   # float32, shape (n_trials, n_channels, n_times)
            sub-01_eeg_test.npy
        sub-02/
            ...
    image_metadata.npy                # structured array with 'image_concept' and 'image_path' fields
    image_set/                        # JPEG stimulus images

The dataset returns (eeg_tensor, image_label, clip_embedding) tuples.
CLIP embeddings are encoded once and cached to <data_root>/embeddings_cache.npy.
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
    import mne
    mne.set_log_level("WARNING")
except ImportError as e:
    raise ImportError("mne is required: pip install mne") from e

try:
    import open_clip
except ImportError as e:
    raise ImportError("open-clip-torch is required: pip install open-clip-torch") from e

from PIL import Image


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def bandpass_filter(eeg: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 40.0) -> np.ndarray:
    """Apply zero-phase FIR bandpass filter to EEG array of shape (n_channels, n_times)."""
    return mne.filter.filter_data(
        eeg.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_window="hamming",
        verbose=False,
    ).astype(np.float32)


def zscore_normalize(eeg: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently over the time axis.
    Input shape: (n_channels, n_times)
    """
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero on flat channels
    return (eeg - mean) / std


def crop_epoch(eeg: np.ndarray, sfreq: float, t_start: float, duration_ms: float = 500.0) -> np.ndarray:
    """Crop epoch to [t_start, t_start + duration_ms] ms post-stimulus.
    Input shape: (n_channels, n_times)
    """
    start_sample = int(t_start * sfreq / 1000.0)
    n_samples = int(duration_ms * sfreq / 1000.0)
    end_sample = start_sample + n_samples
    return eeg[:, start_sample:end_sample]


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
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings.append(feats.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    cache = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(cache_path, cache)
    print(f"Cache saved to {cache_path}  shape={cache.shape}")
    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThingsEEGDataset(Dataset):
    """PyTorch Dataset wrapping the THINGS-EEG preprocessed data.

    Args:
        data_root:     Path to the root directory containing preprocessed_data/,
                       image_metadata.npy, and image_set/.
        subjects:      List of subject IDs as zero-padded strings, e.g. ["01", "02"].
        split:         "training" or "test".
        sfreq:         EEG sampling frequency in Hz (default 1000 Hz for THINGS-EEG).
        t_start_ms:    Start of post-stimulus window in ms (default 0).
        epoch_ms:      Duration of epoch window in ms (default 500).
        clip_device:   Device to run CLIP encoding on ("cpu" or "cuda").
        rebuild_cache: Force re-computation of CLIP embedding cache.
    """

    SFREQ_DEFAULT = 1000.0  # THINGS-EEG preprocessed sampling rate

    def __init__(
        self,
        data_root: str | Path,
        subjects: Sequence[str],
        split: str = "training",
        sfreq: float = SFREQ_DEFAULT,
        t_start_ms: float = 0.0,
        epoch_ms: float = 500.0,
        clip_device: str = "cpu",
        rebuild_cache: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.subjects = list(subjects)
        self.split = split
        self.sfreq = sfreq
        self.t_start_ms = t_start_ms
        self.epoch_ms = epoch_ms

        self._load_metadata()
        self._load_eeg()
        self._load_or_build_clip_cache(clip_device, rebuild_cache)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        meta_path = self.data_root / "image_metadata.npy"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"image_metadata.npy not found at {meta_path}. "
                "Run the download cell in notebooks/phase1_setup.ipynb first."
            )
        meta = np.load(meta_path, allow_pickle=True).item()

        # Support both structured-array and dict layouts used in THINGS-EEG releases
        if isinstance(meta, dict):
            self.image_concepts: list[str] = list(meta["image_concept"])
            self.image_files: list[str] = list(meta["image_path"])
        else:
            self.image_concepts = [str(c) for c in meta["image_concept"]]
            self.image_files = [str(p) for p in meta["image_path"]]

        self.n_images = len(self.image_concepts)

    def _load_eeg(self) -> None:
        """Load and concatenate EEG arrays for all requested subjects."""
        eeg_list: list[np.ndarray] = []
        trial_subject: list[str] = []

        for sub in self.subjects:
            sub_dir = self.data_root / "preprocessed_data" / f"sub-{sub}"
            npy_path = sub_dir / f"sub-{sub}_eeg_{self.split}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"EEG file not found: {npy_path}\n"
                    "Run the download cell in notebooks/phase1_setup.ipynb first."
                )
            arr = np.load(npy_path)  # (n_trials, n_channels, n_times)
            if arr.ndim != 3:
                raise ValueError(f"Expected 3-D array for {npy_path}, got shape {arr.shape}")
            eeg_list.append(arr)
            trial_subject.extend([sub] * len(arr))

        self.eeg_data: np.ndarray = np.concatenate(eeg_list, axis=0)  # (N, C, T)
        self.trial_subjects: list[str] = trial_subject
        self.n_trials, self.n_channels, self.n_times = self.eeg_data.shape

        # Image index per trial repeats for each subject (THINGS-EEG structure)
        trials_per_subject = self.n_trials // len(self.subjects)
        self.trial_image_idx: np.ndarray = np.tile(
            np.arange(trials_per_subject), len(self.subjects)
        )

    def _load_or_build_clip_cache(self, device: str, rebuild: bool) -> None:
        cache_path = self.data_root / "embeddings_cache.npy"
        if cache_path.exists() and not rebuild:
            self.clip_embeddings = np.load(cache_path)
            return

        image_set_dir = self.data_root / "image_set"
        if not image_set_dir.exists():
            raise FileNotFoundError(
                f"image_set/ directory not found at {image_set_dir}. "
                "Download stimulus images first."
            )
        full_paths = [str(image_set_dir / f) for f in self.image_files]
        self.clip_embeddings = build_clip_embedding_cache(
            full_paths, cache_path, device=device
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> tuple[Tensor, str, Tensor]:
        eeg = self.eeg_data[idx].copy()  # (n_channels, n_times)

        # Preprocessing
        eeg = bandpass_filter(eeg, self.sfreq)
        eeg = zscore_normalize(eeg)
        eeg = crop_epoch(eeg, self.sfreq, self.t_start_ms, self.epoch_ms)

        img_idx = self.trial_image_idx[idx]
        label = self.image_concepts[img_idx]
        clip_emb = self.clip_embeddings[img_idx]

        return (
            torch.from_numpy(eeg),
            label,
            torch.from_numpy(clip_emb),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict of dataset properties."""
        return {
            "n_subjects": len(self.subjects),
            "n_trials": self.n_trials,
            "n_channels": self.n_channels,
            "n_times_raw": self.n_times,
            "n_times_cropped": int(self.epoch_ms * self.sfreq / 1000.0),
            "sampling_rate_hz": self.sfreq,
            "epoch_window_ms": f"{self.t_start_ms}–{self.t_start_ms + self.epoch_ms}",
            "n_unique_images": self.n_images,
            "clip_embed_dim": self.clip_embeddings.shape[-1],
            "split": self.split,
        }

    def check_nan_inf(self) -> None:
        """Scan full raw EEG array for NaN/Inf values. Raises ValueError if found."""
        if np.any(np.isnan(self.eeg_data)):
            raise ValueError("NaN values detected in raw EEG data.")
        if np.any(np.isinf(self.eeg_data)):
            raise ValueError("Inf values detected in raw EEG data.")
        if np.any(np.isnan(self.clip_embeddings)):
            raise ValueError("NaN values detected in CLIP embeddings.")
        print("check_nan_inf passed — no NaN or Inf values found.")
