# EEG-FineTune

Fine-tune a CLIP foundation model to align EEG brain signals with semantic image embeddings. A user uploads a brain recording and gets back the closest matching visual concepts.

## Stack

- Python 3.10+
- PyTorch + torchvision
- MNE-Python (EEG preprocessing)
- OpenCLIP (ViT-B/32)
- FastAPI (backend, Phase 4)
- React + Vite (frontend, Phase 4)

## Dataset

[THINGS-EEG](https://osf.io/3jk45/) — EEG recorded while subjects viewed 22,248 object images across 10 subjects.

---

## Setup

### Local Development (code editing, no GPU needed)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab (data download + training)

Open `notebooks/phase1_setup.ipynb` in Colab. Cell 1 installs the required packages:

```python
!pip install -q -r requirements_colab.txt
```

The dataset (~15–20 GB) is downloaded once to Google Drive and persists across sessions.

---

## Project Structure

```
EEG-FineTune/
├── requirements.txt           # local dev
├── requirements_colab.txt     # Colab installs
├── src/
│   └── dataset.py             # ThingsEEGDataset — reusable across local + Colab
└── notebooks/
    └── phase1_setup.ipynb     # Phase 1: download, preprocess, sanity check
```

Data lives in Google Drive at `MyDrive/EEG-FineTune/data/things-eeg/` and is **not committed to git**.

---

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Data + environment | In progress |
| 2 | Signal encoder (EEGNet-style) | Pending |
| 3 | Contrastive CLIP fine-tuning | Pending |
| 4 | Webapp (FastAPI + React) | Pending |
