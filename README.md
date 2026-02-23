# Galaxy Zoo Morphology (3-Class) — ResNet18 + “Max GPU, No OOM”

Classify Galaxy Zoo 2 images into **elliptical**, **spiral**, or **edge-on disk** using **ResNet18** (PyTorch).  
Includes a training loop that **automatically finds a VRAM-safe micro-batch** and then pushes GPU utilization hard **without crashing**.

## Highlights
- **Test accuracy:** **93.18%** (3-class, high-confidence labels)
- **Best val loss:** **0.1701**
- **Stable high-VRAM training:** ~**70–76 GB** allocated on an A100 80GB (no OOM)
- **Safety mechanism:** auto micro-batch search + AMP + `channels_last` + OOM spike handling

## Task
Given a galaxy image, predict:
- `0` → elliptical
- `1` → spiral
- `2` → edge_on_disk

## Data
This repo does not include the full image dataset (too large for GitHub). The notebook downloads it.

Sources:
- Galaxy Zoo 2 images (Kaggle dataset)
- Hart et al. GZ2 debiased vote fractions (`gz2_hart16`)

## Labeling (high-confidence)
We build labels from debiased vote fractions with a threshold **T = 0.8**, after filtering:
- `total_classifications ≥ 10`
- `star_or_artifact < 0.5`

Rules:
- **elliptical** if `smooth ≥ T`
- **edge_on_disk** if `disk ≥ T` and `edgeon_yes ≥ T`
- **spiral** if `disk ≥ T` and `edgeon_no ≥ T` and `spiral ≥ T`

Then we train only on label ids `{0,1,2}`.

## Model + Training
- Backbone: `torchvision.models.resnet18`
- Input: `224×224`
- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-2)`
- Imbalance: `WeightedRandomSampler`

### “Max GPU, No OOM” loop
The notebook contains a training cell that:
- probes for the **largest micro-batch** that fits in VRAM (with a safety margin)
- uses **AMP** (bf16 if supported), **TF32**, and `channels_last`
- uses grad accumulation to scale effective batch size without extra VRAM
- catches rare OOM spikes and continues safely

## Results (example run)
- **VAL:** best loss `0.1701`
- **TEST:** loss `0.1778`, accuracy `0.9318`

## Experiment Tracking
Weights & Biases report PDF (GPU utilization, power, VRAM allocation, system stats):
- `reports/galaxy_cnn_wandb_report.pdf`

## How to run (Colab)
Open `Galaxy.ipynb` and run cells top-to-bottom:
1) download images + Hart labels  
2) build `galaxy_level2.csv`  
3) train (safe GPU-max cell)  
4) eval (confusion matrix + classification report + mistakes)

## Repo Structure
- `Galaxy.ipynb` — end-to-end pipeline
- `reports/galaxy_cnn_wandb_report.pdf` — W&B run report
- `requirements.txt` — minimal deps (optional)

## Notes / Limitations
- Labels are **high-confidence filtered**, so this is “clean subset” performance, not raw-noisy Galaxy Zoo labels.
- For a stronger paper-style result, report per-class precision/recall/F1 (especially edge_on_disk recall) and include the confusion matrix image in `assets/`.
