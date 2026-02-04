# digit-recognizer-cv
A CNN Computer Vision Digit Recognizer (MNIST) Pipeline

PyTorch solution for Kaggle’s **Digit Recognizer** competition using a compact CNN and **Optuna** hyperparameter tuning.  
The workflow includes: data loading → preprocessing/normalization → optional affine augmentation → tuning → cross-validation → final training → `submission.csv`.

---

## Competition

- **Kaggle**: Digit Recognizer (MNIST digits 0–9)
- Goal: classify handwritten digits from 28×28 grayscale images.

---

## What’s in this repo

- `model.ipynb`: end-to-end notebook (tuning + CV + final training + submission generation)
- `data/`: Kaggle CSVs (`train.csv`, `test.csv`, `sample_submission.csv`)
- `outputs/submission.csv`: produced after running the final cell
- `requirements.txt`: Python dependencies

---

## Approach

### Data format
- `train.csv`: first column is `label`, remaining 784 columns are pixel intensities (0–255).
- `test.csv`: 784 pixel columns (no label).

### Preprocessing
- Reshape into NCHW tensors: `(N, 1, 28, 28)`
- Convert to float and normalize with **train-set mean/std**:
  - `x = x/255.0`
  - `x = (x - PIX_MEAN) / PIX_STD`

### Model (CNNNet)
A simple CNN backbone repeated for `n_blocks` blocks:

Each block:
- Conv(3×3) → BN → ReLU
- Conv(3×3) → BN → ReLU
- MaxPool(2)
- Dropout

Head:
- AdaptiveAvgPool2d(1) → Flatten
- Linear → ReLU → Dropout
- Linear → 10 classes

Key tunables:
- `base_ch`, `n_blocks`, `dropout`, `head_dim`

### Training
- Optimizer: **AdamW**
- Scheduler: **CosineAnnealingLR**
- Mixed precision on CUDA (`torch.amp.autocast` + GradScaler)
- Gradient clipping (`clip_grad_norm_`)
- Optional label smoothing (`F.cross_entropy(..., label_smoothing=...)`)

### Augmentation (optional)
If `torchvision` is available:
- `RandomAffine` (rotation/translation/scale)
- Default “sane” settings for CV/final training; tuning optionally supported.

### Hyperparameter tuning (Optuna)
- Uses a stratified train/validation split for trial evaluation
- Early stopping with patience during training inside each trial

### Cross-validation
- StratifiedKFold with `n_folds=5`
- Reports best validation accuracy per fold and mean ± std

### Submission
Final model is trained on the full training set using the best Optuna parameters, then predicts test labels and writes:
- `submission.csv` with columns: `ImageId`, `Label`

---

## Results
This notebook prints:
- Optuna best parameters (`study.best_params`)
- 5-fold CV best accuracy per fold + mean ± std
- Final training loss per epoch
- A preview of the generated submission

> Exact leaderboard score depends on the Optuna search outcome and run-to-run randomness.

---

## Quickstart

### 1) Setup environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
