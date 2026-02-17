# CLI Execution Examples

This document provides practical examples for running the fraud detection ML pipeline using the CLI.

---

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify data path:**
   Ensure the data file exists at `fraud-detection-mlops/data/raw/onlinefraud.csv` or update the path in your config.

---

## Basic Usage

### 1. Train with Default Configuration

Run the complete pipeline (data prep → train → evaluate → save artifacts) using all default settings:

```bash
python -m src.main
```

**What happens:**
- Loads config from `configs/default.yaml`
- Trains final model without hyperparameter tuning
- Saves model artifacts to `outputs/models/`
- Generates evaluation reports in `outputs/reports/`
- Creates plots in `outputs/figures/`

---

### 2. Train with Custom Data Path

```bash
python -m src.main --data-path /path/to/your/data.csv
```

---

### 3. Custom Output Directory

```bash
python -m src.main --output-dir outputs/models_run_001
```

---

### 4. Reproducibility Seed

```bash
python -m src.main --seed 42
```

---

### 5. Training Speed vs. Quality

```bash
python -m src.main --epochs 30 --batch-size 512
```

---

### 6. Skip Evaluation (Quick Smoke Test)

```bash
python -m src.main --skip-evaluation
```

---

## Key Feature Flags

### Enable Optuna Tuning

```bash
python -m src.main --use-optuna
```

Customize Optuna:

```bash
python -m src.main \
  --use-optuna \
  --optuna-n-trials-coarse 10 \
  --optuna-n-trials-fine 20 \
  --optuna-cv-folds 3 \
  --optuna-subsample 200000 \
  --optuna-epochs 10
```

---

### Imbalance Strategy

```bash
python -m src.main --use-smote
```

```bash
python -m src.main --use-class-weights
```

Adjust SMOTE:

```bash
python -m src.main --use-smote --smote-sampling-strategy 0.25
```

---

### Drop Highly Correlated Features

```bash
python -m src.main --drop-collinear --corr-threshold 0.95
```

```bash
python -m src.main --drop-collinear --corr-threshold 0.90 --corr-sample-size 200000
```

Exclude features:

```bash
python -m src.main --drop-collinear --corr-exclude amount log1p_amount
```

---

### Model Architecture and Training

```bash
python -m src.main \
  --hidden-layers 512 256 128 64 \
  --dropout-rate 0.4 \
  --l2-reg 0.0005 \
  --learning-rate 0.0005
```

---

## Using Configuration Files (Recommended)

Create `configs/experiment_001.yaml`:

```yaml
data_path: fraud-detection-mlops/data/raw/onlinefraud.csv
output_root: experiments/exp_001
output_dir: experiments/exp_001/models

# Data splits
test_size: 0.20
val_size: 0.15
random_state: 42

# Class imbalance
use_smote: true
smote_sampling_strategy: 0.30
use_class_weights: false

# Model architecture
hidden_layers: [512, 256, 128, 64]
dropout_rate: 0.35
l2_reg: 0.0008
activation: relu
learning_rate: 0.0008

# Training
batch_size: 1024
epochs: 80
patience_early_stop: 20

# Optuna
use_optuna: true
optuna_n_trials_coarse: 25
optuna_n_trials_fine: 40
optuna_cv_folds: 3

# Collinearity
drop_collinear: true
corr_threshold: 0.92
corr_sample_size: 200000
```

Run with config:

```bash
python -m src.main --config configs/experiment_001.yaml
```

Override config values:

```bash
python -m src.main \
  --config configs/experiment_001.yaml \
  --epochs 120 \
  --batch-size 2048
```

---

## Monitoring Training

### Summary Report

Saved to:

- `{output_root}/reports/training_summary.txt`

### TensorBoard (Optional)

TensorBoard logs are saved under the model output directory:

```bash
tensorboard --logdir outputs/models/tensorboard_logs
```

---

## Deployment Preparation

```python
from src.model import load_artifacts
from src.predict import predict

model, pipeline, feature_names, config = load_artifacts("outputs/models")
predictions = predict(model, pipeline, X_new, feature_names)
```

---

## Summary

- **Quick start:** `python -m src.main`
- **Custom config:** `python -m src.main --config configs/my_config.yaml`
- **Key overrides:** `--data-path`, `--output-dir`, `--epochs`, `--batch-size`, `--seed`

For more details, see:
- `docs/refactor_plan.md` - Complete architecture documentation
- `configs/default.yaml` - All available parameters
- `src/main.py` - CLI arguments and orchestration
