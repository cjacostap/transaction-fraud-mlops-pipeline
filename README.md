# Transaction Fraud Detection — MLOps Pipeline

ML pipeline for **transaction fraud detection** with an MLOps focus: experiment tracking with MLflow, data versioning with DVC, YAML-based configuration, and artifacts ready for registry and deployment.

## Objective

Train and evaluate a neural network (TensorFlow/Keras) that classifies transactions as legitimate or fraudulent, with:

- **Reproducibility**: YAML config, fixed seed, DVC for data.
- **Traceability**: MLflow for experiments, parameters, metrics, and model registry.
- **Flexibility**: CLI with many overrides, Optuna for tuning, SMOTE/collinearity options.

## Data

This project uses the **Kaggle Credit Card Fraud Dataset**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## Repository structure

```text
.
├── src/
│   ├── main.py              # Main entry: training pipeline
│   ├── data.py              # Dataset loading and preparation
│   ├── model.py             # Architecture, training, Optuna, artifact saving
│   ├── validation.py        # Evaluation (metrics, curves, optimal threshold)
│   ├── predict.py           # Inference (batch and single) for API/batch use
│   ├── mlflow_integration.py # MLflow logging (params, metrics, artifacts, tags)
│   └── mlflow_pyfunc_wrapper.py # PyFunc wrapper to register model + pipeline in MLflow
├── configs/
│   └── default.yaml        # Default config (data, model, MLflow, registry)
├── data/
│   └── raw/
│       └── onlinefraud.csv  # Dataset (versioned with DVC)
├── outputs/                 # Local outputs (models, reports, figures)
│   ├── models/
│   ├── reports/
│   └── figures/
├── infra/
│   ├── docker-compose.yaml  # MLflow + PostgreSQL + MinIO
│   └── Dockerfile.mlflow    # MLflow server image
├── notebooks/
│   └── eda.ipynb            # Exploratory analysis
├── docs/
│   ├── cli_examples.md      # CLI usage examples
│   └── refactor_plan.md     # Architecture documentation
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Optional: Docker and Docker Compose for MLflow + MinIO + PostgreSQL

## Installation

```bash
git clone <repo-url>
cd transaction-fraud-mlops-pipeline
pip install -r requirements.txt
```

## Quick start

### Train with default configuration

```bash
python -m src.main
```

- Loads `configs/default.yaml`
- Reads data from `data/raw/onlinefraud.csv` (or the configured path)
- Trains the model, evaluates on the test set, and saves artifacts to `outputs/`
- If `model_registry.enabled` is `true`, registers the model in MLflow Model Registry (PyFunc: model + preprocessing pipeline)

### MLflow infrastructure (optional)

To use MLflow with a PostgreSQL backend and MinIO for artifacts:

```bash
cd infra
cp .env.example .env   # Adjust variables if needed
docker compose up -d
```

Then point the client to `http://localhost:5001` (in `configs/default.yaml`: `mlflow.tracking_uri`).

### CLI examples

- Different data file: `python -m src.main --data-path path/to/data.csv`
- More epochs: `python -m src.main --epochs 80`
- Enable Optuna: `python -m src.main --use-optuna`
- SMOTE: `python -m src.main --use-smote --smote-sampling-strategy 0.3`
- Drop highly correlated features: `python -m src.main --drop-collinear --corr-threshold 0.95`
- Skip evaluation (quick smoke test): `python -m src.main --skip-evaluation`

More examples in `docs/cli_examples.md`.

## What the pipeline includes

- **Data**: CSV loading, stratified train/val/test splits, optional drop of features by correlation.
- **Imbalance**: Configurable SMOTE and/or class weights.
- **Model**: Neural network (hidden layers, dropout, L2, early stopping, ReduceLROnPlateau).
- **Tuning**: Optional Optuna (coarse + fine, configurable metric, e.g. AUC-PR).
- **Evaluation**: Precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix, optimal threshold, curves and figures.
- **MLflow**: Parameters, metrics, tags (e.g. `data_hash` when using DVC), data statistics, model registration as PyFunc (model + pipeline) with name and description.
- **Local artifacts**: Keras model, preprocessing pipeline (pickle), feature names, training config, reports and plots.

## Inference

The `src.predict` module uses the saved model and pipeline for predictions (batch or single transaction), ready to plug into an API (e.g. FastAPI) or batch jobs:

```python
from src.model import load_artifacts
from src.predict import predict

model, pipeline, feature_names, config = load_artifacts("outputs/models")
results = predict(model, pipeline, X_new, threshold=0.5)
# results["predictions"], results["probabilities"], results["risk_levels"]
```

## Further documentation

- `docs/cli_examples.md` — Detailed CLI and config examples.
- `docs/refactor_plan.md` — Architecture and refactor plan.
- `configs/default.yaml` — All available options.

## License

See `LICENSE` in the repository.
