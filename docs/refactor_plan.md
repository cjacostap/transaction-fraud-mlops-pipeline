
"""
Refactorization Plan: Fraud Detection Deep Learning Pipeline
0. Current State Analysis
Notebook sections (19 total):
#	Section	Lines (approx)	Current Role
1	Importacion de Librerias	~50	Imports everything for the whole project
2	Configuracion Global del Proyecto	~50	Paths, CONFIG dict, directory creation
3	Pipeline de Feature Engineering	~200	FraudDetectionPipeline class (load, clean, features)
4	Carga y Preprocesamiento de Datos	~50	Runs the pipeline, extracts X and y
5	Analisis Exploratorio de Datos (EDA)	~170	Plots, distributions, correlations
6	Division de Datos (Train/Val/Test)	~60	train_test_split x2
7	Manejo Desbalanceo con SMOTE	~40	SMOTE on train set
8	Escalado de Features	~30	StandardScaler fit on train, transform all
9	Construccion del Modelo DL	~80	build_fraud_detection_model() function
10	Configuracion de Callbacks	~70	EarlyStopping, ReduceLR, ModelCheckpoint, TensorBoard
11	Entrenamiento del Modelo	~50	model.fit() with class_weights
12	Visualizacion del Entrenamiento	~80	Loss/metric curves
13	Evaluacion en Test Set	~50	model.evaluate() + predictions
14	Metricas Detalladas y Confusion Matrix	~140	Classification report, CM plot
15	Curvas ROC y Precision-Recall	~70	ROC, PR curves
16	Analisis de Umbral Optimo	~60	F1 vs threshold sweep
17	Guardar Modelo y Resultados	~120	Save .keras, .pkl, .json
18	Funcion de Prediccion	~80	predict_fraud() function
19	Conclusiones	narrative	Markdown only
1. Target Project Structure
fraud-detection-mlops/
  src/
    __init__.py
    data.py            # Ingestion + cleaning + feature engineering
    model.py           # Preprocessing pipeline, DL model, Optuna, final training
    validation.py      # Metrics, plots, evaluation logic
    main.py            # Single entry point (orchestrator)
  notebooks/
    eda.ipynb          # Exploratory analysis only
  configs/
    default.yaml       # Default hyperparameters + paths
  data/
    raw/               # onlinefraud.csv lives here
  outputs/
    models/            # Final artifacts only
    reports/           # Metrics JSON, executive summary
    figures/           # Evaluation plots
  requirements.txt
  README.md
fraud-detection-mlops/  src/    __init__.py    data.py            # Ingestion + cleaning + feature engineering    model.py           # Preprocessing pipeline, DL model, Optuna, final training    validation.py      # Metrics, plots, evaluation logic    main.py            # Single entry point (orchestrator)  notebooks/    eda.ipynb          # Exploratory analysis only  configs/    default.yaml       # Default hyperparameters + paths  data/    raw/               # onlinefraud.csv lives here  outputs/    models/            # Final artifacts only    reports/           # Metrics JSON, executive summary    figures/           # Evaluation plots  requirements.txt  README.md
2. Section-to-Module Mapping
Notebook Section	Target Module	Notes
1. Imports	Distributed	Each module imports only what it needs
2. Config Global	main.py + configs/default.yaml	Config loaded/parsed only in main.py
3. Feature Engineering Pipeline	data.py	Core of data.py. Refactor FraudDetectionPipeline class
4. Carga y Preprocesamiento	data.py	Becomes the public API of data.py
5. EDA	notebooks/eda.ipynb	Move as-is; import from data.py for loading only
6. Data Split	main.py	Orchestrator owns the split strategy
7. SMOTE	model.py	SMOTE is model-specific preprocessing
8. Feature Scaling	model.py	Scaler is part of the preprocessing pipeline
9. Model Architecture	model.py	build_model() function
10. Callbacks	model.py	Part of train_model()
11. Training	model.py	train_model() function
12. Training Visualization	validation.py	plot_training_history()
13. Test Evaluation	validation.py	evaluate_model()
14. Metrics + Confusion Matrix	validation.py	compute_metrics(), plot_confusion_matrix()
15. ROC / PR Curves	validation.py	plot_roc_curve(), plot_pr_curve()
16. Threshold Optimization	validation.py	find_optimal_threshold()
17. Save Model	model.py (save) + main.py (orchestration)	save_artifacts()
18. Prediction Function	model.py	predict() -- needed for future API serving
19. Conclusions	notebooks/eda.ipynb or README	Narrative only
3. Detailed Module Design
3.1 data.py -- Ingestion + Cleaning + Feature Engineering
Source material: Sections 3 and 4 of the notebook.
Public API:
def load_raw_data(path: str) -> pd.DataFrame
def clean_data(df: pd.DataFrame) -> pd.DataFrame
def engineer_features(df: pd.DataFrame) -> pd.DataFrame
def prepare_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]
    """Full pipeline: load -> clean -> features -> (X, y)"""
def load_raw_data(path: str) -> pd.DataFramedef clean_data(df: pd.DataFrame) -> pd.DataFramedef engineer_features(df: pd.DataFrame) -> pd.DataFramedef prepare_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]    """Full pipeline: load -> clean -> features -> (X, y)"""
What goes IN:
load_data() -- CSV/Parquet loading, memory optimization
remove_leakage_columns() -- extract target before anything else
create_missing_flags() -- model-agnostic missing indicators
handle_outliers() -- winsorization on balance columns
create_balance_features() -- balance deltas, inconsistencies, emptied_account, dest_account_new
create_amount_features() -- log1p_amount, amount ratios, round-amount flag
create_temporal_features() -- hour_of_day, day_of_month, is_night, is_weekend
create_log_transformations() -- log1p of balance columns
Drop raw ID columns (nameOrig, nameDest)
What MUST be removed from data.py:
One-hot encoding of type column (currently create_transaction_type_features). This IS encoding and belongs in model.py. data.py should preserve type as a categorical string column.
No StandardScaler, no RobustScaler, no SMOTE -- all are model-specific.
Key design decisions:
Refactor from a stateful class (FraudDetectionPipeline with self.df_raw, self.df_clean, etc.) into pure stateless functions. A class with mutable state is unnecessary here and makes testing harder.
Each function takes a DataFrame and returns a new DataFrame (functional style, no side effects).
prepare_dataset() is the single public entry point that chains them all.
Target column isFraud is separated and returned alongside the feature DataFrame.
Anti-patterns to fix:
The current load_data() has a bare except: clause -- replace with explicit exception types.
The pipeline loads data twice (once in __init__ via load_data and again in clean_data). Fix this.
3.2 model.py -- Preprocessing + DL Model + Optuna + Training
Source material: Sections 7, 8, 9, 10, 11, 17, 18.
Public API:
def build_preprocessing_pipeline(config: dict) -> sklearn.pipeline.Pipeline
def build_model(input_dim: int, config: dict) -> keras.Model
def optuna_objective(trial, X_train, y_train, config: dict) -> float
def run_tuning(X_train, y_train, config: dict, stage: str) -> dict
def train_final_model(X_train, y_train, X_val, y_val, config: dict) -> tuple[keras.Model, dict]
def save_artifacts(model, preprocessing_pipeline, config, output_dir: str)
def load_artifacts(model_dir: str) -> tuple[keras.Model, Pipeline, dict]
def predict(raw_features: pd.DataFrame, model, pipeline, threshold: float) -> dict
def build_preprocessing_pipeline(config: dict) -> sklearn.pipeline.Pipelinedef build_model(input_dim: int, config: dict) -> keras.Modeldef optuna_objective(trial, X_train, y_train, config: dict) -> floatdef run_tuning(X_train, y_train, config: dict, stage: str) -> dictdef train_final_model(X_train, y_train, X_val, y_val, config: dict) -> tuple[keras.Model, dict]def save_artifacts(model, preprocessing_pipeline, config, output_dir: str)def load_artifacts(model_dir: str) -> tuple[keras.Model, Pipeline, dict]def predict(raw_features: pd.DataFrame, model, pipeline, threshold: float) -> dict
Preprocessing pipeline (sklearn):
Encoding: ColumnTransformer with OneHotEncoder for the type column (currently hardcoded binary columns in data.py).
Scaling: StandardScaler (default) or RobustScaler, configurable.
SMOTE: Applied only during training (NOT inside the pipeline object, but as a training step). Reason: SMOTE must not be applied to validation/test data, and must be applied after the scaler to avoid leaking fit statistics.
The preprocessing pipeline (encoder + scaler) must be fitted only on training data and serialized alongside the model.
Model architecture (build_model):
Migrate the existing build_fraud_detection_model() function almost as-is.
Parameterize: hidden layer sizes, dropout rate, L2 regularization, learning rate, activation function -- all from config/Optuna trial.
Compile inside the function.
Optuna tuning strategy (two-stage):
Stage 1 -- Coarse search:
Wide ranges for: number of layers (2-5), units per layer (32-512), dropout (0.1-0.5), L2 reg (1e-5 to 1e-2), learning rate (1e-5 to 1e-2), batch size (256, 512, 1024), SMOTE sampling strategy (0.1-0.5).
Fewer trials (e.g., 20-30), fewer epochs per trial (e.g., 10-15 with early stopping).
Objective: maximize AUC-PR on validation set (more appropriate than F1 for imbalanced data).
Stage 2 -- Fine tuning:
Narrow ranges centered around Stage 1 best params (+/- 20-30% of each value).
More trials (e.g., 30-50), more epochs per trial (e.g., 20-30).
Same objective metric.
When to use single-stage vs two-stage:
Single-stage: If the search space is already narrow (e.g., you have strong priors from domain knowledge or prior experiments). Also acceptable for fast iteration during development.
Two-stage: Default recommendation for production. Avoids wasting compute on fine-grained search across an enormous space.
Cross-validation inside Optuna (critical consideration):
The dataset has 6.3M rows. Full K-fold CV with DL training is expensive.
Recommended approach: Use a single train/val split inside Optuna (not K-fold), but ensure the split is stratified and consistent across trials (fix the random seed for the split). This is pragmatic for DL at this scale.
If higher rigor is required: Subsample to ~500K rows, then use 3-fold StratifiedKFold inside the Optuna objective. Report mean AUC-PR across folds.
Never: Apply SMOTE before the split inside the objective. Always split first, then SMOTE only on the training fold.
Final training:
After Optuna finds the best hyperparameters, train a completely new model from scratch.
Use all available training data (train + validation combined) for final fitting.
The test set remains completely untouched until validation.py evaluates the final artifact.
Save ONLY this final model. Intermediate Optuna trial models should be discarded.
Save: .keras model, preprocessing pipeline (joblib), feature names, config used, optimal threshold.
3.3 validation.py -- Metrics, Plots, Evaluation
Source material: Sections 12, 13, 14, 15, 16.
Public API:
def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict
def compute_classification_metrics(y_true, y_pred, y_proba) -> dict
def find_optimal_threshold(y_true, y_proba, metric: str = "f1") -> tuple[float, float]
def plot_training_history(history: dict, output_dir: str)
def plot_confusion_matrix(y_true, y_pred, output_dir: str)
def plot_roc_curve(y_true, y_proba, output_dir: str)
def plot_pr_curve(y_true, y_proba, output_dir: str)
def plot_threshold_analysis(y_true, y_proba, output_dir: str)
def generate_evaluation_report(metrics: dict, output_dir: str)
def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dictdef compute_classification_metrics(y_true, y_pred, y_proba) -> dictdef find_optimal_threshold(y_true, y_proba, metric: str = "f1") -> tuple[float, float]def plot_training_history(history: dict, output_dir: str)def plot_confusion_matrix(y_true, y_pred, output_dir: str)def plot_roc_curve(y_true, y_proba, output_dir: str)def plot_pr_curve(y_true, y_proba, output_dir: str)def plot_threshold_analysis(y_true, y_proba, output_dir: str)def generate_evaluation_report(metrics: dict, output_dir: str)
Key principles:
Read-only: This module NEVER trains models, NEVER modifies data.
All plotting functions save to disk (to output_dir) AND optionally return the figure for notebook use.
evaluate_model() is the single entry point called from main.py. It calls all sub-functions internally and produces the full report.
Metrics dict includes: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix components (TP/FP/TN/FN), optimal threshold.
3.4 main.py -- Single Entry Point / Orchestrator
Source material: Sections 2, 6 (data split), 17 (orchestration of save).
Structure:
def parse_config() -> dict:
    """Load from YAML, override with CLI args."""

def setup_logging(config: dict):
    """Configure Python logging module."""

def main(config: dict):
    """
    1. Load & process data        (data.py)
    2. Split train/val/test       (here)
    3. Run Optuna tuning          (model.py)
    4. Train final model          (model.py)
    5. Evaluate on test set       (validation.py)
    6. Save artifacts             (model.py)
    7. Log summary
    """

if __name__ == "__main__":
    config = parse_config()
    setup_logging(config)
    main(config)
def parse_config() -> dict:    """Load from YAML, override with CLI args."""def setup_logging(config: dict):    """Configure Python logging module."""def main(config: dict):    """    1. Load & process data        (data.py)    2. Split train/val/test       (here)    3. Run Optuna tuning          (model.py)    4. Train final model          (model.py)    5. Evaluate on test set       (validation.py)    6. Save artifacts             (model.py)    7. Log summary    """if __name__ == "__main__":    config = parse_config()    setup_logging(config)    main(config)
Only main.py contains if __name__ == "__main__".
Configuration management:
Use a configs/default.yaml file for all hyperparameters, paths, and flags.
Use argparse for CLI overrides (e.g., --config path/to/custom.yaml, --seed 42, --skip-tuning).
The resolved config dict is passed to every module function. No module reads config files directly.
4. Data Split Strategy and Leakage Prevention
This is the most critical correctness concern. The current notebook has potential leakage issues.
Correct split flow:
Raw Data (6.3M rows)
      |
      v
  data.py: load -> clean -> feature engineering
      |
      v
  X, y (full dataset, 34 features + target)
      |
      v
  main.py: stratified split
      |
      +---> X_train+val, y_train+val (80%)
      |         |
      |         +---> [Optuna uses this portion]
      |         |         - Internal stratified split: train (85%) / val (15%)
      |         |         - SMOTE applied only on train fold
      |         |         - Scaler fitted only on train fold
      |         |
      |         +---> [Final training uses ALL of this]
      |                   - SMOTE on full train+val
      |                   - Scaler fitted on full train+val
      |                   - New model from scratch with best params
      |
      +---> X_test, y_test (20%) -- NEVER touched until final evaluation
Raw Data (6.3M rows)      |      v  data.py: load -> clean -> feature engineering      |      v  X, y (full dataset, 34 features + target)      |      v  main.py: stratified split      |      +---> X_train+val, y_train+val (80%)      |         |      |         +---> [Optuna uses this portion]      |         |         - Internal stratified split: train (85%) / val (15%)      |         |         - SMOTE applied only on train fold      |         |         - Scaler fitted only on train fold      |         |      |         +---> [Final training uses ALL of this]      |                   - SMOTE on full train+val      |                   - Scaler fitted on full train+val      |                   - New model from scratch with best params      |      +---> X_test, y_test (20%) -- NEVER touched until final evaluation
Explicit leakage checks:
SMOTE is applied after the split, only on training data. Never on val/test.
Scaler is fitted on training data only. Val/test are transformed with the same scaler.
During Optuna, each trial gets a fresh scaler fitted on its training fold.
The final model uses a scaler fitted on the full train+val set.
Test predictions use the final scaler (fitted on train+val).
No feature engineering step uses information from the target column or from future rows.
5. Anti-Patterns to Avoid
Anti-Pattern	Where It Exists Now	Fix
Bare except: clause	data.py (load_data)	Use explicit exception types
Data loaded twice	FraudDetectionPipeline.__init__ + clean_data	Load once in load_raw_data()
One-hot encoding in feature engineering	create_transaction_type_features	Move to model.py preprocessing pipeline
Print statements for logging	Everywhere	Replace with logging module
CONFIG as a global dict	Section 2	YAML config file + argparse
SMOTE before any validation split	Risk if not careful	Enforce SMOTE only after split, only on train
Hardcoded paths	DATA_PATH = "../data/raw/..."	Config-driven paths
No reproducibility seed management	SEED = 42 scattered	Single seed in config, set once in main.py
model.save() during training (checkpoint) vs final	ModelCheckpoint saves intermediate models	Only save the truly final model in production artifacts
class_weight AND SMOTE simultaneously	Section 11 uses both	Pick one or the other; using both overcompensates for imbalance
No Optuna integration	Hyperparams are static	Add Optuna as designed above
Stateful pipeline class	FraudDetectionPipeline with mutable self.df_*	Refactor to stateless functions
6. Step-by-Step Refactorization Sequence
Phase 1: Foundation (do first)
Create the directory structure: src/, configs/, outputs/, etc.
Create configs/default.yaml: Extract all CONFIG values, paths, and hyperparameter ranges from the notebook.
Create src/__init__.py: Empty or with version info.
Phase 2: data.py (independent, no other module depends on it yet)
Extract data.py: Migrate FraudDetectionPipeline methods as stateless functions. Remove one-hot encoding of type. Ensure prepare_dataset() returns (X, y) with type still as a string column.
Test data.py: Run prepare_dataset() and verify output shape, dtypes, no NaN surprises.
Phase 3: model.py (depends on data.py output shape)
Build preprocessing pipeline: ColumnTransformer with OneHotEncoder for type + StandardScaler for numerical columns.
Migrate build_model(): Parameterize fully from config.
Implement Optuna objective: Single train/val split, SMOTE on train only, scaler fitted on train only.
Implement run_tuning(): Coarse and fine stages.
Implement train_final_model(): From scratch on train+val with best params.
Implement save_artifacts() and load_artifacts(): Serialize model, pipeline, config, feature names.
Implement predict(): Load artifacts, preprocess, predict.
Phase 4: validation.py (independent of training logic)
Migrate all evaluation functions: metrics computation, all 5 plot types, threshold analysis, report generation.
Ensure validation.py is purely read-only: Takes y_true, y_pred, y_proba -- never touches the model internals.
Phase 5: main.py (orchestrator, depends on all other modules)
Implement config loading: YAML + argparse.
Implement logging setup: Python logging module with file + console handlers.
Implement main(): Wire together data -> split -> tuning -> training -> evaluation -> save.
Set reproducibility seeds: numpy, tensorflow, Python hash seed -- all from config, set once.
Phase 6: notebooks/eda.ipynb
Create eda.ipynb: Import from src.data import prepare_dataset. Keep all Section 5 plots and exploration. Add any additional analysis (distributions, correlations, class balance visualization).
Ensure EDA is NOT required: The notebook should never be run as part of the training pipeline. It's for human insight only.
Phase 7: Hardening
Add requirements.txt: Pin versions for tensorflow, numpy, pandas, scikit-learn, imbalanced-learn, optuna, pyyaml, joblib, matplotlib, seaborn.
Validate end-to-end: Run python src/main.py and verify the full pipeline works.
Verify no test set leakage: Add explicit assertions or logging that confirm test set is untouched until final evaluation.
7. Improvement Suggestions
Logging
Replace all print() with logging.info() / logging.debug().
Configure structured logging (JSON format) for future integration with monitoring systems (ELK, CloudWatch, etc.).
Log: data shapes at each step, training loss/metrics per epoch, Optuna trial results, final evaluation metrics, artifact paths.
Configuration
Use YAML (not JSON or Python dicts). YAML supports comments, is human-readable, and is the de facto standard for ML configs.
Support CLI overrides for quick experiments: python src/main.py --config configs/experiment_v2.yaml --seed 123.
Save the resolved config alongside the model artifact for full reproducibility.
Reproducibility
Set PYTHONHASHSEED, numpy.random.seed, tf.random.set_seed -- all from a single config seed.
Pin all dependency versions in requirements.txt.
Save the exact config used for each training run as a JSON alongside the model.
Consider adding a git_commit_hash to the saved metadata.
Deployment Readiness
predict() in model.py accepts a raw DataFrame (post-data.py processing) and returns structured output. This is the function a FastAPI endpoint would call.
The preprocessing pipeline + model are serialized together, ensuring no "forgotten scaler" issues.
main.py is container-friendly: no interactive input, no GUI, no notebook dependencies, exit code 0 on success.
Future: Add a Dockerfile that runs python src/main.py for batch execution, or a serve.py with FastAPI for API mode.
Class Imbalance Strategy
The current notebook uses BOTH class_weight and SMOTE. This double-compensates and can degrade precision. Pick one: SMOTE for oversampling OR class_weight for loss weighting. Make this a config option and let Optuna choose which strategy works best.
Threshold as a First-Class Artifact
The optimal threshold found during validation should be saved alongside the model and used at inference time. Currently it's computed but not persisted in a usable way. Save it in the config/metadata JSON.
This plan gives you a clear, actionable roadmap. Each phase is independent enough to implement and validate before moving to the next. The critical correctness guarantees (no leakage, proper splits, final retraining from scratch) are built into the design from the start.
Ready to start implementing when you give the go-ahead.


"""