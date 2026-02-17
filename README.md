# Transaction Fraud Detection — MLOps Pipeline

Pipeline de ML para **detección de fraude en transacciones** con enfoque MLOps: experimentación trazada con MLflow, versionado de datos con DVC, configuración por YAML, y artefactos listos para registro y despliegue.

## Objetivo

Entrenar y evaluar un modelo de red neuronal (TensorFlow/Keras) que clasifica transacciones como legítimas o fraudulentas, con:

- **Reproducibilidad**: configuración en YAML, semilla fija, DVC para datos.
- **Trazabilidad**: MLflow para experimentos, parámetros, métricas y registro de modelos.
- **Flexibilidad**: CLI con muchos overrides, Optuna para tuning, SMOTE/collinearity como opciones.

## Estructura del repositorio

```text
.
├── src/
│   ├── main.py              # Entrada principal: pipeline de entrenamiento
│   ├── data.py              # Carga y preparación del dataset
│   ├── model.py             # Arquitectura, entrenamiento, Optuna, guardado de artefactos
│   ├── validation.py        # Evaluación (métricas, curvas, umbral óptimo)
│   ├── predict.py           # Inferencia (batch y single) para uso en API/batch
│   ├── mlflow_integration.py # Logging a MLflow (params, métricas, artefactos, tags)
│   └── mlflow_pyfunc_wrapper.py # Wrapper PyFunc para registrar modelo + pipeline en MLflow
├── configs/
│   └── default.yaml        # Configuración por defecto (datos, modelo, MLflow, registry)
├── data/
│   └── raw/
│       └── onlinefraud.csv  # Dataset (versionado con DVC)
├── outputs/                 # Salidas locales (modelos, reportes, figuras)
│   ├── models/
│   ├── reports/
│   └── figures/
├── infra/
│   ├── docker-compose.yaml  # MLflow + PostgreSQL + MinIO
│   └── Dockerfile.mlflow    # Imagen del servidor MLflow
├── notebooks/
│   └── eda.ipynb            # Análisis exploratorio
├── docs/
│   ├── cli_examples.md      # Ejemplos de uso del CLI
│   └── refactor_plan.md     # Documentación de arquitectura
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Opcional: Docker y Docker Compose para MLflow + MinIO + PostgreSQL

## Instalación

```bash
git clone <url-del-repositorio>
cd transaction-fraud-mlops-pipeline
pip install -r requirements.txt
```

## Uso rápido

### Entrenar con la configuración por defecto

```bash
python -m src.main
```

- Carga `configs/default.yaml`
- Lee datos desde `data/raw/onlinefraud.csv` (o el path configurado)
- Entrena el modelo, evalúa en test y guarda artefactos en `outputs/`
- Si `model_registry.enabled` está en `true`, registra el modelo en MLflow Model Registry (PyFunc: modelo + pipeline de preprocesamiento)

### Infraestructura MLflow (opcional)

Para usar MLflow con backend en PostgreSQL y artefactos en MinIO:

```bash
cd infra
cp .env.example .env   # Ajustar variables si es necesario
docker compose up -d
```

Luego apuntar el cliente a `http://localhost:5001` (en `configs/default.yaml`: `mlflow.tracking_uri`).

### Ejemplos de CLI

- Otro archivo de datos: `python -m src.main --data-path ruta/a/datos.csv`
- Más épocas: `python -m src.main --epochs 80`
- Activar Optuna: `python -m src.main --use-optuna`
- SMOTE: `python -m src.main --use-smote --smote-sampling-strategy 0.3`
- Eliminar features muy correlacionadas: `python -m src.main --drop-collinear --corr-threshold 0.95`
- Omitir evaluación (prueba rápida): `python -m src.main --skip-evaluation`

Más ejemplos en `docs/cli_examples.md`.

## Qué incluye el pipeline

- **Datos**: preparación desde CSV, splits train/val/test estratificados, opción de eliminar features por correlación.
- **Desbalance**: SMOTE y/o class weights configurables.
- **Modelo**: red neuronal (capas ocultas, dropout, L2, early stopping, ReduceLROnPlateau).
- **Tuning**: Optuna opcional (coarse + fine, métrica configurable, ej. AUC-PR).
- **Evaluación**: precisión, recall, F1, AUC-ROC, AUC-PR, matriz de confusión, umbral óptimo, curvas y figuras.
- **MLflow**: parámetros, métricas, tags (p. ej. `data_hash` si usas DVC), estadísticas de datos, registro del modelo como PyFunc (modelo + pipeline) con nombre y descripción.
- **Artefactos locales**: modelo Keras, pipeline de preprocesamiento (pickle), nombres de features, config de entrenamiento, reportes y gráficos.

## Inferencia

El módulo `src.predict` permite usar el modelo y el pipeline guardados para predicciones (batch o una transacción), listo para integrar en una API (p. ej. FastAPI) o jobs por lotes:

```python
from src.model import load_artifacts
from src.predict import predict

model, pipeline, feature_names, config = load_artifacts("outputs/models")
results = predict(model, pipeline, X_new, threshold=0.5)
# results["predictions"], results["probabilities"], results["risk_levels"]
```

## Documentación adicional

- `docs/cli_examples.md` — Ejemplos detallados del CLI y configs.
- `docs/refactor_plan.md` — Arquitectura y plan de refactor.
- `configs/default.yaml` — Todas las opciones disponibles.

## Licencia

Ver `LICENSE` en el repositorio.
