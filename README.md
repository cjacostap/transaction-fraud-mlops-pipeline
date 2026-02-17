# Mentorship Program: MLOps Transition
*Transitioning from Economics/Quant to Production Machine Learning Engineering*

## 🎯 Objective
This repository tracks the mentorship journey to transition an **Economist/Quantitative Analyst** into an **MLOps Engineer**. The goal is to move from "Notebook-based" development to "Production-grade" pipelines.

We use a practical **Deep Learning for Fraud Detection** use case to master these skills.

## 📚 Repository Structure
```text
.
├── fraud-detection-mlops/      # 🛡️ MAIN PROJECT WORKSPACE (Target Architecture)
├── sessions/                   # Practical guides for each mentorship session
├── proposal/                   # Mentorship curriculum and PDF docs
├── environment.yml             # Conda environment definition
└── README.md                   # This roadmap
```

---

## 🚀 Mentorship Roadmap

### ✅ Phase 1: MLOps Fundamentals (Sessions 1-3 Completed)
In the first three sessions, we covered the conceptual backbone of Modern MLOps.

*   **Session 1: The MLOps Landscape**
    *   **What is MLOps?**: Bringing DevOps discipline to Machine Learning.
    *   **Roles Defined**: Differences between *ML Engineer* (Deployment/Scale), *Data Engineer* (Pipelines), *Data Scientist* (Modeling), and *DevOps* (Infra).
    *   **The Lifecycle**: Design -> Data -> Modeling -> Deployment -> Monitoring.
*   **Session 2: Infrastructure as Code & Data as Code**
    *   **IaC**: Treating server configuration as Git-versioned code (Terraform/Ansible logic).
    *   **Data as Code**: Why storing CSVs in Git is bad, and the need for DVC.
*   **Session 3: Containers & Orchestration**
    *   **Docker**: The unit of deployment. "It runs on my machine" solver.
    *   **Kubernetes (K8s)**: How to manage valid containers at scale.

### 🚧 Phase 2: Building the Production Pipeline (Sessions 4-7)
We are now moving to **Hands-on Implementation**. We will build the `fraud-detection-mlops` project from scratch.

#### Session 4: The Foundation - Reproducibility & Tracking
*   **Goal**: Tracking Data (DVC) and Experiments (MLflow).
*   **Activities**:
    *   Initialize `fraud-detection-mlops` structure.
    *   Track `raw/onlinefraud.csv` with DVC.
    *   Add `mlflow.autolog()` to the experiments.
*   **➡️ Homework**: Setup local stack (MLflow+MinIO) and run fully tracked experiments.

#### Session 5: Modularization & Containerization
*   **Goal**: Refactoring Notebooks into Production Scripts.
*   **Activities**:
    *   Split notebook into `src/preprocess.py` and `src/train.py`.
    *   Create `Dockerfile` for training jobs.
*   **➡️ Homework**: Run training *inside* a container and confirm it logs to MLflow.

#### Session 6: Pipeline Orchestration & Registry
*   **Goal**: Automating the workflow with Kubeflow Pipelines (KFP).
*   **Activities**:
    *   Define a DAG connecting Preprocessing -> Training.
    *   Register the best model to MLflow Model Registry.
*   **➡️ Homework**: Compile the pipeline and implement conditional registration.

#### Session 7: Serving & Feedback Loop
*   **Goal**: Exposing the model via API.
*   **Activities**:
    *   Build a `FastAPI` serving app.
    *   Load models from MLflow Registry.
*   **➡️ Homework**: Deploy API + DB logging for predictions.

---

## 📝 Getting Started

1.  **Clone & Checkout**:
    ```bash
    git clone <repo-url>
    cd cristhian-mlops-transition
    git checkout feature/mentorship-guide
    ```

2.  **Environment Setup**:
    ```bash
    conda env create -f environment.yml
    conda activate mlops_transition
    ```

3.  **Start Learning**:
    Go to `sessions/session_04_tracking.md` to begin the next session.
