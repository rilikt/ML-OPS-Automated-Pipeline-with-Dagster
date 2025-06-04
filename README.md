# 🔁 ML-Ops | Part 3: From Notebook to Production Pipeline

This is the final part of the ML-Ops trilogy. In this stage, we moved beyond notebooks and into **production-grade MLOps workflows**, focusing on automation, reproducibility, and modularity.

We used **Dagster** to orchestrate our ML pipeline and **LakeFS** to version the data. This allowed us to build a robust pipeline that detects data changes, retrains models automatically, and follows best practices used in modern machine learning systems.

---

## 🧠 Overview

### ✅ What We Did

- Rebuilt a full ML pipeline from scratch using **Dagster**
- Automated retraining of models when new data appears via **Dagster Sensors**
- Versioned datasets using **LakeFS**
- Wrote modular, type-annotated, and testable code
- Enforced code quality with **Ruff** and **Pyright**
- Logged and registered models with **MLflow**
- Committed everything in a clear GitHub repository

---

## 🔧 Tools & Technologies

- [Dagster](https://dagster.io/) – Orchestration and asset management
- [LakeFS](https://lakefs.io/) – Data version control
- [mlflow](https://mlflow.org/) – Model tracking and registry
- [lakefs-spec](https://github.com/treeverse/lakefs-spec) – LakeFS Python SDK

---

## 🔄 Automated Pipeline

We defined each ML step as a **Dagster asset**:

- `load_data()` – Load versioned data from LakeFS
- `preprocess()` – Clean and transform input features
- `train_model()` – Train model and log to MLflow
- `evaluate_model()` – Assess performance with custom metrics

Assets can be independently executed or automatically triggered by changes.

---

## 📁 Data Versioning with LakeFS

LakeFS acts like Git for data. We:

- Split the dataset into chunks (e.g., by year or source)
- Simulated new data arrivals via separate commits
- Used a **Dagster Sensor** to monitor LakeFS branches
- Triggered the full ML pipeline on new commits

---

## 🚨 Sensors & Automation

Our **Dagster Sensor** watches a LakeFS repository. When new data is detected:

- It triggers the corresponding asset materialization
- The pipeline automatically reloads, preprocesses, retrains, and logs a new model
- Model versions are stored and tracked via MLflow

---

## 🧪 Testing & Quality

We followed professional software practices:

- ✅ Type hints across all code
- ✅ Modular functions with docstrings
- ✅ Unit tests with `pytest`
- ✅ Linting and formatting with `ruff`
- ✅ Static checks with `pyright`

---

## 🚀 Key Takeaways

- Transitioning from notebook experimentation to production pipelines
- Decoupling ML workflow steps using asset-based orchestration
- Monitoring and reacting to data changes
- Maintaining reproducibility and traceability across runs

This part wrapped together everything we learned in Parts 1 and 2 and gave a real-world view of how MLOps is done in modern AI workflows.

---
