<div align="center">

# 🔍 Explainable AI for Cross-Project Software Defect Prediction

### Leveraging SHAP-Based Feature Weighting for Transfer Learning Across Software Projects

<p align="center">
A research project demonstrating how <b>explainability can become an active learning mechanism</b> rather than just a post-hoc interpretation tool.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=for-the-badge)
![LIME](https://img.shields.io/badge/XAI-LIME-success?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-Cross--Project_SDP-red?style=for-the-badge)

</p>

<p align="center">

**Birla Institute of Technology, Mesra**

**Priyanshu Kumar • Kumar Rajnish • Shubham Kumar**

</p>

---

*"Can explanations learned from one software project improve predictions on another?"*

</div>

---

# 📖 Overview

Software Defect Prediction (SDP) models often struggle when applied to new software projects because feature distributions vary significantly across repositories. Traditional explainable AI techniques such as **SHAP** and **LIME** help explain model predictions, but their outputs are rarely reused for improving future models.

This project investigates a different idea:

> **Can explanations themselves become transferable knowledge?**

Instead of using SHAP solely for interpretation, we transform **global SHAP feature importance** into a reusable feature-weighting vector that is transferred to completely unseen software projects. The weighted features are then evaluated against the original features using multiple machine learning models to determine whether explanation-guided learning improves cross-project defect prediction.

---

# ✨ Highlights

- 🚀 Novel SHAP-based feature weighting strategy for Cross-Project Software Defect Prediction
- 🔒 Leakage-free Out-of-Fold SHAP computation
- ⚙️ Automatic hyperparameter optimization using RandomizedSearchCV
- 📊 VIF analysis for multicollinearity
- 🌳 Evaluation across **8 machine learning models**
- 🔍 LIME stability analysis across multiple random seeds
- 📈 Performance comparison on **5 Eclipse ecosystem datasets**
- 📑 Comparison against published research

---

# 🏗 Project Workflow

```text
                  Phase 1 (Source Projects)
        Eclipse JDT + Mylyn Datasets
                    │
                    ▼
         Hyperparameter Optimization
                    │
                    ▼
          5-Fold Cross Validation
                    │
                    ▼
          Best Performing Models
                    │
                    ▼
       Out-of-Fold SHAP Computation
                    │
                    ▼
      Global SHAP Feature Weight Vector
                    │
────────────────────────────────────────────────
                    │
                    ▼
            Phase 2 (Target Projects)

       Equinox • Lucene • Eclipse PDE

                    │
        ┌───────────┴───────────┐
        ▼                       ▼

 Original Features      SHAP Weighted Features

        │                       │
        └───────────┬───────────┘
                    ▼

        Performance Comparison
      (F1 • ROC-AUC • PR-AUC)
```

---

# 💡 Key Idea

Unlike conventional Explainable AI pipelines, this work treats explanations as **knowledge that can be transferred**.

The transformation is intentionally simple.

Instead of learning another deep neural network or feature extractor, each feature column is multiplied by its normalized global SHAP importance.

```text
Weighted Feature = Original Feature × SHAP Weight
```

This allows knowledge learned from one software project to influence another project **without requiring target labels**.

> **The contribution of this work is not a new classifier—it is a new way of using explanations.**

---

# 📂 Datasets

The experiments use the **Eclipse Software Defect Prediction Benchmark** proposed by D'Ambros et al.

| Dataset | Role | Samples | Buggy | Clean |
|---------|------|---------:|-------:|------:|
| Eclipse JDT | Source | 997 | 206 | 791 |
| Mylyn | Source | 1862 | 620 | 1242 |
| Equinox | Target | 324 | 129 | 195 |
| Lucene | Target | 691 | 64 | 627 |
| Eclipse PDE | Target | 1497 | 209 | 1288 |

Only five common defect-history metrics are retained across every dataset.

| Feature |
|---------|
| NBFU |
| NNTBFU |
| NMBFU |
| NCBFU |
| NHPBFU |

Columns directly derived from defect labels are removed before training to eliminate information leakage.

---

# 🔬 Methodology

The complete experimental pipeline consists of six stages.

### 1️⃣ Data Preparation

- Feature selection
- Leakage removal
- Log transformation (linear models)
- Robust Scaling
- Conditional SMOTE

### 2️⃣ Hyperparameter Optimization

Each classifier is independently optimized using:

- RandomizedSearchCV
- 3-Fold Cross Validation
- F1-Macro optimization

### 3️⃣ Model Training

Eight machine learning models are evaluated.

- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors

### 4️⃣ Explainability

The best model from each source dataset generates Out-of-Fold SHAP explanations.

Global SHAP importance is averaged and normalized to produce a transferable feature-weight vector.

### 5️⃣ Cross-Project Transfer

Each target dataset is evaluated twice.

- Original Features
- SHAP Weighted Features

Both experiments use identical train-test splits.

### 6️⃣ Performance Evaluation

Performance is measured using

- F1 Macro
- ROC-AUC
- PR-AUC

allowing direct comparison between weighted and non-weighted features.

---
