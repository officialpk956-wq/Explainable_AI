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

# 📊 Experimental Results

## Phase 1 — In-Project Performance

The source datasets were evaluated using **5-Fold Stratified Cross Validation** after hyperparameter optimization.

| Dataset | Best Model | F1-Macro | ROC-AUC | PR-AUC |
|----------|------------|:--------:|:-------:|:------:|
| Eclipse JDT | Extra Trees | **0.728** | **0.792** | **0.623** |
| Mylyn | Extra Trees | **0.635** | **0.732** | **0.351** |

Both source datasets consistently selected **Extra Trees** as the best-performing classifier.

---

## SHAP Feature Importance

The global SHAP values generated from the source datasets were averaged to obtain a transferable feature-weight vector.

| Feature | Weight |
|---------|:------:|
| NBFU | **0.279** |
| NNTBFU | 0.270 |
| NHPBFU | 0.198 |
| NMBFU | 0.168 |
| NCBFU | 0.085 |

> [!TIP]
> The generated SHAP weights are computed **Out-of-Fold**, ensuring that no information from the test fold leaks into the explanation process.

---

## Phase 2 — Cross-Project Results

Each target dataset was evaluated twice:

- Original Features
- SHAP Weighted Features

| Target Dataset | Best Improvement | Gain |
|---------------|-----------------|------|
| Equinox | Gradient Boosting | **+1.75% F1** |
| Lucene | Support Vector Machine | **+3.48% ROC-AUC** |
| Eclipse PDE | Logistic Regression | **+2.68% F1** |

Although improvements are modest, they are achieved **without using any target labels**, making the approach inexpensive and easily transferable.

---

## Key Observation

> [!IMPORTANT]
> SHAP weighting consistently improves **linear and margin-based classifiers** but has little effect on tree-based ensembles.

Why?

Decision trees depend on the **ordering of feature values**, not their magnitude. Multiplying a feature by a positive constant preserves that ordering, so tree splits remain unchanged.

Linear models and Support Vector Machines, however, depend directly on feature magnitudes, allowing SHAP-based weighting to influence decision boundaries.

---

# 🧪 LIME Stability Analysis

To evaluate explanation consistency, LIME explanations were generated using three different random seeds.

| Dataset | Best Model | Mean σ |
|---------|------------|:------:|
| Eclipse JDT | Extra Trees | **0.00002** |
| Mylyn | Extra Trees | **0.00011** |

Lower variance indicates more stable explanations.

The results show that tree ensembles produce highly consistent local explanations compared with distance-based models.

---

# 📁 Repository Structure

```text
Explainable_AI
│
├── data/                  # Eclipse benchmark datasets
├── outputs/               # Results, models and explanations
│   ├── models/
│   ├── shap/
│   ├── lime/
│   ├── plots/
│   └── tables/
│
├── src/                   # Source code
├── notebooks/             # Exploratory notebooks
│
├── pipeline.py            # Main pipeline
├── requirements.txt
├── README.md
│
├── main.tex
├── references.bib
└── xai.pdf
```

---

# 🚀 Getting Started

## Installation

```bash
git clone https://github.com/officialpk956-wq/Explainable_AI.git

cd Explainable_AI

pip install -r requirements.txt
```

---

## Run the Pipeline

```bash
python pipeline.py
```

The pipeline automatically performs

- Dataset preprocessing
- Hyperparameter optimization
- Cross validation
- SHAP computation
- LIME explanation
- Cross-project evaluation
- Result generation

All outputs are automatically saved inside the **outputs/** directory.

---

# 📈 Reproducibility

Every experiment is fully reproducible.

| Setting | Value |
|---------|-------|
| Random Seed | 42 |
| Cross Validation | 5-Fold Stratified |
| Hyperparameter Search | RandomizedSearchCV |
| Search CV | 3-Fold |
| Evaluation Metrics | F1, ROC-AUC, PR-AUC |

---

# ⚠️ Limitations

- Experiments are limited to Eclipse ecosystem datasets.
- Only five shared bug-history metrics are considered.
- SHAP weights are transferred from two source projects only.
- Improvements are relatively small but computationally inexpensive.
- Dynamic feature weighting remains future work.

---

# 🛣️ Future Work

- [x] Leakage-free SHAP computation
- [x] Hyperparameter optimization
- [x] LIME stability evaluation
- [x] Cross-project feature weighting
- [ ] Evaluation on Apache projects
- [ ] Deep learning based feature weighting
- [ ] Active learning for dynamic weight updates
- [ ] Transformer-based defect prediction

---

# 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@article{kumar2026xai,
  title={Enhancing Software Defect Prediction Using Explainable AI and Cross-Project Feature Weighting},
  author={Priyanshu Kumar and Kumar Rajnish and Shubham Kumar},
  year={2026},
  institution={Birla Institute of Technology Mesra}
}
```

---

# 🙏 Acknowledgements

This work is based on the Eclipse Software Defect Prediction Benchmark introduced by:

> D'Ambros, M., Lanza, M., & Robbes, R. (2012). *Evaluating Defect Prediction Approaches: A Benchmark and an Extensive Comparison.* Empirical Software Engineering.

Special thanks to the developers of:

- SHAP
- LIME
- Scikit-learn
- XGBoost
- LightGBM

---

# 👥 Authors

| Name | Affiliation |
|------|-------------|
| **Priyanshu Kumar** | Birla Institute of Technology, Mesra |
| **Kumar Rajnish** | Birla Institute of Technology, Mesra |
| **Shubham Kumar** | Birla Institute of Technology, Mesra |

---

<div align="center">

### ⭐ If you found this repository useful, please consider starring it.

**Explainability should not only interpret models—it should help build better ones.**

</div>
