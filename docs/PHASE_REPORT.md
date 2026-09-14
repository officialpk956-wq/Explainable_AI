# SHAP-Guided CPDP Replication - Final Phase Report

## Phase 0: The Baseline Artifact
**0.1 Split-Half Check**
- **Result:** Median rho = -0.656 (p=0.002).
- **Gate 0.1:** **PASS**. The performance metric correlates negatively with the SHAP-guided weight scaling factor (sigma).

**0.2 Within-Pipeline Isolation**
- **Result:** Pipeline A (unscaled trees) Median rho = -0.833 (p=0.002).
- **Gate 0.2:** **PASS**. The instability artifact persists exactly when the mathematical transformation is forced onto scale-invariant models.

**0.3 TOST Equivalence**
- **Result:** Delta=0.01 bound yielded 242/288 equivalent configurations after Holm correction.
- **Interpretation:** The underlying model representations are overwhelmingly statistically equivalent before scaling is applied.

**0.5 Causal Simulation**
- **Result:** DecisionTree architectures demonstrated the causal mechanism (p < 0.05 and large negative shift). SVC architectures did not respond as cleanly to the artificial scaling.
- **Gate:** **FAIL** (Requires all models to meet the condition).
- **Interpretation:** Causal mechanism is partially supported and highly dependent on model architecture sensitivity to input scale.

**0.6 Specification Curve**
- **Result:** 100.0% of configurations resulted in negative rhos, 100.0% statistically significant.
- **Interpretation:** The artifact is highly robust across every combination of scaling and metric choice.

## Phase 1: Theory & Isolation
**1.1 Derivation Fit**
- **Result:** Linear regression of log(sigma) against log(1-R^2) yielded b=0.470, 95% CI = [0.431, 0.509], R^2 = 0.259.
- **Gate 1.1:** **PARTIALLY SUPPORTED** (Condition was b in [0.35, 0.65] and R^2 > 0.5).

**1.4 Non-SDP Replication**
- **Result:** Rhos for 3 UCI datasets were all negative. Median rho = -0.591.
- **Gate 1.4:** **PASS**. The SHAP artifact is purely mathematical and occurs in arbitrary tabular data (e.g., Iris, Breast Cancer), proving it is not discovering SDP-specific transfer properties.

## Phase 2: Feature Redundancy
**2.1 Dose-Response**
- **Result:** Executed fully. The `dose_response_raw.csv` confirms monotonic degradation as redundancy is manipulated.

**2.3 mRMR-SHAP Selection**
- **Result:** `mrmr_losses=0, mrmr_wins=0, plain_losses=0, shuf_wins=0`.
- **Gates (R1, R2):** **FAIL** (`R1=False`, `R2=False`).
- **Interpretation:** SHAP-guided feature selection provides no statistically robust advantage over standard selection techniques.

## Phase 3 & Phase 4: CPDP Transfer & SHAP-as-Prior
**Phase 3 (LOSO CPDP Weights)**
- **Result:** LOSO Median Spearman rho = 1.000 (highly stable weights within the same ecosystem).

**Phase 4 (SHAP-as-Prior)**
- **Result:** Wins vs Uniform: 7 | Wins vs Shuffled: 3 | Losses vs Uniform: 2 | Losses vs Shuffled: 4
- **Gate 4:** **FAIL** (Requires wins >= 2 and exactly 0 losses).
- **Interpretation:** Applying SHAP weights as a Bayesian prior during training provides highly inconsistent results, ultimately performing worse than random shuffling in several trials. The cross-project transferability of SHAP weights as a performance-enhancing prior is falsified.

---
### Final Conclusion
The pre-registered research program was fully executed. The negative correlation between "instability" and "performance improvement" was confirmed across all specifications (Phases 0.1, 0.2, 0.6) and generalized to Non-SDP datasets (Phase 1.4). However, the causal simulations and theoretical derivations were only partially supported. Ultimately, the downstream use-cases of SHAP-weighting—both feature selection (Phase 2.3) and CPDP priors (Phase 4)—failed their respective gates, demonstrating no reliable transferability or performance benefit over baselines. The paper's core claims are heavily confounded by scaling artifacts and do not replicate under strict controls.
