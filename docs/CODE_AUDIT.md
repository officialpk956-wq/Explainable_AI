# Research code audit — 9 September 2026

The repository already contains implementations of all five principal explanation-transfer approaches and the main stability experiments. It did not have a complete reproduction runner or standalone generators for several descriptive tables. Those gaps have been addressed below. **Code availability is not the same as independent numerical reproduction.** The saved experiment results were audited; the complete model-training campaign was not rerun.

## Scope and evidence

Read the supplied `paper/Do_Explanations_Transfer.docx`, compared methodology with project scripts and inspected saved result schemas, counts and main claims. `outputs/audit/file_inventory.csv` inventories and hashes project files recursively, parses all project Python files, inspects notebook structure and counts CSV records. Environment packages, Git internals and caches are excluded. Notebooks and historical document versions were inventoried, not independently executed or scientifically validated. The extracted DOCX text omits Word equation objects; equations were cross-checked against `paper/main.tex` where needed.

Existing uncommitted work was present before this audit. No historical experiment CSV was replaced, no training result was fabricated, and no manuscript claim was changed to make a test pass.

## Code coverage

| Paper component | Implementation | Evidence / qualification |
|---|---|---|
| Table 1: IQR row removal | **New:** `manuscript_diagnostics.py` | `outputs/audit/iqr_removal.csv`; operates on predictors, excludes identifiers and label |
| Table 2: dataset counts, imbalance | **New:** `manuscript_diagnostics.py` | `outputs/audit/dataset_summary.csv`, includes excluded projects and active feature counts |
| Source training, within-target weighting | `pipeline.py`, `cross_ecosystem.py` | Existing weights, CV, repeated scores, significance files |
| Tables 4–5: matched ablation / equivalence | `ablation_control.py`, `equivalence_tests.py` | Existing scores and paired tests; raw and adjusted equivalence must be distinguished |
| Table 6: zero-IQR scaling mechanism | **New:** `manuscript_diagnostics.py` | `outputs/audit/scaler_diagnostics.csv`, confirms Equinox differences 8.8 and 3.2 |
| Scaling / dose / mechanism | `scale_invariant_fix.py`, `dose_response.py`, `mechanism_ablation.py` | Existing experiments; mechanism gate fails, see below. Table 7's “spread retained” percentage lacks a defined reproducible estimator; no number was invented for it |
| Feature selection and mRMR | `shap_feature_selection.py`, `mrmr_shap.py`, `reconcile_selection.py` | Corrected selection comparison is in the reconciliation script |
| Tables 8–9: redundancy | `redundancy_all_targets.py`, `synthetic_redundancy.py`, `synthetic_redundancy_v2.py` | Original and corrected synthetic designs retained |
| Table 10: SFA | `sfa_cross_project.py` | Existing scores, tests and verdict: 9 wins / 30 losses vs prediction augmentation |
| Tables 11,16: native priors | `shap_prior_zeroshot.py`, `shap_as_prior_fixed.py` | Corrected scripts; legacy `shap_prior.py` writes colliding filenames and is excluded from runner |
| Tables 12–15: ecosystems, power, CPDP, LOSO | `cross_ecosystem.py`, `power_analysis.py`, `genuine_cpdp.py` | Zero-shot CORAL 51 wins / 21 losses; SHAP 8 / 7 |
| Table 17: stability, causality, non-SDP | `powered_phase_h.py`, `h_robustness.py`, `causal_simulation.py`, `h_nonsdp.py` | Existing results; 748 specifications are stability specifications |
| Theory and residualized stability | `stability_theory.py`, `deconfounded_stability_tasks.py` | **Fixed:** theory now writes `derivation_fit.csv`, no longer overwrites validation results |
| Convergence | `stability_convergence.py` | Implemented separately; misleading theory stub message removed |
| Result figures | `make_figures.py`, `make_figures2.py`, `dose_response.py` | Main result figures wired into runner; document layout/workflow diagrams are manuscript assets |
| Historical analysis | `src/`, seven notebooks, notebook-generation scripts | Separate earlier analysis, not authoritative reproduction entrypoints for this paper |

## Changes made

- Replaced nine-stage `run_all.py` with 29 dependency-ordered stages, selectable stages, listing mode, logs, output-presence checks and run manifests. Dependencies are included automatically. Cached outputs are explicitly labelled unverified.
- Added `manuscript_diagnostics.py`, `audit_research.py` and focused regression/smoke tests in `test_research_updates.py`.
- Fixed `stability_theory.py` output collision and missing-input handling.
- Fixed verifier exit status and removed the invalid restriction that AOPC must be nonnegative: a deletion can increase predicted confidence, producing a negative confidence drop.
- Made central data/output paths repository-relative in `pipeline.py`, `cross_ecosystem.py`, and the verifier.
- Created an isolated `.venv-audit` and recorded its installed versions in `requirements-audit.lock.txt`. The old `sdp_env` points to a missing Python 3.11 installation. The new lock is a validation environment, not a claim about historical training versions.

## Findings requiring manuscript correction or further experiments

1. **SMOTE protocol mismatch.** Section 4.2 says fewer than 100 minority examples and oversampling to 1:2. `ConditionalSMOTE` actually activates when majority/minority exceeds 2, and default SMOTE balances to 1:1. The historical CSVs cannot be claimed to follow the stated rule. Choose between reporting the actual implementation or rerunning under a revised protocol.
2. **Tuning is not nested for every repeated split.** In `ablation_control.py`, tuning uses the seed-42 training portion once, then reuses parameters for seeds 42–61. Examples used in tuning can later occur in another split's test set. The claim that each split has independently training-only tuning is inaccurate. A full correction would require new training and updated results, not just a wording change.
3. **OOF protocol differs.** `shap_weights_out_of_fold` uses five folds on source training data, not twenty repeated folds as Section 4.2 states. Validation covariates are explained OOF; distinguish that from deriving weights from target test data. Hyperparameters selected before OOF are not fully nested.
4. **Main classifier list differs.** The eight main models contain ExtraTrees, not DecisionTree. DecisionTree appears in the broader stability suite.
5. **TOST raw vs adjusted.** At margin 0.01 the saved file has 268/288 raw equivalences but 242/288 Holm-adjusted equivalences. At 0.005 it has 236 raw and 202 adjusted. The manuscript's 268 is supportable only when explicitly labelled unadjusted. At 0.005, raw counts per control are 117 uniform and 119 shuffled, not the manuscript's 121 and 115.
6. **The 748 specification paragraph is assigned to the wrong experiment.** Axes are Spearman/Kendall, 5/10 instances, leave-one-dataset-out and leave-one-model-out (2 × 2 × 11 × 17). It is about surrogate-fit/stability correlation, not metric/ecosystem/TOST robustness of the transfer null.
7. **Mechanism claims overstate the saved test.** `mechanism_ablation.py` tests SHAP vs original in two pipelines across six targets, three models and three metrics. Nine significant cells occur with log1p; zero without. These are not 9/10 datasets and there is no remove-RobustScaler arm. The no-log median absolute effect remains 0.00285 and the recorded `mechanism_demonstrated` is False. Zero-IQR columns are a supported partial explanation, not proof that all mechanisms are resolved.
8. **Dataset definitions need precision.** POI's majority/minority ratio is 1.745, not 0.57 (the latter is clean/buggy). Lucene has five raw predictors but only two after the implemented zero-variance filter. Several captions imply all five are used. Table 1 also rounds ant defective deletion 70.48% to 71%, and xalan retention 49.49% to 50%; regenerate consistently.
9. **Prior control caveat.** `shap_prior_zeroshot_verdict.csv` records `all_priors_nonconstant=False`. Inspect `shap_prior_group_audit.csv` before claiming all prior arms are informative or nondegenerate.
10. **Power scope.** `power_analysis.py` estimates per-cell unadjusted Wilcoxon power, whereas primary claims use Holm families. Repeated holdout splits overlap. Its MDE cannot alone establish family-wise power or justify the unrestricted statement that any practically meaningful effect would have been found.
11. **Legacy output collisions.** `cpdp_harness.py`, `task6_loso.py`, `shap_prior.py`, and `h_robustness.py` can write schemas also produced by later scripts. Runner uses corrected CPDP/prior producers and runs the dedicated redundancy generator after robustness. Avoid mixing legacy stages into a publication run.
12. **Few-shot label budget.** The manuscript says 10 labelled target instances; `shap_as_prior_fixed.py` uses `k = 25`. Report the actual budget or rerun the intended design.
13. **Descriptive table limits.** The exact estimator behind Table 7's spread-retention percentages is not specified. An undefined measure cannot be reconstructed honestly; define it before adding a generator or revising those percentages.

## Validation and use

```powershell
.\.venv-audit\Scripts\python.exe manuscript_diagnostics.py
.\.venv-audit\Scripts\python.exe audit_research.py
.\.venv-audit\Scripts\python.exe test_research_updates.py
.\.venv-audit\Scripts\python.exe verify_outputs.py
.\.venv-audit\Scripts\python.exe run_all.py --list
# Expensive training only if outputs are absent:
.\.venv-audit\Scripts\python.exe run_all.py
# Select a stage and its prerequisites:
.\.venv-audit\Scripts\python.exe run_all.py --stage equivalence
```

`--force` reruns stage entrypoints but retains internal tuning/robustness caches. It is not a clean-room reproduction switch. For independent reproduction, use a separate copy containing code and raw data with an empty outputs directory. Preserve this repository's historical outputs for comparison. Full runs may be very expensive; source attribution via KernelSHAP is particularly costly.

The existing output contracts pass and the six audited score grids have finite in-range metrics, no duplicate experimental keys, and 20 seeds per group. See `outputs/audit/contract_checks.txt`, `outputs/audit/evidence_summary.json` and `outputs/audit/smoke_tests.txt` for measured evidence. Full retraining, scientific revalidation of historical notebooks, external citation verification and manuscript rendering were not performed in this audit. The original DOCX is preserved; this report records corrections explicitly so unsupported numerical changes are not silently published.

Validation completed: all six focused tests passed, including 16 real-data model fits (eight classifiers on each ecosystem), five-fold RandomForest SHAP extraction, scaler equivalence against scikit-learn, IQR counts, dependency ordering, and the theory output-collision regression. A small native XGBoost prior fit also passed. The new runner includes the previously omitted surrogate-fit analysis and bias-free ranking stage.
