Rules frozen before execution; any deviation must be reported as a protocol violation.

=====================================================================
PHASE 0: GATES FOR FINDING F3
=====================================================================
FROZEN GATE 0.1 (Split-Half Check): 
per-dataset Spearman rho(R^2_A, sigma_B) over 16 models; one-sample Wilcoxon on the 10 rhos. PASS = median rho < 0 AND p < 0.05. Report attenuation vs the original -0.646.

FROZEN GATE 0.2 (Within-Pipeline Check): 
PASS = Pipeline-A-only median rho < 0 AND Wilcoxon over the 10 Pipeline-A rhos p < 0.05. Pipeline-B result reported descriptively.

=====================================================================
PHASE 0.5: CAUSAL SIMULATION
=====================================================================
FROZEN PREDICTIONS: 
(P1) within each family, R^2 decreases with the jaggedness parameter: Spearman rho(param_rank, R^2) < -0.7 per family pooled over replicates (for SVC use gamma rank; for tree use depth). 
(P2) within each family, Spearman rho(R^2, sigma_bar) < 0 with p < 0.05 pooled over replicates and configs. Both P1 and P2 met in both families = causal mechanism DEMONSTRATED; partial = report exactly which prediction failed where.

=====================================================================
PHASE 1: THEORY, DECONFOUNDED METRIC, NON-SDP
=====================================================================
FROZEN INTERPRETATION (Derivation Fit): 
confirmed if b in [0.35, 0.65] AND fit R^2 > 0.5; partially supported if b > 0 with CI excluding 0; else not supported.

FROZEN CLAIM RULE (Deconfounded Stability): 
"deconfounding restores validity" claimed only if the correlation improves (abs improvement toward the faithfulness ranking, i.e. rho(sigma_star,faith) > rho(sigma,faith)) in >= 8/10 datasets with Wilcoxon p < 0.05; otherwise report the numbers without that claim.

FROZEN PREDICTION (Non-SDP Replication): 
all 3 rhos < 0; median in [-0.8, -0.4].
FROZEN CLAIM RULE: Paper-A "property of LIME in general" claim only if 3/3 negative; otherwise the finding is scoped as a boundary condition and reported as such.

=====================================================================
PHASE 2: FALSIFICATION CLOSURE
=====================================================================
FROZEN PREDICTION (Dose-Response): 
performance is a smooth function of c; the shap-weighted point lies within the c-curve's local noise band.

FROZEN RULES (mRMR-SHAP): 
(R1) "redundancy mechanism CONFIRMED" iff mrmr_vs_random Holm-significant LOSSES = 0 while plain_vs_random losses >= 1 replicate in this run. 
(R2) "importance carries transferable signal" (stronger) iff additionally >= 1 Holm-significant WIN in mrmr_vs_random AND >= 1 in mrmr_vs_shuffled_mrmr.

=====================================================================
PHASE 3 & 4: GENUINE CPDP HARNESS and SHAP-AS-PRIOR
=====================================================================
FROZEN INTERPRETATION (CPDP Harness): 
cross-source weight Spearman < 0.5 within an ecosystem = "the importance vector is itself source-unstable", an independent failure explanation, reported as such.

FROZEN GATE 4 (SHAP-As-Prior): 
PASS iff in zero-shot primary: shap_prior has >= 2 Holm-significant wins over uniform-prior AND >= 2 over shuffled-prior AND 0 significant losses to either. Anything else = FAIL, reported as the completing negative result. NO retries with different tau/thresholds beyond the pre-declared sensitivity arms.
