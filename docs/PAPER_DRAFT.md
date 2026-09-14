# Do Explanations Transfer? A Controlled Falsification of SHAP-Guided Cross-Project Defect Prediction

**Priyanshu Kumar, Kumar Rajnish, Shubham Kumar**
Birla Institute of Technology, Mesra, Ranchi 835215, India

> **Status:** rewrite of the original manuscript. Every number below is reproduced from
> the artefacts in `outputs/` and was independently re-verified from the raw score files.
> Provenance for each claim is given in square brackets.

---

## Abstract

Explainable AI methods are increasingly proposed not only to interpret software defect
prediction models but to improve them: SHAP-derived feature importance is used to weight,
select, or prioritise features, on the premise that an explanation learned on one project
carries transferable information to another. We set out to reproduce and strengthen one
such method and instead falsify its central premise. Across two software ecosystems
(five Eclipse-family projects with bug-history metrics; five Apache projects with CK/OO
metrics), eight to sixteen classifiers, and twenty repeated stratified splits with
Holm-corrected paired testing, we evaluate five distinct operationalisations of
explanation-guided transfer: multiplicative feature weighting, feature selection,
redundancy-corrected (mRMR) selection, native model priors, and Shapley feature
augmentation. **None survives a magnitude-matched control.** The reported gains of
multiplicative weighting are statistically indistinguishable from multiplying every
feature by a constant containing no explanation information (0 of 288 comparisons
significant; 268 of 288 formally equivalent under TOST at δ = 0.01), and are reproduced
by SHAP weights attached to the wrong features. We identify the mechanism — normalising
weights to unit sum makes the multiplier a function of feature count, which interacts
with the pipeline's logarithmic transform — and confirm it with a dose–response sweep.
The effect reverses on the second ecosystem. Critically, the same evaluation harness
detects a large, significant transfer effect from a standard domain-adaptation baseline
(CORAL: 51 significant improvements), establishing that the null result is not an
artefact of an insensitive design. We further show that the field's usual instrument for
judging explanation quality is itself confounded: LIME-measured stability is determined
largely by how well a linear surrogate fits the model, not by explanation quality
(ρ = −0.646, negative in 10/10 datasets, p = 0.002; replicated on three non-software
datasets; demonstrated causally by intervention; and predicted analytically). We report
what does and does not follow for practice.

**Keywords:** software defect prediction; explainable AI; SHAP; LIME; cross-project
prediction; negative results; controlled ablation.

---

## 1. Introduction

A small fraction of software components accounts for most defects, so identifying them
before testing concentrates effort where it pays. Two obstacles have proved persistent:
practitioners distrust predictions they cannot interpret, and a model trained on one
project degrades on another because feature distributions shift.

An attractive idea connects the two. If an explanation method such as SHAP tells us which
features a model genuinely relies on, perhaps that knowledge is *transferable* — perhaps
it identifies structure that holds beyond the project it was derived from. Several
recent works pursue this, using SHAP importance to weight, select or engineer features
rather than merely to describe a fitted model.

This paper began as an attempt to strengthen one such method. It became a falsification.

**Our contributions.**

1. A faithful, leakage-free reimplementation of SHAP-guided cross-project feature
   weighting, with hyperparameter tuning and out-of-fold SHAP, improving on the
   originally reported in-project performance (Eclipse JDT AUC 0.720 → 0.792).
2. **A controlled falsification.** Against magnitude-matched controls that carry no
   explanation information, SHAP-derived importance contributes nothing measurable —
   across five operationalisations, two ecosystems and 20 repeated splits.
3. **A mechanism**, confirmed by a dose–response sweep: the transform is an undeclared
   preprocessing hyperparameter whose strength depends on feature count.
4. **A positive control** demonstrating the evaluation is capable of detecting real
   transfer, which is what licenses the null result.
5. **A methodological finding of independent interest:** LIME-based explanation-stability
   metrics measure surrogate fit rather than explanation quality, so model comparisons
   built on them are confounded.
6. **An answer to an open question** posed by Antwarg et al. [SFA] on whether
   Shapley-based methods attend to redundant features: they concentrate redundancy
   rather than avoiding it.

We report failures at the same volume as successes, including three pre-registered
predictions of our own that did not hold.

---

## 2. Related Work and Positioning

**Explanation-guided model improvement.** Antwarg et al. [SFA] propose SFA, a two-stage
learner that appends out-of-fold predictions and their Shapley values as augmented
features. Their evaluation is careful in the way that matters most here: it includes
`P augmented`, an ablation supplying the predictions *without* the Shapley values, which
isolates the Shapley contribution. SFA beats it significantly for XGBoost and LightGBM.
**We do not dispute this result.** Their design is within-distribution — train and test
originate from the same dataset, and out-of-fold explanations of the same
data-generating process are available at inference. Whether the benefit survives
distribution shift is left open, and Section 6.5 answers it.

**SHAP-based feature engineering in applied work.** Kim [APPX] applies SHAP-guided
selection, construction and *weighting* to cancer prediction with LightGBM, reporting
cumulative accuracy gains of +1.74, +0.12 and +0.06 percentage points respectively. We
note that the weighting step — the component our study targets — contributes 0.0006
accuracy on a tree ensemble, which is what our scale-invariance analysis predicts
(Section 6.1). The paper reports a single split without significance testing, so the
component's contribution has not been tested. We treat this as evidence that the
practice is actively recommended and under-examined, not as a criticism of the authors.

**Metric choice under class imbalance.** Richardson et al. [ROC] show by simulation and
case study that ROC-AUC is invariant to class imbalance while PR-AUC changes drastically
with it and cannot be corrected for it. Our targets span imbalance ratios 0.57 to 9.80,
so we adopt ROC-AUC as the primary cross-project metric and report PR-AUC as a secondary
within-project metric. An earlier version of this work made the opposite choice; we
record the correction explicitly.

**Explanation stability.** Prior work reports that models can be accurate while producing
unstable explanations, and ranks models by explanation stability. Section 7 shows that
this ranking is confounded.

---

## 3. Research Questions

- **RQ1** Does SHAP-derived feature importance, transferred from source projects, improve
  prediction on unseen target projects?
- **RQ2** If an improvement is observed, is it attributable to the *information* in the
  attributions, or to the *magnitude* of the transformation? *(This is the question the
  prior literature does not ask.)*
- **RQ3** Does any answer generalise across software ecosystems and feature schemas?
- **RQ4** Is the evaluation capable of detecting a real transfer effect at all?
- **RQ5** Are the instruments used to judge explanation quality themselves valid?

---

## 4. Experimental Design

**Datasets.** Eclipse family (JDT, Mylyn, Equinox, Lucene, PDE), five cumulative
bug-history predictors, from the D'Ambros benchmark. Apache family (ant-1.7, camel-1.6,
xalan-2.6, poi-3.0, velocity-1.6), twenty CK/OO static metrics, from the PROMISE/Jureczko
corpus. Projects enter only if the minority class has ≥ 30 instances — a rule fixed
before inspection, which excludes jedit-4.3 (11 buggy) and log4j-1.2 (16 clean).

**Protocol.** Twenty repeated stratified 80/20 splits (seeds 42–61). Eight base
classifiers spanning tree, boosting, linear, kernel and instance-based families; sixteen
for the stability analysis. Tree models receive an identity preprocessing pipeline
(splits are scale-invariant); linear and distance-based models receive
log1p → RobustScaler → conditional SMOTE, all fitted on training folds only. SHAP is
computed out-of-fold, never on a held-out test split. Hyperparameters are tuned by
randomised search on training data only. All families of tests are Holm–Bonferroni
corrected. Primary metric ROC-AUC [ROC]; F1-macro and PR-AUC also reported throughout.

**Controls.** Every variant is evaluated alongside magnitude-matched controls carrying no
explanation information:

| Control | Construction | Information retained |
|---|---|---|
| `uniform` | every feature × 1/F | none |
| `shuffled` | real SHAP weights permuted across features | magnitudes only; mapping destroyed |

**Pre-registration.** Decision rules, effect thresholds and predictions were fixed in
each script's header before execution and are reproduced in `preregistration.md`.
Deviations are reported as protocol violations; there were none.

---

## 5. RQ1 — The apparent effect, reproduced

Applying the weight vector to the three Eclipse targets reproduces the qualitative result
reported originally: linear and margin-based classifiers gain, tree ensembles do not.
Under 20 repeated splits and Holm correction, 3 of 432 tested combinations are
significant; the largest is +1.26% F1 for LogisticRegression on PDE.
`[outputs/significance_tests.csv]`

Tree ensembles are unaffected to within floating-point boundary effects. This is
mechanically expected: split selection compares raw feature values, and multiplying a
column by a positive constant is a monotone rescaling that cannot change which threshold
wins.

The effect is therefore real, small, and confined to the model families whose geometry
depends on feature magnitude. **The question the literature stops at is whether that
effect comes from SHAP. It does not.**

---

## 6. RQ2 — What the effect is actually caused by

### 6.1 Magnitude-matched controls

On the paper's own best case — Eclipse PDE, LogisticRegression, F1-macro, 20 splits:

| Variant | Mean F1 | Gain vs original | p |
|---|---:|---:|---:|
| SHAP weights | 0.6133 | +1.26% | 0.0001 |
| **Uniform (1/F), no SHAP** | 0.6139 | **+1.31%** | 0.0009 |
| **Shuffled SHAP, wrong features** | 0.6158 | **+1.50%** | 0.0003 |

Across 288 SHAP-vs-control comparisons spanning both ecosystems, three metrics, eight
classifiers and twenty splits: **0 of 144 significant against uniform, 0 of 144 against
shuffled** (Holm-corrected). Equivalence testing is positive rather than merely
non-significant: **268 of 288 combinations are formally equivalent at δ = 0.01** (TOST),
236 of 288 at δ = 0.005.
`[outputs/ablation_tests.csv, outputs/ablation_equivalence.csv]`

Shuffled SHAP performs marginally *better* than real SHAP. The identity of which feature
receives which weight is irrelevant; only the magnitude matters.

### 6.2 The mechanism

Normalising the weight vector to unit sum makes each weight ≈ 1/F, so the transform is an
**undeclared preprocessing hyperparameter whose strength is a function of feature count**.
The multiplier reaches the model through **two independent channels**.

*Channel 1 — the concave transform.* The pipeline applies `log1p` *after* weighting.
Because `log1p(cx)` is not an affine function of `log1p(x)`, `RobustScaler` cannot undo the
multiplication, and the geometry of the feature space changes.

*Channel 2 — degenerate columns.* `RobustScaler` computes `(x − median)/IQR`, which for a
positive constant `c` satisfies `(cx − c·median)/(c·IQR) = (x − median)/IQR` — the scaling
is exactly cancelled. **Except when IQR = 0**, where scikit-learn sets the scale factor to
1 and the multiplier passes straight through untouched. Defect count features are sparse,
so this case is common rather than exotic: **three of five** Eclipse predictors (NMBFU,
NCBFU, NHPBFU) have zero IQR, as does one of twenty Apache features (`noc`).

Direct verification on Equinox, multiplying every column by 0.2 and re-scaling:

| Column | IQR | max \|difference\| after RobustScaler |
|---|---|---:|
| NBFU, NNTBFU | > 0 | **0.0000** (exactly cancelled) |
| NMBFU | 0 | 8.80 |
| NCBFU, NHPBFU | 0 | 3.20 |

*How we found channel 2.* We pre-registered a sharp test of channel 1 alone: with `log1p`
removed, weighting should have **no** effect. It failed — the effect weakened but did not
vanish (median |Δ| 0.0102 → 0.0029; 9 significant cells → 0). Investigating that failure
produced the two-channel account above, which explains both numbers, whereas the
single-channel account explains only the first. We report the failed prediction and the
revision rather than the revision alone. `[outputs/mechanism_ablation_verdict.csv]`

**Neither channel involves explanation information.** The revision strengthens rather than
weakens the conclusion: there are two distinct ways for an arbitrary multiplier to reach
the model, and SHAP is required by neither.

| | Features | Mean weight | Feature spread retained | Outcome |
|---|---:|---:|---:|---|
| Eclipse | 5 | 0.200 | 73% | small gain |
| Apache | 20 | 0.050 | 44% | significant loss |

A dose–response sweep over a constant multiplier c ∈ [10⁻³, 10] confirms performance is a
smooth function of c, with the SHAP-weighted variant lying on that curve.
`[outputs/dose_response.csv, outputs/dose_response.png]`

**A corrective normalisation.** Rescaling weights to unit *mean* rather than unit *sum*
preserves every importance ratio while removing the feature-count dependence. It
eliminates the harm on Apache (3 significant degradations → 0) and simultaneously
eliminates the benefit on Eclipse — leaving the transform inert. That the correction
removes the entire effect while preserving all importance information is independent
confirmation that the effect was never carried by the information.
`[outputs/scale_invariant_tests.csv]`

### 6.3 Feature selection

Selecting the top-half of features by importance fails against the same controls: 1
significant loss to random selection, 0 wins.
`[outputs/reconcile_selection_tests.csv]`

*A correction to our own earlier analysis:* an initial run reported four significant
losses. Re-running both selection strategies inside a single script on identical splits
revealed that the discrepancy arose from a mislabelled comparison in the intermediate
implementation. The corroborated figure is one significant loss; we report the weaker,
corrected number.

### 6.4 Redundancy-corrected selection, and why importance ranking fails

SHAP scores features individually, so it can assign high importance to several mutually
redundant features while discarding weak-but-complementary ones. Measured against 500
random subsets of equal size, the SHAP-selected set sits at the **88.4th, 96.6th and
97.2nd percentile** of internal correlation on the three twenty-feature projects; the
effect is absent at five features, where k = 3 leaves no room to concentrate.
`[outputs/redundancy_all_targets.csv]`

A controlled synthetic study confirms this where ground truth is known (features labelled
informative / redundant / repeated / noise by construction): SHAP's selected set captures
**19.1% of the available redundancy headroom beyond chance** (positive in 31/40
configurations, p = 1.7 × 10⁻⁵), and the bias is constant rather than growing with
redundancy. `[outputs/synthetic_redundancy_v2.csv]`

The obvious remedy — penalising redundancy (mRMR) — does not repair transfer: 1 win, 27
losses. The synthetic study explains why. mRMR does halve set redundancy as designed
(mean |corr| 0.121 vs 0.299, p < 0.0001), but the diversity penalty **over-corrects when
there is little redundancy to remove**, discarding informative features to buy diversity
that is not needed. Its advantage over plain importance ranking only appears at high
redundancy, so a fixed penalty across datasets of differing redundancy is the wrong
instrument. `[outputs/mrmr_tests.csv, outputs/synthetic_redundancy.csv]`

### 6.5 Shapley feature augmentation under distribution shift

We reimplement SFA [SFA] faithfully and transplant it, with its authors' own isolating
ablation, to zero-shot cross-project prediction. 6 targets × 6 models × 4 arms × 20
source bootstraps.

| Comparison | Significant | Wins | Losses |
|---|---:|---:|---:|
| `p_aug` vs `base` (plain stacking) | 21 | 5 | 16 |
| `ps_aug` vs `base` (full SFA) | 42 | 9 | 33 |
| **`ps_aug` vs `p_aug`** *(their ablation)* | 39 | **9** | **30** |

Beating `base` was ruled insufficient in advance: augmenting with a prediction is ordinary
stacking. The decisive comparison — do the Shapley values add anything beyond the
prediction — is **not supported**. Even plain prediction-augmentation degrades under
shift, indicating the two-stage strategy as a whole is fragile once source and target
differ. `[outputs/sfa_cross_project_tests.csv]`

**Antwarg et al.'s result stands where they established it.** Our finding bounds its
scope: the benefit is a within-distribution phenomenon.

### 6.6 Native model priors

Injecting importance into the estimator rather than the data (per-feature L2 penalty for
linear models; native `feature_contri` / `feature_weights` for boosters) also fails the
pre-registered gate (4 wins / 3 losses vs uniform; 3 / 5 vs shuffled).

Two observations are worth recording. First, scaling-based "priors" for SVC and
LogisticRegression fail as the mechanism predicts — they are the same multiply-the-features
operation. Second, LightGBM's `feature_contri`, which multiplies split *gain* and is
therefore not a rescaling, yields 6 significant wins and 0 losses, winning on all three
metrics against both controls on one target (velocity-1.6, 17–19 of 20 seeds). We report
this as **hypothesis-generating, not established**: the family-wide gate failed, five of
the six wins come from a single target, and reading a subgroup after a failed gate is not
evidence. It is, however, the only mechanism tested here that is not feature scaling, and
it warrants independent pre-registered replication.

We further record an implementation subtlety: XGBoost's `feature_weights` governs
column-*sampling* probability and is inert at the default `colsample_bytree = 1.0`. With
subsampling enabled the arm becomes active and yields a clean null (0/36 significant).
`[outputs/shap_prior_zeroshot_tests.csv]`

---

## 7. RQ3 — Cross-ecosystem generalisation

Repeating the full protocol on Apache/PROMISE (20 CK/OO metrics — a different ecosystem
*and* feature schema) does not merely fail to replicate; it reverses. Multiplicative
weighting produces **5 significant degradations and 0 improvements**, and CORAL
outperforms it. `[outputs/xe_significance_tests.csv]`

The only claim that survives is the one that is mechanically necessary: tree models are
unaffected (0/45 moved by more than one point).

---

### 6.7 Is the null informative? A power analysis

A null claim is only meaningful if the design could have detected an effect. We estimate
the minimum detectable effect (MDE) by resampling the *observed* paired-difference
distributions — preserving their real shape, skew and variance rather than assuming
normality — shifting them by candidate true effects, and applying the same paired Wilcoxon
test used throughout. 1,000 bootstrap resamples per candidate, α = 0.05, 80% power,
n = 20 splits, over all 288 cells.

| Quantity | Value |
|---|---:|
| Median MDE at 80% power | **≤ 0.0025** |
| Cells at or below the grid floor | 210 / 288 (73%) — true MDE is lower still |
| Cells never reaching 80% power | **0** |
| Median observed SHAP-vs-control effect | **0.000047** |
| Detection floor ÷ observed effect | **≈ 53×** |
| Cells where the observed effect exceeds its own MDE | 2 / 288 |
| CORAL's median detected effect, same harness | 0.0746 = **30× the MDE** |

`[outputs/power_analysis.csv, outputs/power_analysis_verdict.csv]`

The design resolves a true difference of a quarter of a percentage point with 80% power.
The observed difference between SHAP-derived weighting and a magnitude-matched control is
roughly fifty times smaller than that floor, while a domain-adaptation baseline in the
same harness produced effects thirty times *above* it and was duly detected. **The null is
informative: an effect large enough to matter would have been found.**

---

## 8. RQ4 — The positive control

Every result above was produced in a design where the classifier trains on the target's
own labels, importing only a weight vector. Transfer therefore had limited headroom. We
rebuilt the evaluation as genuine zero-shot CPDP — train on pooled source projects,
predict the entire target, no target labels anywhere — and re-ran the arms.

| Arm | Significant | Wins | Losses | Mean gain when winning |
|---|---:|---:|---:|---:|
| **CORAL** (domain adaptation) | 72 | **51** | 21 | **+0.080** |
| SHAP weighting | 15 | 8 | 7 | — |
| Uniform control | 20 | 8 | 12 | — |

CORAL's gains concentrate where distribution alignment should help (ROC-AUC 23/27
significant results are wins, mean +0.066). `[outputs/cpdp_significance.csv]`

> **In a setting where a standard domain-adaptation baseline achieves significant
> improvements in 51 of 144 tested combinations, SHAP-derived importance achieves 8 wins
> against 7 losses.** The null result is not an artefact of an insensitive design.

Note also that CORAL *harmed* performance in the earlier within-target design and helps
here — direct evidence that the conventional design gives transfer no headroom.

**An independent reason the weights cannot transfer.** Deriving the importance vector
separately from each source project and comparing them pairwise within an ecosystem gives
a median Spearman ρ of **0.462** overall (Apache 0.435; Eclipse 0.791, but computed over
only five features where the statistic is coarse). Below the pre-registered 0.5
threshold: the importance vector is itself source-unstable. `[outputs/loso_weight_stability.csv]`

---

## 9. RQ5 — The instrument is confounded

Explanation *stability* (σ̄, the mean across features of the standard deviation of LIME
weights across random seeds) is widely used to compare models. We show it largely measures
how well LIME's **linear surrogate** fits the model — a property of the model's smoothness,
not of explanation quality.

| Evidence | Result |
|---|---|
| 16 classifiers × 10 datasets, 2 ecosystems (160 pairs) | median ρ = **−0.646**, negative in **10/10**, pre-registered Wilcoxon **p = 0.002** |
| Split-half (R² and σ̄ from *disjoint* LIME seeds) | ρ = −0.656, 10/10, p = 0.002 — not estimation coupling |
| Within a single preprocessing pipeline (identical feature space) | ρ = **−0.833**, 10/10, p = 0.002 |
| 748 analysis specifications | **100%** negative |
| Non-software data (breast cancer, wine, digits) | 3/3 negative, median −0.591 |
| Causal intervention (tree depth swept, data fixed) | ρ = −0.907, **p = 1.4 × 10⁻⁸** |
| Analytic prediction (ridge coefficient variance ∝ residual variance ⇒ exponent 0.5) | measured **b = 0.470**, CI [0.431, 0.509] |

`[outputs/powered_h_verdict.csv, h_splithalf.csv, h_within_pipeline.csv,
h_specification_curve.csv, h_nonsdp.csv, causal_simulation_verdict.csv, derivation_fit.csv]`

**Consequence.** Ranking models by LIME stability ranks them substantially by linearity.
Stability is also uncorrelated with faithfulness (Kendall τ ≈ 0), so a "more stable"
explanation is not a more correct one.

**Limits we must state.** The analytic model explains only part of the variance
(fit R² = 0.259), so the mechanism is confirmed in form but not in full. A deconfounded
metric (σ̄ residualised on surrogate fit) did **not** validate: it tracks faithfulness
better in only 4 of 10 datasets (p = 0.336). We can therefore diagnose the confound but
cannot yet offer a validated replacement. Our recommendation is consequently a reporting
standard rather than a new metric: **explanation-stability figures must be accompanied by
surrogate fidelity (R²), and stability alone must not be used to rank models.**

---

## 10. Threats to Validity

**Internal.** SHAP is computed out-of-fold, removing the leakage present in the original
design. Controls are magnitude-matched and run within the same script and splits as the
treatment. Three implementation defects were found and corrected during the study and are
reported rather than silently fixed: a shuffled control that collapsed into the uniform
control by averaging permuted vectors; tree models forced through linear preprocessing;
and an inert XGBoost prior arm.

**External.** Two ecosystems and ten projects, all Java. Transfer to other languages or to
process-metric schemas is untested. The `digits` replication of the stability result is
weaker (ρ = −0.241 at 64 features) than the others, suggesting possible attenuation with
dimensionality.

**Construct.** ROC-AUC is primary for cross-project comparison [ROC]; PR-AUC is reported
but not used for cross-dataset ranking. The stability finding concerns LIME's linear
surrogate specifically and does not extend to SHAP-based stability measures.

**Our own failed predictions.** Three pre-registered predictions in the synthetic study
did not hold. One (P1) was a mis-specified test of a real effect — over-selection cannot
grow without bound, so testing for monotonic growth in a saturating quantity was invalid;
a corrected replication is reported alongside the original. One (P2) reversed, which
produced the mRMR explanation in Section 6.4. One (P3, Antwarg et al.'s conjecture that
Shapley variance discriminates informative from redundant features) is statistically
supported at n = 50 but with an effect so small (0.86% relative, Cohen's d = 0.33, correct
in 29/50 configurations) that it has no practical discriminative value.

---

## 11. Conclusion

SHAP attributions on these datasets are *faithful*: deleting features in the order SHAP
ranks them degrades predictions significantly faster than random ordering
(p = 0.0001 / 0.007). The explanations are honest descriptions of the models.

They are nonetheless not *transferable* in any of the five forms we tested. The gains
reported for explanation-guided feature weighting are reproduced exactly by multiplying
every feature by a constant, are formally equivalent to that constant under TOST, reverse
on a second ecosystem, and vanish when the feature-count dependence is removed. Feature
selection, redundancy-corrected selection, native priors and Shapley augmentation each
fail their own controls. Meanwhile a conventional domain-adaptation baseline achieves
large, significant gains in the same harness — so the setting supports transfer; SHAP-derived
importance simply does not supply it.

For practice, three recommendations follow. Report explanation-guided methods against
magnitude-matched controls, not merely against an untreated baseline. Do not normalise
importance vectors in a way that makes the transformation's strength depend on feature
count. And do not select models by LIME explanation stability without reporting surrogate
fidelity alongside it.

Explainability outputs may yet become active ingredients. On this evidence, multiplying
by them is not how.

---

## Data and Code Availability

All code, seeds, pre-registered decision rules and ~40 result artefacts are available in
the accompanying repository. Every table above is regenerable by the named script.

## References (to complete)

- **[SFA]** Antwarg, Galed, Shimoni, Rokach, Shapira. Shapley-based feature augmentation.
  *Information Fusion* 96 (2023) 92–102.
- **[ROC]** Richardson, Trevizani, Greenbaum, Carter, Nielsen, Peters. The receiver
  operating characteristic curve accurately assesses imbalanced datasets. *Patterns* 5
  (2024) 100994.
- **[APPX]** Kim. Improving appendix cancer prediction with SHAP-based feature
  engineering for machine learning models. *Ewha Med J* 48(2) (2025) e31.
- D'Ambros, Lanza, Robbes. Evaluating defect prediction approaches: a benchmark and an
  extensive comparison. *Empirical Software Engineering* 17(4) (2012) 531–577.
- Lundberg & Lee (SHAP, NeurIPS 2017); Ribeiro, Singh & Guestrin (LIME, KDD 2016);
  Sun & Saenko (CORAL, 2016); Shepperd et al. (NASA MDP data quality, TSE 2013).
- *(Carry over the remaining citations from the original manuscript's bibliography.)*
