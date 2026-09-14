# Positioning Against the Four Supplied Papers

Note: two of the supplied PDFs (`shapely basrs feature augmentatin.pdf` and
`1-s2.0-S156625352300091X-main.pdf`) are the **same paper** — Antwarg et al.,
*Information Fusion* 96 (2023) 92–102. So there are three distinct works.

---

## 1. Antwarg, Galed, Shimoni, Rokach & Shapira (2023) — "Shapley-based feature augmentation", *Information Fusion*

**What they do.** SFA: a two-stage ensemble. Stage 1 produces out-of-fold (OOF)
predictions and their per-instance SHAP values; these are appended as *augmented
features*; stage 2 trains on original + augmented features. Evaluated on 20 OpenML
AutoML benchmark datasets with XGBoost, LightGBM and Random Forest, 5 train/test
splits, Wilcoxon signed-rank tests.

**What they establish.** SFA significantly beats `base`, Featuretools and PCA-Augment
(p < 0.01, all three algorithms). Critically, they *do* include the ablation that
isolates SHAP's contribution — `P augmented` (original features + OOF predictions,
**no** SHAP values). SFA significantly beats `P augmented` for XGBoost and LightGBM.

**This is a strong paper and we must not overclaim against it.** They ran the
right control and got a positive result. Our work does not contradict it.

**The gaps we legitimately occupy:**

| Their scope | Our scope |
|---|---|
| Explanations added as **features** (augmentation) | Explanations used as **importance weights / selection / priors** (transformation) |
| Single distribution — train and test from the same dataset | **Cross-project transfer** — importance derived on project A, applied to project B |
| Same-distribution OOF explanations available at inference | Target-domain explanations unavailable by construction |
| ROC-AUC only | F1-macro, ROC-AUC and PR-AUC |
| 5 splits | 20 repeated splits + Holm correction across families |
| Ablation = "remove SHAP entirely" | Ablation = **magnitude-matched controls** (uniform, shuffled) that keep the SHAP magnitudes but destroy the feature↔importance mapping |

**The one observation of theirs that supports our mechanism.** SFA's advantage over
`P augmented` holds for XGBoost and LightGBM but **not** for Random Forest (~60% of
datasets, not significant). They attribute this to instance/feature ratio. Our work
offers a mechanistic account of why XAI-derived quantities affect model families
unequally — and demonstrates the extreme case (monotone rescaling cannot alter tree
split selection at all).

**How to cite them.** As the strongest existing positive evidence that XAI outputs can
be *active ingredients* rather than post-hoc annotation — and as the correct
methodological benchmark, because they ran the isolating ablation. Then state our
distinct question: does that hold under **distribution shift**, and does it hold for
*importance-weighting* rather than *feature augmentation*?

---

## 2. Kim (2025) — "Improving appendix cancer prediction with SHAP-based feature engineering", *Ewha Med J* 48(2):e31

**What they do.** Kaggle appendix-cancer data (260k × 21). LightGBM. SHAP guides three
cumulative steps: (i) select top-15 features, (ii) construct interaction features,
(iii) **weight features by SHAP values**. Single 80:20 split, SMOTE, accuracy /
precision / recall / F1.

**Their reported results:**

| Configuration | Accuracy | Δ vs previous step |
|---|---:|---:|
| Baseline LightGBM | 0.8794 | — |
| + SHAP feature selection | 0.8968 | **+1.74 pp** |
| + SHAP feature construction | 0.8980 | +0.12 pp |
| + **SHAP feature weighting** | 0.8986 | **+0.06 pp** |

**This paper is the clearest example of the practice our work examines, and its own
numbers corroborate our mechanism.** The SHAP *weighting* step — applied to LightGBM,
a tree ensemble — contributes **+0.0006 accuracy**. Our analysis predicts exactly this:
multiplying feature columns by positive constants is a monotone rescaling, and tree
split selection is invariant to monotone rescaling, so the step *cannot* materially
change a tree model's decisions. The paper nonetheless reports feature weighting as a
contributing component of an effective framework.

**Methodological gaps we address:** a single train/test split (no repeated runs), no
significance testing of any kind, no confidence intervals, no magnitude-matched
control, and no ablation separating the weighting step's contribution from the
selection and construction steps that precede it.

**How to cite them.** Not as a target for criticism of the authors, but as evidence
that SHAP-based feature weighting is an *actively recommended practice* whose
component-level contribution has not been tested — and to note that their own
reported increment for the weighting step (+0.06 pp on a tree model) is consistent
with our finding that the operation is inert for tree ensembles.

---

## 3. Richardson, Trevizani, Greenbaum, Carter, Nielsen & Peters (2024) — "The receiver operating characteristic curve accurately assesses imbalanced datasets", *Patterns* 5:100994

**What they establish.** Via simulation and a real case study: ROC-AUC is *invariant*
to class imbalance when the score distribution is unchanged, whereas **PR-AUC changes
drastically with class imbalance and cannot be normalised or corrected to remove it**.
They recommend ROC-AUC precisely for *fair comparison across datasets with different
imbalance*.

**This paper corrects an error in our current pipeline.** We presently designate
PR-AUC as the primary metric for our high-imbalance datasets:

```python
HIGH_IMBALANCE = {"lucene", "pde"}
result["primary_metric"] = "PR-AUC" if name in HIGH_IMBALANCE else "F1-macro"
```

Our six target projects span an imbalance ratio range of **0.57 to 9.80 (a 17× spread)**:

| Project | IR (clean/buggy) |
|---|---:|
| poi-3.0 | 0.57 |
| xalan-2.6 | 1.15 |
| equinox | 1.51 |
| velocity-1.6 | 1.94 |
| pde | 6.16 |
| lucene | 9.80 |

Under Richardson et al., comparing PR-AUC across projects with such different base
rates conflates classifier quality with class balance. **Required change:** promote
ROC-AUC to the primary cross-project metric, retain PR-AUC as a secondary
within-project metric only, and cite Richardson et al. for the decision. This
strengthens our evaluation and removes a defensible reviewer objection.

**Note this does not alter any of our conclusions** — our falsification result holds
across F1, ROC-AUC and PR-AUC simultaneously — but it makes the metric methodology
defensible rather than conventional-but-wrong.

---

## 4. Haldar & Capretz (2024) — "Interpretable Software Defect Prediction from Project Effort and Static Code Metrics", *Computers* 13(2):52

**This is the closest paper to ours yet, and it is already cited in the original manuscript
as reference [14].** SHAP + LIME, cross-project defect prediction, PROMISE repository,
four classifiers (SVM, KNN, RF, ANN). It is peer-reviewed and open access.

**What they do.** Five NASA MDP datasets (cm1, kc1, kc2, pc1, jm1) with 21 McCabe/Halstead
static code metrics. Preprocessing removes missing values, implausible records, duplicates,
outliers beyond 1.5×IQR, and features correlated above 70%. Four classifiers are trained
per project; SHAP and LIME are applied to the best-performing models. "Cross-project"
reliability is assessed by building a dataset called **CP**.

**What is genuinely good.** They compare four classifiers rather than one, apply both SHAP
and LIME rather than only one, document their preprocessing criteria explicitly, and — this
matters — **publish their full per-model results tables rather than only the best numbers.**
Everything below is visible in their own Tables 6 and 7. The problem is framing and method,
not concealment.

### Four issues we improve on

**(a) The "cross-project" design does not test transfer.** Their CP dataset is created by
"merging all the selected files" into one dataframe with an added `project` column, then
split for evaluation. Test instances therefore come from the *same* projects as training
instances. This measures pooled within-project prediction with a project identifier, not
prediction on an unseen project. **We use genuine zero-shot CPDP:** train on pooled source
projects, predict an entirely held-out project, with no target labels at any point.

**(b) The models being explained are largely at chance.** Counting every model-project
combination in their own tables:

| | Table 6 (all features) | Table 7 (reduced features) |
|---|---:|---:|
| Combinations | 24 | 24 |
| AUC ≤ 0.52 (chance) | **13 / 24** | **17 / 24** |
| F1 = 0.00 (never predicts the positive class) | **8 / 24** | **10 / 24** |
| Median AUC | 0.52 | **0.50** |
| AUC range | 0.47–0.60 | 0.46–0.63 |

SHAP and LIME are then used to interpret these models, and the abstract concludes the
models "showed reliability on independent and cross-project data." **An explanation of a
model with AUC 0.50 describes a coin flip.** Our in-project models reach AUC 0.792 and our
zero-shot cross-project models 0.73+, so our explanations at least describe functioning
predictors — and we additionally verify the explanations are faithful (AOPC beats random
ordering, *p* = 0.0001), which no paper in this group does.

**(c) Accuracy is reported prominently under severe class imbalance.** Accuracies of
0.72–0.96 appear alongside F1 = 0.00. On pc1 (7.3% defect rate) a model that predicts
"not defective" for every module scores 0.95 accuracy. Following Richardson et al. [ROC] we
use ROC-AUC as the primary cross-project metric and never lead with accuracy.

**(d) Outlier removal appears to delete the positive class.** Their criterion drops any
record outside 1.5×IQR on any feature. Defective modules are frequently the large, complex
ones — precisely the upper tail. We tested this rule on the Jureczko/Apache PROMISE
projects we already hold:

| Project | Rows kept | Defective rows **deleted** |
|---|---:|---:|
| ant-1.7 | 58% | **71%** |
| camel-1.6 | 52% | **56%** |
| xalan-2.6 | 50% | **68%** |
| poi-3.0 | 43% | **53%** |
| velocity-1.6 | 49% | **55%** |
| **Mean** | **50%** | **61%** |

The rule retains half the data but removes **61% of defective modules** on average.
*Caveat: this is measured on the Jureczko sub-corpus, not on the NASA MDP files they use;
we demonstrate the mechanism on comparable static-metric defect data rather than on their
exact inputs.* It is a plausible contributor to the near-chance AUCs, and it is a
preprocessing step we deliberately do not apply.

**Additionally:** the NASA MDP family has documented labelling and data-quality problems
(Shepperd et al., TSE 2013) — cited in our own manuscript — which this paper does not
address. And there are no controls, no repeated splits, and no significance testing
anywhere; every number is a single point estimate.

### How to cite them

Not as a target. As **the clearest available evidence that the practice our paper examines
is standard, peer-reviewed, and under-validated**: explanations are generated and trusted
without checking that the underlying model works, without a control establishing that the
explanation contributes anything, and under a "cross-project" label that does not involve
transfer. Our contribution is precisely the missing validation layer.

This paper also strengthens our motivation section considerably: we can now point to a 2024
paper in the same venue class, on the same repository, doing the same thing, and show
concretely what goes wrong without controls.

---

## Consolidated positioning statement

> Antwarg et al. (2023) demonstrated that SHAP values can improve predictive
> performance when appended as augmented features, and — importantly — verified this
> against an ablation that removes the SHAP component. Their result is established
> for the **within-distribution** setting, where out-of-fold explanations of the same
> data-generating process are available at inference time. A separate line of applied
> work (e.g. Kim, 2025) extends the idea to SHAP-based *feature weighting*, in which
> feature columns are rescaled by their importance scores. This second family has not
> been subjected to the same scrutiny: reported gains are typically single-split,
> untested for significance, and unaccompanied by any control establishing that the
> *identity* of the importance values matters.
>
> We evaluate the weighting family under distribution shift, with magnitude-matched
> controls. Across two software ecosystems, four operationalisations (multiplicative
> weighting, feature selection, redundancy-corrected selection, and native model
> priors), 8–16 classifiers and 20 repeated splits with Holm correction, SHAP-derived
> importance is statistically indistinguishable from uniform and from randomly
> permuted importance of the same magnitude. We identify the mechanism — normalising
> weights to sum to unity makes the multiplier a function of feature count, which
> interacts with the pipeline's log transform — and show the effect is reproduced by
> multiplying every feature by an arbitrary constant.

---

## Concrete changes to make to our work

1. **Metric fix (required).** Promote ROC-AUC to primary for all cross-project
   comparisons; demote PR-AUC to secondary/within-project. Cite Richardson et al.
   Update `pipeline.py` (`HIGH_IMBALANCE` logic), the README, and re-render any
   table that leads with PR-AUC. Conclusions are unaffected.

2. **Add an augmentation arm (recommended).** Our falsification currently covers the
   *weighting* family. Antwarg et al.'s *augmentation* family is the strongest
   competing approach, and it is untested cross-project. Add one arm to the zero-shot
   harness: append source-model SHAP vectors (computed on target instances) as extra
   features, with `P augmented` (source predictions only, no SHAP) as the control —
   i.e. their own ablation, transplanted to the transfer setting. This directly
   answers "does the Antwarg result survive distribution shift?" and it is the single
   most valuable remaining experiment.

3. **Report per-model-family effects explicitly.** Antwarg et al. observed their
   effect weakening for Random Forest without a mechanistic account. Our
   scale-invariance argument supplies one. Frame this as a contribution: *why*
   XAI-derived transformations affect model families unequally.

4. **Cite Kim (2025) precisely and fairly.** Report their step-wise increments
   verbatim and note the +0.06 pp weighting increment on a tree model as
   consistent with mechanistic prediction. Do not characterise the paper as wrong —
   characterise the *component* as untested.

5. **Adopt their reporting strengths.** Antwarg et al. report per-dataset tables with
   standard deviations and Wilcoxon post-hoc tests; we should match that format, and
   we exceed it with Holm correction, pre-registration and equivalence testing (TOST).
