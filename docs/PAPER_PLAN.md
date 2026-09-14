# Research Plan: Building the Paper on the Cited Works' Own Open Questions

## The strategic insight

Antwarg et al. (*Information Fusion* 96, 2023) close their paper with four explicit
future-work items. **Three of them are questions we are already positioned to answer**,
and one of them is a question we have *already answered without knowing it*.

Verbatim from their Section 5:

> "In future work, we plan to: (1) examine whether recursively adding more layers of
> models ... (2) **analyze the usefulness of our method with different base learning
> algorithms.** Note that in such cases we will need to replace TreeExplainer with an
> appropriate method ... (3) **examine other XAI algorithms such as Integrated
> gradients, LIME and more**, and (4) **examine the contribution of Shapley values
> further using a synthetic dataset, in which feature variance and noise are
> controlled.** This allows exploration of whether Shapley values may be used to trace
> instances with problematic feature values, and **whether features for which there is
> high variance in the Shapley values distribution are given more attention than
> redundant features.**"

And Kim (2025) closes with:

> "external validation using institutional electronic health records is necessary to
> assess the model's generalizability"

**Positioning sentence for our paper:**

> Antwarg et al. (2023) established that Shapley values improve predictive performance
> when used as augmented features, and identified three open questions: whether the
> result generalises beyond decision forests, whether other XAI methods behave
> equivalently, and whether Shapley-based methods attend to redundant features under
> controlled conditions. We answer all three, and extend the evaluation to the setting
> their design excludes — distribution shift across projects — where we find the
> effect does not survive.

This is the strongest available framing: we are not attacking their paper, we are
**completing their stated research agenda** and reporting what it yields.

---

## What we already hold that answers them

| Their open question | Our existing evidence | Status |
|---|---|---|
| **FW2** — generality beyond decision forests | 16 classifiers spanning 8 families (linear, kernel, instance-based, probabilistic, discriminant, single-tree, bagging, boosting) across 10 datasets | Data exists; needs to be applied to *augmentation*, not just weighting |
| **FW3** — other XAI algorithms (LIME etc.) | Full LIME analysis: stability, surrogate fidelity R², faithfulness (AOPC), LIME↔SHAP agreement (Kendall τ ≈ 0) | Substantially done |
| **FW4** — synthetic controlled study; do Shapley methods attend to redundant features? | (a) `causal_simulation.py`: synthetic data, controlled model smoothness. (b) **Redundancy measurement: SHAP-top-k feature sets are *more* mutually correlated (0.288) than random subsets (0.236), 88th percentile** | **We have a direct answer to their question — and it is "yes, SHAP concentrates redundancy"** |
| Kim — external validation / generalisability | Two ecosystems (Eclipse 5-feature, Apache 20-feature CK/OO), 10 datasets | Done |

**FW4 is the headline opportunity.** They hypothesised that Shapley values might help
by *avoiding* redundant features. Our measurement shows the opposite: importance
ranking **concentrates** redundancy, and this is why SHAP-guided feature selection
underperformed random selection. That is a direct, empirical, contrary answer to a
question posed by a well-regarded paper in a strong venue.

---

## PHASE 1 — Corrections and consolidation (required, ~1 day)

**1.1 Metric correction (Richardson et al. 2024).**
Promote ROC-AUC to the primary cross-project metric; demote PR-AUC to a secondary,
within-project metric. Our targets span IR 0.57–9.80 (17×), so PR-AUC is not
comparable across them. Update `pipeline.py` (`HIGH_IMBALANCE` logic), README, and all
lead tables. Cite Richardson et al. for the decision.
*Conclusions unaffected — the falsification holds on all three metrics.*

**1.2 Finish the outstanding runs.**
Complete Task 6 (zero-shot / few-shot CPDP harness + LOSO weight vectors) and rerun
Task 7 (SHAP-as-prior) on top of it with all 6 targets and complete cells.

**1.3 Consolidate the redundancy result.**
Extend the redundancy percentile measurement from xalan-2.6 alone to all 6 targets
(`redundancy_all_targets.csv` — already specified, confirm it ran).

---

## PHASE 2 — Answer Antwarg FW4 directly: does Shapley attention track redundancy? (~1.5 days)

**This is the highest-value new work. It answers their exact question with a controlled
synthetic design they proposed themselves.**

**2.1 Synthetic redundancy experiment.**
`make_classification` with explicitly controlled structure: `n_informative` genuinely
informative features, `n_redundant` linear combinations of them, `n_repeated` exact
duplicates, plus pure noise features. Sweep the redundancy fraction from 0 to 0.6.

For each configuration measure:
- Does mean |SHAP| rank redundant features above genuinely informative but weaker ones?
- What fraction of the SHAP-top-k set is redundant/duplicated vs the ground-truth
  informative set?
- Is the **variance** of the per-instance Shapley distribution higher for informative
  than redundant features (their specific hypothesis)?

**Pre-registered predictions:**
- P1: as redundancy fraction rises, the proportion of SHAP-top-k that is redundant
  rises faster than chance.
- P2: SHAP-guided top-k selection recovers fewer *distinct* informative signals than
  redundancy-aware (mRMR) selection at equal k.
- P3 (their hypothesis, tested fairly): Shapley-variance is **not** systematically
  higher for informative than redundant features.

**Deliverable:** `outputs/synthetic_redundancy.csv`, plus one figure — redundancy
fraction on x, proportion of SHAP-top-k that is redundant on y, with the chance line.
This figure alone answers a published open question.

**2.2 Link to the real-data result.** Show the same pattern on all 6 real targets
(redundancy percentile) and on the VIF structure of the Eclipse data (NBFU/NNTBFU at
VIF > 150 are near-duplicates that SHAP ranks 1st and 2nd).

---

## PHASE 3 — Answer Antwarg FW2 + FW3, and extend SFA to distribution shift (~2 days)

**3.1 Implement SFA faithfully** (their two-stage method: OOF predictions + per-instance
Shapley values as augmented features), including their own controls: `base`,
`P augmented` (predictions only, no Shapley), and `PFA`.

**3.2 FW2 — base-learner generality.** They evaluated 3 decision forests and noted the
effect weakened for Random Forest without explanation. Evaluate across our 16
classifiers spanning 8 model families, using KernelExplainer where TreeExplainer does
not apply (they flagged this as the obstacle; we have the infrastructure).
**Contribution:** a mechanistic account of *why* XAI-derived transformations affect
model families unequally — scale-invariance for monotone rescaling, and surrogate-fit
dependence for LIME-derived quantities.

**3.3 FW3 — other XAI algorithms.** Repeat the augmentation with **LIME**-derived
attributions in place of Shapley values. We already know LIME attributions disagree
with SHAP (τ ≈ 0 on ranking, ρ = 0.15–0.60 on magnitude) and that LIME's surrogate
fidelity is below 0.6 for 7 of 8 non-linear models. **Prediction:** LIME-based
augmentation underperforms SHAP-based augmentation, and the gap tracks surrogate
fidelity. If confirmed, this is a mechanistic explanation for when XAI-based
augmentation works — genuinely new.

**3.4 The extension neither paper covers — cross-project SFA.**
Run SFA in the zero-shot transfer setting: source-model SHAP vectors computed on target
instances, appended as features, with their own `P augmented` control transplanted.
**This is the decisive experiment.** Antwarg et al.'s result is established
within-distribution; nobody has tested it under shift.
- If it survives: we report a **positive** result for augmentation and a negative one
  for weighting — a much stronger, more balanced paper.
- If it fails: the falsification extends to the strongest competing method *using its
  authors' own control*.

Either outcome is publishable. This is the highest-information experiment remaining.

---

## PHASE 4 — Kim (2025): the untested component (~0.5 day)

Reproduce Kim's three-step pipeline structure (selection → construction → weighting) on
our data and decompose the contribution of each step, with repeated splits and
significance tests. Their reported increments were +1.74 pp (selection), +0.12 pp
(construction), **+0.06 pp (weighting)** on LightGBM — a tree model.
**Prediction:** the weighting step contributes ~0 for tree models by scale-invariance,
and their own single-split numbers already show this.
**Contribution:** the first component-level ablation of a widely-recommended
"SHAP-based feature engineering" recipe.

---

## PHASE 5 — Assembly

**Paper structure:**
1. **Motivation** — XAI outputs as active ingredients; Antwarg et al. as the strongest
   positive precedent; the weighting family as the under-tested extension.
2. **RQ1 (FW2)** — does the benefit generalise across model families? *Mechanism:
   scale-invariance and surrogate-fit dependence.*
3. **RQ2 (FW3)** — do other XAI methods behave equivalently? *No; and LIME stability is
   confounded by surrogate fit — 10/10 datasets, p = 0.002, replicated on non-SDP data,
   causally demonstrated, theoretically predicted (b = 0.470 vs 0.5).*
4. **RQ3 (FW4)** — do Shapley methods attend to redundancy? *Yes, they concentrate it —
   synthetic + 6 real datasets.*
5. **RQ4 (new)** — does any of it survive distribution shift? *Weighting: no, across 4
   operationalisations with magnitude-matched controls, TOST-equivalent to a constant.
   Augmentation: [Phase 3.4 result].*
6. **Threats** — written from actual observed failures, not boilerplate.

**Venues:** *Information Fusion* is the natural target — it published Antwarg et al.,
and a paper that answers three of their stated open questions is squarely in scope.
Alternatives: EMSE, TOSEM, ESEM/PROMISE for the SDP-framed version.

---

## Effort summary

| Phase | Work | Cost | Risk |
|---|---|---|---|
| 1 | Metric fix, finish Tasks 6–7, consolidate redundancy | ~1 day | None |
| 2 | Synthetic redundancy (answers FW4) | ~1.5 days | Low — descriptive result either way |
| 3 | SFA implementation, FW2/FW3, cross-project SFA | ~2 days | Genuine — 3.4 could go either way |
| 4 | Kim component ablation | ~0.5 day | Low |
| 5 | Assembly | writing | Execution risk |

**Total: ~5 days of bench work.**

---

## Why this plan is stronger than "make our results better"

1. **We stop competing and start completing.** Answering a published paper's stated
   open questions is a recognised, reviewer-friendly contribution type, and it makes
   the citation relationship collaborative rather than adversarial.
2. **We already hold the answer to their hardest question (FW4)** — and it contradicts
   their hypothesis, which is exactly what makes it worth publishing.
3. **It converts our negatives into a balanced paper.** Currently everything is a
   falsification. Phase 3.4 gives a real chance of a positive result on the
   augmentation family, and Phase 3.3 gives a mechanistic *when-does-it-work* account.
4. **It fixes a real methodological error** (PR-AUC) with a citation, pre-empting a
   reviewer objection.
5. **The venue becomes obvious.** *Information Fusion* published the paper whose agenda
   we complete.
