"""Derive manuscript_v3.tex.in from the v2 template.

Introduction, background, data and most of the methods carry over unchanged. The
research questions, the results sections and the conclusion are rewritten, because
v3 adds a scale-sensitive learner, a structure-preserving augmentation control and
four predeclared sensitivities, and because the augmentation claim v2 made does
not survive them.
"""
import io
from pathlib import Path

HERE = Path(__file__).parent
src = io.open(HERE / 'manuscript_v2.tex.in', encoding='utf-8').read()

src = src.replace('% GENERATED-BY: strengthened.build_paper_v2 (external-static-v2)',
                  '% GENERATED-BY: strengthened.build_paper_v3 (external-static-v3)')

# ---- abstract -------------------------------------------------------------
old_abs_start = src.index(r'\textbf{Method:}')
old_abs_end = src.index(r'\keywords')
src = src[:old_abs_start] + r"""\textbf{Method:} Seven documented project families and 54 static code metrics are evaluated leave-one-project-out. Two attribution teachers, five downstream learners, ten source subsamples and two tuning regimes give @@score_records@@ scored predictions across @@arm_count@@ arms. Three arms are harness controls absent from prior explanation-transfer evaluations: a featureless reference, a source-label permutation preserving per-project class counts, and a documented adaptation comparator. Two further arms decompose augmentation: one replaces the attribution columns with width-matched independent noise, the other permutes whole attribution rows, preserving every column marginal and every inter-column correlation while destroying only the row correspondence. Because a positive per-feature rescaling cannot move an axis-aligned split, weighting is reported per learner and for the scale-sensitive learners separately. Four predeclared sensitivities vary teacher cross-fitting, model capacity, control draws and the source cap.
\textbf{Results:} The harness registers signal removal and addition: the untreated learner exceeds the featureless reference by @@ctl_prevalence@@ ROC-AUC, permuting source labels costs @@ctl_permuted@@, and the adaptation comparator gives @@ctl_coral@@. Weighting against a matched-magnitude constant is @@weight@@ over all learners but @@weight_scale@@ over the scale-sensitive learners alone, and is exactly zero for XGBoost by construction. Selection against an equal-sized random subset is @@selection@@. Augmentation is negative at @@augmentation@@; width-matched noise alone accounts for @@aug_noise_vs_p@@ of that, and the structure-preserving row permutation for @@aug_perm_vs_p@@. The residual attributable to the row correspondence is @@aug_shap_vs_perm@@, which is sign-stable under target omission and strengthens under project-grouped teacher cross-fitting, but @@aug_robust_short@@.
\textbf{Conclusions:} On this benchmark the tested attribution operations contribute little beyond matched controls, and the harness demonstrably responds to signal. The augmentation penalty is dominated by added width. @@aug_verdict_short@@ These are descriptive fixed-benchmark results, not equivalence findings, and target rotations share training projects, so they are not independent replications.
""" + src[old_abs_end:]

# ---- research questions ---------------------------------------------------
old = src[src.index(r'\textbf{RQ1:}'):src.index(r'Figure~\ref{fig:workflow} shows')]
src = src.replace(old, r"""\textbf{RQ1:} Does the evaluation harness register the removal and the addition of predictive signal, as measured by a featureless reference, a source-label permutation and a documented adaptation comparator?

\textbf{RQ2:} How do source-derived SHAP weighting and selection compare with controls matched on multiplier magnitude, multiplier distribution and subset size, and how much of the aggregate is determined by learners that cannot respond to a rescaling at all?

\textbf{RQ3:} Is the augmentation penalty attributable to the row-level attribution content, to the joint structure of the attribution columns, or to the added dimensions they occupy?

\textbf{RQ4:} Do the answers survive project-grouped teacher cross-fitting, larger model capacity, different control draws, a larger source cap, and the omission of any single target from the aggregate?

""")

# ---- results ---------------------------------------------------------------
res_start = src.index(r'\subsection{Attribution utility under matched controls (RQ2)}')
res_end = src.index(r'\section{Relationship to the earlier experiments}')
src = src[:res_start] + r"""\subsection{Attribution utility under matched controls (RQ2)}
Table~\ref{tab:baseline} gives untreated absolute performance and Table~\ref{tab:effects} the paired effects under both tuning regimes.

@@baseline_table@@

@@effects_table@@

@@effects_figure@@

A positive per-feature multiplier cannot change which threshold an axis-aligned split selects, so three of the five learners cannot respond to the weighting arms at all. Table~\ref{tab:bylearner} makes this explicit rather than averaging over it. @@learner_interpretation@@

@@bylearner_table@@

@@effects_interpretation@@

Secondary metrics need not order the arms identically, since average precision depends on prevalence and macro F1 uses a fixed threshold. Table~\ref{tab:secondary} reports them, and differences from the ranking metric are retained rather than omitted.

@@secondary_table@@

\subsection{What the augmentation penalty is made of (RQ3)}
Appending attribution columns beyond the prediction channel changes ROC-AUC by @@augmentation@@. That operation also adds @@aug_width@@ columns and imports whatever joint structure the attribution matrix carries, so the comparison alone identifies nothing. Two controls separate the three candidate explanations. Width-matched independent noise isolates the cost of dimensionality alone. The row permutation keeps every column marginal and every inter-column correlation of the real attribution matrix and destroys only the correspondence between a row and its own attributions, so the difference between it and the real arm is the row-level attribution content.

@@augmentation_table@@

@@augmentation_figure@@

@@augmentation_interpretation@@

\subsection{Sensitivity of the design and the aggregate (RQ4)}
Table~\ref{tab:loto} reports leave-one-target-out aggregation sensitivity and Figure~\ref{fig:loto} shows it graphically. @@loto_interpretation@@

@@loto_table@@

@@loto_figure@@

Four predeclared sensitivities vary the design itself rather than the aggregate. Table~\ref{tab:sensitivity} reports each against the main run restricted to the same seeds. @@sensitivity_interpretation@@

@@sensitivity_table@@

Repeat variability is separated into its sources. The mean within-cell standard deviation across source subsamples is @@repeat_sd@@ ROC-AUC for the selection comparison, while re-drawing only the control permutations with the source sample, teacher and downstream fits held fixed moves the augmentation contrast by up to @@control_spread@@. @@tuning_interpretation@@

Attribution rankings are compared across source subsamples and between teachers in Figure~\ref{fig:stability}; the mean between-teacher rank correlation is @@teacher_stability@@. A stable ranking can still be unhelpful as a multiplier, and a variable ranking can select similarly informative correlated metrics, so these diagnostics describe the fitted representation and are not offered as a mechanism.

@@stability_figure@@

""" + src[res_end:]

# ---- earlier experiments ---------------------------------------------------
src = src.replace(
    'An earlier execution of this benchmark used three source subsamples, two downstream learners and eight arms, without harness controls or a dimension-matched augmentation control.',
    'Two earlier executions of this benchmark exist. The first used three source subsamples, two downstream learners and eight arms, with no harness controls and no augmentation control. The second added harness controls and a width-matched noise control but still averaged weighting over learners that cannot respond to it, and its augmentation conclusion rested on a contrast that the structure-preserving control in this study shows to be the wrong one.')

io.open(HERE / 'manuscript_v3.tex.in', 'w', encoding='utf-8').write(src)
print('wrote manuscript_v3.tex.in')
print('placeholders:', sorted({s for s in src.split('@@')[1::2]}))
