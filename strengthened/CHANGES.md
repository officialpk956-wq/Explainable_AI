# Scientific revision record

The external selection record was written before the selected CSVs were downloaded;
the full computational protocol was written after schema inspection and before
new model execution. Both are exploratory records, not preregistration.

The smoke run used four-tree models and the first selected target. It passed;
all three isolation/control/data tests passed. The complete run used the fixed
32-tree protocol, three seeds and all seven project rotations. No full-run target
result was used to change the learner, teacher, grid or source cap.

After execution, the analysis and presentation code were added. They reconstruct
scores and summarise every predeclared arm and regime. Secondary-metric differences
and teacher-dependent changes were retained in the manuscript. Diagnostic plots
are descriptive and do not implement post hoc subgroup selection.

The primary/supplementary split follows the local data-provenance uncertainty:
the new primary experiment uses documented static metrics and excludes local
Eclipse history predictors. Matching JDT labels and Ant numeric contents to a
curated copy improves traceability but does not certify history-feature timing.

The prior `revised/` experiment files and `original/` directory were not rewritten
to make them look like the new execution. `strengthened/` contains the new protocol,
implementation, validations and manuscript generation.

## Four further sensitivity/generalization modules and their disclosed deviations

Four modules were added after the v3 main run and its four predeclared sensitivities
completed: `experiment_decorrelation.py`, `experiment_labelnoise.py`,
`experiment_linear_teacher.py` and `experiment_lime_teacher.py`. None edits
`experiment_v3.py` or `experiment.py` in place, because either file's own fingerprint
covers the already-completed v3 main run and all four of its sensitivities; editing
either would invalidate every one of those checkpoints. Where a module needed logic
`experiment_v3.py` does not expose as an overridable parameter, it copies the
relevant function bodies (`representations`/`fit_predict`/`task`) into its own file
instead, and imports everything unchanged (`learner`, `permute_within_groups`,
`streams`, `teacher`) rather than duplicating it. Each deviation from a literal
reading of its governing spec is disclosed below rather than left implicit.

`experiment_decorrelation.py` (decorrelated-feature sensitivity, protocol frozen in
`protocol_decorrelation.json` before execution) reuses `experiment_v3.config`/`task`/
`fingerprint` unchanged via a config override -- no new teacher or attribution logic
was needed, only a reduced feature list and arm set, both of which the existing code
already parameterises.

`experiment_labelnoise.py` (label-noise sensitivity, protocol frozen in
`protocol_labelnoise.json` before execution) needed a "labelnoise" random stream
distinct from the existing `labelperm` role. Extending `experiment_v3.ROLES` to add
it would have changed `experiment_v3.py`'s own hash. Instead the module derives an
independent RNG seeded from `(seed, noise level)` alone, entirely outside
`experiment_v3`'s stream bookkeeping -- a distinct stream in substance, implemented
outside `ROLES` rather than inside it.

`experiment_linear_teacher.py` (new: a LogisticRegression teacher explained with
`shap.LinearExplainer`, answering the reviewer threat that tree-SHAP is distorted by
correlated features) is exploratory, like v3 itself -- a first-time measurement with
no prior same-mechanism result to preregister a ceiling against, so it has no
protocol document and no predeclared threshold. Its internal preprocessing
(signed-log then RobustScaler, matching every other linear learner already in the
codebase) is a design choice for the teacher's own attribution computation only; the
resulting per-feature weight vector is applied to the original unscaled features,
exactly as the tree teacher's is. `shap.LinearExplainer`'s default
`feature_perturbation='interventional'` path silently subsamples its background to
100 rows; this was overridden to an explicit `shap.maskers.Independent` masker using
the full training fold, so the attribution does not depend on an internal RNG this
module does not seed or control. Reduced to the same six weighting/selection arms as
the two sensitivity modules above (dropping the augmentation and coral arms, which
are orthogonal to the attribution-mechanism question).

`experiment_lime_teacher.py` (new: the same RandomForest teacher as the main study,
explained with LIME instead of SHAP, isolating the attribution-algorithm question
from `experiment_linear_teacher.py`'s model-family question) is exploratory for the
same reason and likewise has no protocol document. BreakDown was not implemented
alongside LIME: neither `breakdown` nor `pyBreakDown` installs in this environment,
and this is disclosed here rather than silently dropped. LIME explains one instance
at a time rather than returning a whole batch in one call the way `TreeExplainer`
does; a benchmark against this study's own RandomForest teacher (32 trees, depth 5)
found roughly 0.03-0.06s/instance, which was used to size `LIME_NUM_SAMPLES=1000` and
confirm the full v3 grid (7 targets x 10 seeds, unchanged from the main study) was
tractable without a seed or target reduction -- unlike the two sensitivity modules
above, this one required no scope cut. Arms are named with a `lime_` prefix
(`lime_sum`, `select_lime`, `shuffled_lime`) rather than reusing the `shap_*` names,
so a `scores.csv` row is unambiguous about which attribution mechanism produced it
even outside this module's own output directory.

All four modules' results are reported in the manuscript's "Threats to validity"
section (Predeclared sensitivity outcomes; Attribution-mechanism generalization),
including the findings that do not favour the paper's original framing: the
decorrelated-feature selection effect is roughly seven times larger than the
54-feature estimate, and the linear-teacher selection effect loses leave-one-target-out
robustness that the tree-teacher and LIME estimates both keep.
