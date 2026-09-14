# Corrected research: implementation and execution status

The corrected code and full revised experiment have completed. The original snapshot has not changed.

- Primary: **40 checkpoints, 14,880 score records**, eight classifiers, six targets, twenty source repeats, five explanation-guided operations and their controls.
- Sensitivities: **120 checkpoints, 3,560 score records**; nested target tuning, exactly ten acquired target labels, complete log/scaler factorial, and dose response.
- Validation: every metric reconstructed from stored predictions; split separation, label budgets, grids, fingerprints and checkpoint hashes checked.
- Tests: **14 revised scientific regression tests + 6 historical regression tests passed**.
- Original: **all 70 files verified unchanged**.

## Deliverables

- [Revised Word manuscript](paper/revised/Do_Explanations_Transfer_Revised.docx)
- [Revised PDF manuscript](paper/revised/Do_Explanations_Transfer_Revised.pdf)
- [Revised Markdown manuscript](paper/revised/manuscript.md)
- [Execution instructions](revised/README.md)
- [Audit corrections](revised/CHANGELOG.md)
- [Primary results](outputs/revised/full/summary.csv)
- [Prediction/grid validation](outputs/revised/full/validation.json)
- [Sensitivity validation](outputs/revised/full/supplement/validation.json)
- [Release hashes](outputs/revised/full/release_manifest.json)

The Word paragraphs and four tables were structurally checked. The eight-page PDF was visually reviewed; short tables and their headings were kept together. PDF and Word are independently rendered from the same validated content, not a Word-to-PDF layout conversion.

The revised results do not preserve the old universal negative claim. Mean AUC differences include +0.000373 for weighting versus uniform scaling, +0.007800 for SHAP selection versus random selection, -0.006186 for adding SHAP beyond prediction augmentation, and +0.021587 for CORAL versus the untreated source learner. These are descriptive benchmark effects, not significance or equivalence claims. Target-level heterogeneity is retained.

The historical runner is `run_legacy.py`; `run_all.py` now runs the corrected package. Historical LIME/causal/synthetic claims were excluded from the revised main evidence rather than falsely presented as freshly reproduced. The supplementary model scope and protocol changes are disclosed.

## Still required before submission

Original dataset acquisition and temporal metadata, redistribution permission, a persistent public archive, venue requirements and final author declarations need confirmation. The local Apache files exist, but the user could not provide their original download provenance. The paper states these limits. This computational revision does not guarantee acceptance or prove the absence of every possible software defect.

## LaTeX submission package (2026-09-10)

The latest editorial package is `paper/submission/`, with compiled author and anonymous
PDFs (11 pages), complete LaTeX/BibTeX, five vector/PNG figures, seven tables, result
CSVs and a corrected-source snapshot. `paper/Do_Explanations_Transfer_LaTeX.zip`
contains 58 files and was checked against SHA-256 hashes. Compilation has no overfull
boxes or unresolved references. All 1,584 paired effects were reconstructed from
primary score records, and 48 baseline cells across three metrics were checked.
The original 70-file snapshot remains unchanged.

This is a prepared manuscript package, not guaranteed acceptance. Dataset observation
windows/provenance, author declarations, a public research archive and venue-specific
requirements remain in `paper/submission/SUBMISSION_CHECKLIST.md`.
