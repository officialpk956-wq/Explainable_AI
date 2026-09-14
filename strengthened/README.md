# External documented-data evaluation

This is the primary implementation for the EMSE rewrite. The earlier `revised/`
experiment remains intact as supplementary exploratory evidence.

## Reproduce

Use Python 3.12 and the supplied root `requirements-audit.lock.txt`:

```
python -m pip install -r requirements-audit.lock.txt
python -m strengthened.download_data
python -m unittest strengthened.test_experiment
python -m strengthened.experiment --workers 2
python -m strengthened.analyze
python -m strengthened.build_paper
```

The experiment reuses only hash-matching checkpoints. To independently rerun the
same code, use a fresh extraction of the source artifact without the completed
`outputs/revised/strengthening/full` directory. Retain the original downloaded
data manifest and exact dependency versions. To explore changed code or protocol,
use a separate checkout so the completed run remains traceable.

Build the paper in `paper/emse` using `build.ps1` or pdfLaTeX, BibTeX and repeated
pdfLaTeX passes. `manuscript.tex.in` is the editorial source; `build_paper.py` fills
all result placeholders only after validation. Regeneration overwrites main.tex
and figure/table outputs. The official Springer class and bibliography style must
be retained in the paper directory. The journal upload ZIP flattens asset paths.

## Exact scope

- Seven Rnalytica JIRA families absent from the earlier experiment.
- Fifty-four static code metrics; the documented `RealBug` response.
- Each project held out; all other six supply 400 stratified source rows each.
- Three source samples (101–103), RF/ExtraTrees SHAP teachers, RF/LR learners.
- Three source-group tuning folds, representations rebuilt in each fold.
- Each arm independently selects one of two settings; shared-baseline sensitivity.
- Eight arms, two regimes, 42 checkpoints and 1,344 final score rows.
- No target-labelled fitting, transductive adaptation or SMOTE in the external run.
- Descriptive effects; shared training projects and overlapping seeds prevent
  interpreting all cells as independent observations.

The new selection and protocol were recorded before the new model results, after
the earlier study was inspected. This is not a public preregistration.

## Provenance and limits

Source: https://github.com/awsm-research/Rnalytica at immutable commit
`1931ca6537b00234b80e3d96f6c76c97524841f2`.
Dataset study: Yatish et al. (2019), DOI `10.1109/ICSE.2019.00075`.
The 54-feature allowlist agrees with the collector's code-metric schema, and the
response agrees with its loader. Metadata and download digests are supplied.
The upstream DESCRIPTION declares GPL (>=2); upstream content remains subject
to its own terms. The artifact provides a download script rather than repackaging
the upstream datasets under a new licence.

The study is retrospective cross-project classification, not calendar-forward
deployment. No record-level issue adjudication or historical label-availability
reconstruction was performed. The uncertain local Eclipse history inputs are not
used by this primary experiment.
