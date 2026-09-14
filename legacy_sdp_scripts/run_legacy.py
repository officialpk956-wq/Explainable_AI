"""Dependency-ordered experiment runner. Use --list before a costly full run.

Existing CSVs are cached results, not evidence of fresh reproduction. --force
reruns stages but preserves script-level tuning caches. Logs record each run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
# Order is significant: weights precede consumers; corrected scripts supersede
# legacy scripts that write incompatible schemas to the same output filename.
STAGES = [
 ('diagnostics', 'manuscript_diagnostics', [], ['audit/dataset_summary.csv', 'audit/iqr_removal.csv', 'audit/scaler_diagnostics.csv']),
 ('eclipse', 'pipeline', [], ['shap_weights.csv','phase1_cv_eclipse.csv','phase1_cv_mylyn.csv','phase2_repeated_scores.csv','significance_tests.csv','xai_faithfulness_tests.csv','xai_lime_shap_agreement.csv','explainability_scores.csv','surrogate_bias.csv','surrogate_bias_tests.csv']),
 ('apache', 'cross_ecosystem', [], ['xe_shap_weights.csv','xe_phase2_repeated_scores.csv','xe_significance_tests.csv']),
 ('ablation', 'ablation_control', ['eclipse','apache'], ['ablation_scores.csv','ablation_tests.csv']),
 ('equivalence', 'equivalence_tests', ['ablation'], ['ablation_equivalence.csv']),
 ('normalization', 'scale_invariant_fix', ['eclipse','apache'], ['scale_invariant_scores.csv','scale_invariant_tests.csv']),
 ('selection', 'shap_feature_selection', ['eclipse','apache'], ['feature_selection_scores.csv','feature_selection_tests.csv']),
 ('mrmr', 'mrmr_shap', ['eclipse','apache'], ['mrmr_scores.csv','mrmr_tests.csv','mrmr_verdict.csv']),
 ('selection_reconciled', 'reconcile_selection', ['eclipse','apache'], ['reconcile_selection_scores.csv','reconcile_selection_tests.csv','reconcile_verdict.csv']),
 ('dose', 'dose_response', ['eclipse','apache'], ['dose_response_raw.csv','dose_response.png']),
 ('mechanism', 'mechanism_ablation', ['eclipse','apache'], ['mechanism_ablation_scores.csv','mechanism_ablation_tests.csv','mechanism_ablation_verdict.csv']),
 ('cpdp', 'genuine_cpdp', ['eclipse','apache'], ['cpdp_zeroshot_scores.csv','cpdp_fewshot_scores.csv','cpdp_significance.csv','loso_weight_vectors.csv','loso_weight_stability.csv']),
 ('prior_fewshot', 'shap_as_prior_fixed', ['eclipse','apache'], ['shap_prior_scores.csv','shap_prior_fewshot_scores.csv','shap_prior_tests.csv','shap_prior_verdict.csv']),
 ('prior_zeroshot', 'shap_prior_zeroshot', ['eclipse','apache'], ['shap_prior_zeroshot_scores.csv','shap_prior_zeroshot_tests.csv','shap_prior_zeroshot_verdict.csv','shap_prior_group_audit.csv']),
 ('sfa', 'sfa_cross_project', ['eclipse','apache'], ['sfa_cross_project_scores.csv','sfa_cross_project_tests.csv','sfa_cross_project_verdict.csv']),
 ('synthetic_original', 'synthetic_redundancy', [], ['synthetic_redundancy.csv','synthetic_redundancy_verdict.csv']),
 ('synthetic_corrected', 'synthetic_redundancy_v2', [], ['synthetic_redundancy_v2.csv','synthetic_redundancy_v2_verdict.csv']),
 ('stability', 'powered_phase_h', [], ['powered_h_measurements.csv','powered_h_per_dataset.csv','powered_h_verdict.csv']),
 ('bias_free', 'bias_free_stability', ['eclipse'], ['bias_free_stability.csv','ranking_agreement.csv','ranking_agreement_tau.csv']),
 ('stability_robustness', 'h_robustness', ['eclipse','apache'], ['h_splithalf_raw.csv','h_splithalf.csv','h_splithalf_verdict.csv','h_within_pipeline.csv','h_specification_curve.csv']),
 ('theory', 'stability_theory', ['stability_robustness'], ['derivation_fit.csv']),
 ('deconfounded', 'deconfounded_stability_tasks', ['stability_robustness'], ['aopc_full_grid.csv','deconfounded_stability.csv','deconfounded_validation.csv','deconfounded_verdict.csv']),
 ('convergence', 'stability_convergence', [], ['stability_convergence.csv','convergence_rank_swaps.csv']),
 ('redundancy', 'redundancy_all_targets', ['eclipse','apache','stability_robustness'], ['redundancy_all_targets.csv']),
 ('causal', 'causal_simulation', [], ['causal_simulation.csv','causal_simulation_raw.csv','causal_simulation_verdict.csv']),
 ('nonsdp', 'h_nonsdp', [], ['h_nonsdp.csv','h_nonsdp_verdict.csv']),
 ('power', 'power_analysis', ['ablation','cpdp'], ['power_analysis.csv','power_analysis_verdict.csv']),
 ('figures', 'make_figures', ['ablation','stability_robustness','cpdp'], ['../paper/figures/fig_falsification.pdf','../paper/figures/fig_lime_confound.pdf','../paper/figures/fig_positive_control.pdf']),
 ('figures_extended', 'make_figures2', ['power','sfa','prior_zeroshot','selection_reconciled','normalization','redundancy','synthetic_corrected'], ['../paper/figures/fig_five_methods.pdf','../paper/figures/fig_power.pdf','../paper/figures/fig_cross_ecosystem.pdf','../paper/figures/fig_redundancy.pdf']),
]

def select_stages(names):
    known = {s[0]: s for s in STAGES}
    selected = set()
    def visit(name):
        if name not in known:
            raise ValueError(f'Unknown stage: {name}')
        if name not in selected:
            selected.add(name)
            for dep in known[name][2]:
                visit(dep)
    for name in names or known:
        visit(name)
    return [s for s in STAGES if s[0] in selected]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Show stages and cache status without running')
    parser.add_argument('--stage', action='append', help='Run selected stage and dependencies; repeatable')
    parser.add_argument('--force', action='store_true', help='Rerun selected stages; internal tuning caches remain')
    args = parser.parse_args()
    try:
        stages = select_stages(args.stage)
    except ValueError as e:
        parser.error(str(e))
    records = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logdir = ROOT/'outputs'/'run_logs'/stamp
    for name, module, deps, outputs in stages:
        paths = [ROOT/'outputs'/p for p in outputs]
        cached = all(p.is_file() and p.stat().st_size > 0 for p in paths)
        print(f'{name:24} {module + ".py":34} {"cached (unverified)" if cached else "missing outputs"}', flush=True)
        if args.list:
            continue
        if not (ROOT/f'{module}.py').is_file():
            raise FileNotFoundError(module)
        if cached and not args.force:
            records.append(dict(stage=name, status='cached_unverified'))
            continue
        logdir.mkdir(parents=True, exist_ok=True)
        with (logdir/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, str(ROOT/f'{module}.py')], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
        ok = result.returncode == 0 and not missing
        records.append(dict(stage=name, status='completed' if ok else 'failed',
                            returncode=result.returncode, missing=missing))
        (logdir/'run_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        if not ok:
            print(f'FAILED: see {logdir/name}.log; missing={missing}', file=sys.stderr)
            return 1
    if not args.list:
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir/'run_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print('Run record:', logdir/'run_manifest.json')
    return 0

if __name__ == '__main__':
    sys.exit(main())
