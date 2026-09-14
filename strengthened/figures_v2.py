"""Figures for the v2 manuscript. Vector PDF plus EPS and a high-resolution PNG.

Every panel is descriptive. No error bars are drawn anywhere, because the repeats
are overlapping source subsamples of a shared project pool and a bar across them
would read as an interval it is not.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment_v2 import OUT, config

DEST = Path(__file__).resolve().parents[1] / 'paper/emse'
INK, GREY, ACCENT, WARN = '#1a1a1a', '#8a8a8a', '#1f5c8b', '#a8443a'

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .8})


def save(fig, n):
    (DEST / 'figures').mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'eps', 'png']:
        fig.savefig(DEST / 'figures' / f'Fig{n}.{ext}',
                    dpi=600 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def short(name):
    return name.split('-')[0].split('_')[0].title()


def fig_workflow(cfg, n=1):
    fig, ax = plt.subplots(figsize=(5.1, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis('off')
    boxes = [(2.2, 3.9, 'Six source projects\n400 sampled rows each'),
             (7.6, 3.9, 'Three source-project folds\nteacher and controls rebuilt inside each'),
             (2.2, 2.1, 'Cross-fitted teacher\nRandomForest / ExtraTrees'),
             (7.6, 2.1, 'Twelve arms: treatments,\nmatched controls, harness controls'),
             (4.9, .5, 'Held-out project: predict, then score\nresponses enter scoring only')]
    for x, y, t in boxes:
        ax.text(x, y, t, ha='center', va='center', fontsize=7.6,
                bbox=dict(boxstyle='round,pad=.42', fc='white', ec=INK, lw=.8))
    for a, b in [((4.1, 3.9), (5.5, 3.9)), ((2.2, 3.45), (2.2, 2.55)),
                 ((7.6, 3.45), (7.6, 2.55)), ((4.2, 2.1), (5.5, 2.1)),
                 ((7.6, 1.65), (5.6, .85)), ((2.2, 1.65), (4.2, .85))]:
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'lw': .9, 'color': INK})
    fig.tight_layout(); save(fig, n)


def fig_harness(scores, cfg, n=2):
    """The sensitivity ladder: does the harness move when signal is removed or added?"""
    order = [('prevalence', 'Featureless\n(source rate)'), ('permuted_original', 'Permuted\nsource labels'),
             ('original', 'Untreated'), ('coral', 'CORAL\n(transductive)')]
    s = scores[scores.tuning == 'independent']
    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    targets = cfg['projects']
    for i, (arm, label) in enumerate(order):
        vals = s[s.arm == arm].groupby('target').auc.mean().reindex(targets)
        ax.scatter(np.full(len(vals), i) + np.linspace(-.16, .16, len(vals)), vals,
                   s=20, color=GREY, zorder=2)
        ax.hlines(vals.mean(), i - .3, i + .3, color=INK, lw=2.2, zorder=3)
        ax.text(i + .34, vals.mean(), f'{vals.mean():.3f}', ha='left', va='center',
                fontsize=7.6, color=INK,
                bbox=dict(boxstyle='square,pad=.1', fc='white', ec='none'))
    ax.axhline(.5, color=WARN, lw=.9, ls=(0, (4, 3)))
    ax.text(3.42, .505, 'chance', fontsize=7.2, color=WARN, va='bottom', ha='right')
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=8)
    ax.set_ylabel('ROC-AUC on the held-out project')
    ax.set_xlim(-.5, 3.85)
    fig.tight_layout(); save(fig, n)


def fig_effects(effects, cfg, comparisons, n=3):
    targets = cfg['projects']
    p = effects[(effects.metric == 'auc') & (effects.tuning == 'independent')]
    fig, axes = plt.subplots(1, len(comparisons), figsize=(5.1, 3.4), sharey=True)
    for i, (ax, (a, b, title)) in enumerate(zip(np.atleast_1d(axes), comparisons)):
        d = p[(p.arm == a) & (p.control == b)].groupby('target').mean_difference.mean().reindex(targets)
        ax.scatter(d, range(len(targets)), s=26, color=INK, marker='osd^'[i % 4], zorder=3)
        ax.axvline(0, color=GREY, lw=.8, ls=(0, (4, 3)))
        ax.set_yticks(range(len(targets))); ax.set_yticklabels([short(t) for t in targets], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('$\\Delta$ ROC-AUC', fontsize=8)
        ax.set_title(title, fontsize=8.5)
        lim = max(.012, float(d.abs().max()) * 1.25); ax.set_xlim(-lim, lim)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(); save(fig, n)


def fig_loto(loto, comparisons, n=5):
    """Leave-one-target-out aggregation sensitivity. Not an interval."""
    d = loto[(loto.metric == 'auc') & (loto.tuning == 'independent')]
    fig, ax = plt.subplots(figsize=(5.1, 2.9))
    labels = []
    for i, (a, b, title) in enumerate(comparisons):
        r = d[(d.arm == a) & (d.control == b)]
        if not len(r):
            continue
        r = r.iloc[0]
        lo, hi, mid = r.loto_min * 1e3, r.loto_max * 1e3, r.all_targets * 1e3
        ax.hlines(i, lo, hi, color=GREY, lw=3, zorder=2)
        ax.scatter([mid], [i], s=42, color=INK, zorder=4,
                   label='all seven targets' if not i else None)
        ax.scatter([lo, hi], [i, i], s=18, color=WARN, zorder=3,
                   label='most extreme omission' if not i else None)
        labels.append(title)
    ax.axvline(0, color=INK, lw=.9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    # values span a few thousandths; unscaled ticks collide on a column-width figure
    ax.set_xlabel('$\\Delta$ ROC-AUC, equal-target mean ($\\times 10^{-3}$)', fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.2, frameon=False, loc='upper center',
              bbox_to_anchor=(.5, -.3), ncol=2)
    fig.tight_layout(); save(fig, n)


def fig_augmentation(aug, cfg, n=4):
    """Augmentation against both controls, along every reported dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 3.0))
    panels = [('target', 'By target'), ('model', 'By learner'), ('teacher', 'By teacher')]
    pairs = [('ps_aug', 'p_aug', 'vs prediction only', INK, 'o'),
             ('ps_aug', 'pn_aug', 'vs matched noise', ACCENT, 's')]
    for ax, (dim, title) in zip(axes, panels):
        levels = None
        for a, b, lab, colour, marker in pairs:
            d = aug[(aug.metric == 'auc') & (aug.arm == a) & (aug.control == b) &
                    (aug.dimension == dim)]
            if not len(d):
                continue
            d = d.sort_values('level')
            levels = list(d.level)
            ax.scatter(d.mean_difference, range(len(d)), s=22, color=colour, marker=marker,
                       label=lab, zorder=3)
        ax.axvline(0, color=GREY, lw=.8, ls=(0, (4, 3)))
        if levels:
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([short(str(l)) if dim == 'target' else str(l) for l in levels],
                               fontsize=7)
        ax.invert_yaxis(); ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('$\\Delta$ ROC-AUC', fontsize=8); ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.4, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.55, 1.03))
    fig.tight_layout(rect=(0, 0, 1, .92)); save(fig, n)


def fig_stability(stability, cfg, n=6):
    targets = cfg['projects']
    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    series = [('RandomForest across seeds', 'o', INK, 'RF teacher, across source samples'),
              ('ExtraTrees across seeds', 's', GREY, 'ET teacher, across source samples'),
              ('RandomForest vs ExtraTrees teacher', '^', ACCENT, 'between teacher means')]
    for off, (name, marker, colour, label) in zip([-.18, 0, .18], series):
        d = stability[stability.comparison == name].set_index('target').mean_spearman.reindex(targets)
        ax.scatter(d, np.arange(len(targets)) + off, marker=marker, s=26, color=colour, label=label)
    ax.set_yticks(range(len(targets))); ax.set_yticklabels([short(t) for t in targets], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-.05, 1.03)
    ax.set_xlabel('Spearman rank correlation between attribution vectors', fontsize=8)
    ax.legend(fontsize=7.2, frameon=False, loc='upper center',
              bbox_to_anchor=(.5, -.18), ncol=3)
    fig.tight_layout(); save(fig, n)


def build_all(comparisons):
    cfg = config()
    scores = pd.read_csv(OUT / 'scores.csv')
    fig_workflow(cfg, 1)
    fig_harness(scores, cfg, 2)
    fig_effects(pd.read_csv(OUT / 'effects.csv'), cfg, comparisons, 3)
    fig_augmentation(pd.read_csv(OUT / 'augmentation.csv'), cfg, 4)
    fig_loto(pd.read_csv(OUT / 'loto.csv'), comparisons, 5)
    fig_stability(pd.read_csv(OUT / 'stability.csv'), cfg, 6)
    return 6
