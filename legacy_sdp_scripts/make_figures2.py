"""
Second figure set: the results that currently have no visual.

Fig A  five operationalisations, each against its own magnitude-matched control
Fig B  power analysis: detection floor vs observed effect vs CORAL reference
Fig C  cross-ecosystem reversal
Fig D  redundancy concentration, real data and controlled synthetic

Vector PDF for LaTeX. Palette and typography match make_figures.py.

Run: python make_figures2.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline as P

warnings.filterwarnings("ignore")
OUT = P.OUT_DIR
FIG = P.DATA_DIR.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.7, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, GREY, ACCENT, BAD = "#1a1a1a", "#8a8a8a", "#1f5c8b", "#a8443a"
GOOD = "#376a4f"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def counts(fname, comps):
    d = pd.read_csv(OUT / fname)
    sub = d[d.comparison.isin(comps)]
    sig = sub[sub["significant_at_0.05"]]
    return len(sub), int((sig.mean_diff > 0).sum()), int((sig.mean_diff < 0).sum())


# ------------------------------------------------------------------ Fig A
def fig_five_methods():
    rows = [
        ("Multiplicative\nweighting", "ablation_tests.csv",
         ["shap_vs_uniform", "shap_vs_shuffled"], "uniform + shuffled"),
        ("Feature\nselection", "reconcile_selection_tests.csv",
         ["plain_vs_random"], "random selection"),
        ("mRMR\nselection", "reconcile_selection_tests.csv",
         ["mrmr_vs_random", "mrmr_vs_shuffled"], "random + shuffled"),
        ("Native model\npriors", "shap_prior_zeroshot_tests.csv",
         ["prior_vs_uniform", "prior_vs_shuffled"], "uniform + shuffled"),
        ("Shapley\naugmentation", "sfa_cross_project_tests.csv",
         ["ps_vs_p"], "predictions only"),
    ]
    data = [(lbl, *counts(f, c), ctrl) for lbl, f, c, ctrl in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ys = np.arange(len(data))[::-1]
    for y, (lbl, n, w, l, ctrl) in zip(ys, data):
        ax.barh(y, w, height=0.5, color=ACCENT, edgecolor=INK, linewidth=0.6)
        ax.barh(y, -l, height=0.5, color="white", edgecolor=BAD,
                linewidth=0.9, hatch="////")
        ax.text(w + 0.7, y, f"{w} win{'' if w == 1 else 's'}", va="center",
                fontsize=7.5, color=ACCENT if w else GREY)
        ax.text(-l - 0.7, y, f"{l} loss{'' if l == 1 else 'es'}", va="center",
                ha="right", fontsize=7.5, color=BAD if l else GREY)
        ax.text(0, y + 0.36, f"vs {ctrl}   ({n} tests)", ha="center",
                fontsize=6.6, color=GREY, style="italic", zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=8)
    ax.set_xlabel("Holm-significant results against the method's own control")
    ax.set_title("No operationalisation of explanation-guided transfer "
                 "beats a control that contains no explanation", pad=8)
    lo = max(d[3] for d in data) + 12
    hi = max(d[2] for d in data) + 10
    ax.set_xlim(-lo, hi)
    ax.set_ylim(-0.6, len(data) - 0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)
    fig.tight_layout()
    save(fig, "fig_five_methods")
    print("  fig_five_methods    ", [(d[0].replace("\n", " "), d[2], d[3]) for d in data])


# ------------------------------------------------------------------ Fig B
def fig_power():
    r = pd.read_csv(OUT / "power_analysis.csv")
    mde = r.mde_80.dropna().values
    obs = r.observed_effect.abs().values
    try:
        c = pd.read_csv(OUT / "cpdp_significance.csv")
        coral = c[(c.arm == "coral_aligned") & (c["significant_at_0.05"]) &
                  (c.mean_diff > 0)].mean_diff.median()
    except Exception:
        coral = np.nan

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bins = np.logspace(-6, -0.6, 44)
    ax.hist(np.clip(obs, 1e-6, None), bins=bins, color=GREY, alpha=0.55,
            edgecolor="none", label="observed SHAP effect")
    ax.hist(mde, bins=bins, color=ACCENT, alpha=0.65, edgecolor="none",
            label="detection floor (80% power)")
    ax.axvline(np.median(obs), color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(np.median(obs), ax.get_ylim()[1] * 0.45,
            f" median\n observed\n {np.median(obs):.5f}",
            fontsize=6.6, color=INK, va="top")
    if np.isfinite(coral):
        ax.axvline(coral, color=GOOD, lw=1.2)
        ax.text(coral, ax.get_ylim()[1] * 0.55, f" CORAL\n {coral:.3f}",
                fontsize=7, color=GOOD, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("effect size (F1 / AUC / PR-AUC units, log scale)")
    ax.set_ylabel("number of cells")
    ax.set_title("The null is informative", pad=6)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center",
              bbox_to_anchor=(0.42, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    fig.tight_layout()
    save(fig, "fig_power")
    print(f"  fig_power            median MDE={np.median(mde):.4f} "
          f"median obs={np.median(obs):.6f} coral={coral:.4f}")


# ------------------------------------------------------------------ Fig C
def fig_cross_ecosystem():
    e = pd.read_csv(OUT / "significance_tests.csv")
    x = pd.read_csv(OUT / "xe_significance_tests.csv")
    e = e[e.comparison == "shap_vs_original"]
    x = x[x.comparison == "shap_vs_original"]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    groups = [("Eclipse\n(F = 5)", e, ACCENT), ("Apache\n(F = 20)", x, BAD)]
    for i, (lbl, d, col) in enumerate(groups):
        vals = d.mean_diff.values
        xj = np.random.RandomState(0).normal(i, 0.055, len(vals))
        ax.scatter(xj, vals, s=11, color=col, alpha=0.45, edgecolors="none")
        sig = d[d["significant_at_0.05"]]
        if len(sig):
            xs = np.random.RandomState(1).normal(i, 0.055, len(sig))
            ax.scatter(xs, sig.mean_diff.values, s=26, facecolors="none",
                       edgecolors=col, linewidths=1.0, zorder=3)
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color=INK, lw=1.6, zorder=4)
        ax.text(i + 0.27, np.mean(vals), f"mean {np.mean(vals):+.4f}",
                fontsize=7, color=INK, va="center", zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylabel("effect of SHAP weighting", labelpad=3)
    ax.set_title("The effect reverses on the second ecosystem", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    ax.text(0.01, 0.985, "hollow rings = Holm-significant", transform=ax.transAxes,
            fontsize=6.6, color=GREY, va="top", zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    fig.tight_layout()
    save(fig, "fig_cross_ecosystem")
    print(f"  fig_cross_ecosystem  eclipse={e.mean_diff.mean():+.4f} "
          f"apache={x.mean_diff.mean():+.4f}")


# ------------------------------------------------------------------ Fig D
def fig_redundancy():
    real = pd.read_csv(OUT / "redundancy_all_targets.csv")
    syn = pd.read_csv(OUT / "synthetic_redundancy_v2.csv")
    real = real[real.k > 1].sort_values("percentile")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    cols = [ACCENT if p >= 80 else GREY for p in real.percentile]
    ax.barh(np.arange(len(real)), real.percentile, color=cols,
            edgecolor=INK, linewidth=0.6, height=0.62)
    ax.axvline(50, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(50, len(real) - 0.35, " chance", fontsize=6.8, color=INK, va="top")
    ax.set_yticks(np.arange(len(real)))
    ax.set_yticklabels([f"{d}  ($F$={f})" for d, f in
                        zip(real.dataset, real.n_features)], fontsize=7)
    ax.set_xlabel("percentile vs 500 random subsets")
    ax.set_title("SHAP-selected sets are more\nredundant than chance", pad=6)
    ax.set_xlim(0, 104)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)

    ax = axes[1]
    m = syn[~syn.saturated].groupby("budget").excess.agg(["mean", "std", "count"])
    err = m["std"] / np.sqrt(m["count"]) * 1.96
    ax.errorbar(m.index, m["mean"], yerr=err, marker="o", ms=4.5, lw=1.2,
                color=ACCENT, capsize=3, elinewidth=0.8, capthick=0.8)
    sat = syn[syn.saturated].excess.mean()
    satx = syn[syn.saturated].budget.iloc[0]
    ax.scatter([satx], [sat], s=42, marker="s",
               facecolors="white", edgecolors=GREY, linewidths=1.1, zorder=4)
    ax.annotate("saturated:\npool already\n75% redundant",
                xy=(satx, sat), xytext=(satx - 1.2, 0.30),
                fontsize=6.3, color=GREY, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6,
                                shrinkB=4))
    ax.set_xlim(2, satx + 2.5)
    ax.axhline(0, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(0.4, 0.012, "chance", fontsize=6.8, color=INK)
    ax.set_xlabel("redundant features injected")
    ax.set_ylabel("normalised excess redundancy")
    ax.set_title("Controlled synthetic replication", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig_redundancy")
    print(f"  fig_redundancy       real n={len(real)} synthetic mean excess="
          f"{syn[~syn.saturated].excess.mean():+.3f}")


if __name__ == "__main__":
    print("Writing to", FIG)
    fig_five_methods()
    fig_power()
    fig_cross_ecosystem()
    fig_redundancy()
    print("Done.")
