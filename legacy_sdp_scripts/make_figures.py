"""
Publication figures for the paper. Vector PDF output for LaTeX.

Fig 1  the falsification: SHAP vs magnitude-matched controls, with CIs
Fig 2  the LIME confound: surrogate R^2 vs measured stability, 160 pairs
Fig 3  the positive control: CORAL vs SHAP in the same zero-shot harness

Run: python make_figures.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------- Fig 1
def fig_falsification():
    d = pd.read_csv(OUT / "ablation_scores.csv")
    order = ["original", "shap_sum1", "uniform_sum1", "shuffled_shap"]
    labels = ["Original\nno weighting", "SHAP\nweights",
              "Uniform\n$1/F$", "Shuffled\nSHAP"]
    colours = [GREY, ACCENT, INK, INK]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (eco, title) in zip(axes, [("eclipse", "Eclipse (F = 5)"),
                                       ("apache", "Apache (F = 20)")]):
        sub = d[d.ecosystem == eco]
        means = [sub[sub.variant == v]["f1"].mean() for v in order]
        errs = [ci95(sub[sub.variant == v]["f1"].values) for v in order]
        xs = np.arange(len(order))
        ax.bar(xs, means, yerr=errs, capsize=3, width=0.62,
               color=colours, edgecolor=INK, linewidth=0.7,
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
        base = means[0]
        ax.axhline(base, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=0)
        lo = min(m - e for m, e in zip(means, errs))
        hi = max(m + e for m, e in zip(means, errs))
        pad = (hi - lo) * 0.55 + 1e-3
        ax.set_ylim(lo - pad, hi + pad * 0.7)
        for x, m in zip(xs, means):
            ax.text(x, m + pad * 0.10, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=7.2, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.6)
        ax.set_title(title, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5)
    axes[0].set_ylabel("F1-macro (mean, 95% CI)")
    fig.text(0.5, -0.055, "Controls are magnitude-matched: uniform carries no importance information; "
             "shuffled keeps the SHAP magnitudes but attaches them to the wrong features.",
             ha="center", fontsize=7, color=GREY)
    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_falsification.{ext}")
    plt.close(fig)
    print("  fig_falsification  ", [round(m, 4) for m in means])


# ---------------------------------------------------------------- Fig 2
def fig_lime_confound():
    h = pd.read_csv(OUT / "h_splithalf.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    ds = sorted(h.dataset.unique())
    cmap = plt.get_cmap("tab10")
    rhos = {}
    for i, name in enumerate(ds):
        g = h[h.dataset == name]
        ax.scatter(g.r2_all, g.sigma_all, s=13, alpha=0.75,
                   color=cmap(i % 10), edgecolors="none", label=name)
        if len(g) > 2:
            z = np.polyfit(g.r2_all, np.log(g.sigma_all + 1e-12), 1)
            xs = np.linspace(g.r2_all.min(), g.r2_all.max(), 40)
            ax.plot(xs, np.exp(np.polyval(z, xs)), color=cmap(i % 10),
                    lw=0.8, alpha=0.55)
        rhos[name] = spearmanr(g.r2_all, g.sigma_all).statistic
    ax.set_yscale("log")
    ax.set_xlabel("LIME local surrogate fit ($R^2$)")
    ax.set_ylabel(r"measured instability $\bar{\sigma}$  (log scale)")
    ax.set_title("Better surrogate fit $\\rightarrow$ lower apparent instability", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="lower left",
              handletextpad=0.3, columnspacing=0.8)

    ax = axes[1]
    names = sorted(rhos, key=lambda k: rhos[k])
    vals = [rhos[n] for n in names]
    ax.barh(np.arange(len(names)), vals, color=ACCENT, edgecolor=INK,
            linewidth=0.6, height=0.68)
    ax.axvline(0, color=INK, lw=0.8)
    med = float(np.median(vals))
    ax.axvline(med, color=BAD, lw=1.0, ls=(0, (4, 2)))
    ax.text(med, len(names) - 0.3, f" median {med:.2f}", color=BAD,
            fontsize=7, va="top")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(r"Spearman $\rho(R^2,\ \bar{\sigma})$")
    ax.set_title("Negative in 10/10 datasets", pad=6)
    ax.set_xlim(min(vals) - 0.12, 0.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)

    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_lime_confound.{ext}")
    plt.close(fig)
    print("  fig_lime_confound   median rho =", round(med, 3))


# ---------------------------------------------------------------- Fig 3
def fig_positive_control():
    c = pd.read_csv(OUT / "cpdp_significance.csv")
    s = c[c["significant_at_0.05"]]
    arms = [("coral_aligned", "CORAL\n(domain adaptation)"),
            ("shap_sum1", "SHAP\nweighting"),
            ("uniform", "Uniform 1/F\ncontrol")]
    wins = [int((s[s.arm == a].mean_diff > 0).sum()) for a, _ in arms]
    loss = [int((s[s.arm == a].mean_diff < 0).sum()) for a, _ in arms]

    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    xs = np.arange(len(arms))
    ax.bar(xs - 0.19, wins, width=0.37, label="significant wins",
           color=ACCENT, edgecolor=INK, linewidth=0.7)
    ax.bar(xs + 0.19, loss, width=0.37, label="significant losses",
           color="white", edgecolor=BAD, linewidth=0.9, hatch="////")
    for x, w, l in zip(xs, wins, loss):
        ax.text(x - 0.19, w + 0.8, str(w), ha="center", fontsize=7.5, color=INK)
        ax.text(x + 0.19, l + 0.8, str(l), ha="center", fontsize=7.5, color=BAD)
    ax.set_xticks(xs)
    ax.set_xticklabels([lb for _, lb in arms], fontsize=7.4)
    ax.set_ylabel("Holm-significant results (of 144)")
    ax.set_title("Same harness, zero-shot CPDP", pad=6)
    ax.set_ylim(0, max(wins) * 1.22)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_positive_control.{ext}")
    plt.close(fig)
    print("  fig_positive_control wins =", wins, "losses =", loss)


if __name__ == "__main__":
    print("Writing figures to", FIG)
    fig_falsification()
    fig_lime_confound()
    fig_positive_control()
    print("Done.")
