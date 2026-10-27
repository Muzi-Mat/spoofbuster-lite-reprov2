"""Stage 6: every number, table and figure in the write-up, computed from committed files only:
results/scores.csv.gz, results/timing.csv, results/minutiae_counts.csv.  Needs no dataset and no GPU.

Metric: TDR @ FDR = 0.2% (paper). FDR = fraction of bona fide images called spoof, TDR = fraction of
spoofs called spoof. With 1,000 bona fide test images the 0.2% threshold is set by the 3rd-highest
bona fide score, so it is noisy; we also report TDR @ FDR = 1% and a bootstrap error bar.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import get_args  # noqa: E402

# Paper, Table 2 (Redmi Note 4, quantised TFLite MobileNet-v1). "All" averages 35 minutiae per image.
PAPER = pd.DataFrame({"k": [5, 10, 15, 20, 25, 30, 0],
                      "time_ms": [148, 302, 447, 607, 736, 910, 1554],
                      "time_sd": [19, 15, 13, 9, 21, 18, 11],
                      "tdr": [93.9, 95.3, 95.3, 95.3, 95.7, 95.7, 95.7],
                      "tdr_sd": [1.1, 0.5, 0.5, 0.6, 0.5, 0.4, 0.1]})
PAPER_ALL_MINUTIAE = 35

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#888888"  # validated slots 1-3
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "lines.linewidth": 1.5, "legend.frameon": False})


def tdr_at_fdr(bona, spoof, fdr):
    """Threshold so that at most floor(fdr * n_bona) bona fide images score above it."""
    b = np.sort(bona)[::-1]
    n_allowed = int(np.floor(fdr * len(b)))
    thr = b[n_allowed] if n_allowed < len(b) else -np.inf
    return 100.0 * np.mean(spoof > thr), thr


def split_scores(df, col):
    return df.loc[df.label == 0, col].to_numpy(), df.loc[df.label == 1, col].to_numpy()


def summarise(scores, cfg):
    """One record per (model seed, method, k, scoring rule); mean +- sd over the k-means / random seeds."""
    out = []
    rng = np.random.default_rng(0)
    for mseed, sm in scores.groupby("model_seed"):
        base = sm[sm.method == "all"]
        b, s = split_scores(base, "score_w")
        tdr02_all, thr_all = tdr_at_fdr(b, s, cfg["fdrs"][0])
        tdr1_all, _ = tdr_at_fdr(b, s, cfg["fdrs"][1])
        # Test-set sampling error: resample bona fide and spoof images with replacement
        boot = [tdr_at_fdr(rng.choice(b, len(b)), rng.choice(s, len(s)), cfg["fdrs"][0])[0]
                for _ in range(cfg["bootstrap"])]
        out.append(dict(model_seed=mseed, method="all", k=0, rule="w", tdr02=tdr02_all, tdr02_sd=0.0,
                        tdr02_boot_sd=float(np.std(boot)), tdr1=tdr1_all, tdr1_sd=0.0,
                        fixed_tdr=tdr02_all, fixed_fdr=100 * np.mean(b > thr_all),
                        evals=base.n_evals.mean(), n_bona=len(b), n_spoof=len(s)))
        for (method, k), g in sm[sm.method != "all"].groupby(["method", "k"]):
            for rule in ("w", "u"):
                if method == "random" and rule == "u":
                    continue  # random patches have equal weights: the two rules coincide
                if method == "nearest" and rule == "u":
                    continue  # only the weighted version is needed for the fix attempt
                col = f"score_{rule}"
                per_seed = []
                for _, gs in g.groupby("seed"):
                    bb, ss = split_scores(gs, col)
                    per_seed.append((tdr_at_fdr(bb, ss, cfg["fdrs"][0])[0], tdr_at_fdr(bb, ss, cfg["fdrs"][1])[0],
                                     100 * np.mean(ss > thr_all), 100 * np.mean(bb > thr_all)))
                a = np.array(per_seed)
                out.append(dict(model_seed=mseed, method=method, k=k, rule=rule,
                                tdr02=a[:, 0].mean(), tdr02_sd=a[:, 0].std(ddof=1) if len(a) > 1 else 0.0,
                                tdr1=a[:, 1].mean(), tdr1_sd=a[:, 1].std(ddof=1) if len(a) > 1 else 0.0,
                                fixed_tdr=a[:, 2].mean(), fixed_fdr=a[:, 3].mean(), evals=g.n_evals.mean()))
    return pd.DataFrame(out)


def per_material(scores, cfg, k=10):
    rows = []
    sm = scores[scores.model_seed == scores.model_seed.min()]
    base = sm[sm.method == "all"]
    _, thr = tdr_at_fdr(*split_scores(base, "score_w"), cfg["fdrs"][0])
    for mat, g in base[base.label == 1].groupby("material"):
        km = sm[(sm.method == "kmeans") & (sm.k == k) & (sm.material == mat)]
        rows.append(dict(material=mat, n=len(g), all=100 * np.mean(g.score_w > thr),
                         kmeans=100 * np.mean(km.score_w > thr)))  # pooled over k-means seeds
    return pd.DataFrame(rows)


def pm(m, sd, digits=1):
    return f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def tables(summ, timing, mat, args, cfg):
    s0 = summ[summ.model_seed == summ.model_seed.min()]
    base = s0[s0.method == "all"].iloc[0]
    lat = timing.groupby("k").total_ms.agg(["mean", "std"]) if timing is not None else None

    # Table A: Table 2 side by side
    lines = [r"\begin{tabular}{r rr rrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Paper (Redmi Note 4)} & \multicolumn{4}{c}{Ours} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r"$k$ & ms & TDR@0.2 & CNN calls & ms (CPU) & TDR@0.2 & TDR@1 \\", r"\midrule"]
    for k in cfg["ks"] + [0]:
        p = PAPER[PAPER.k == k]
        r = base if k == 0 else s0[(s0.method == "kmeans") & (s0.k == k) & (s0.rule == "w")].iloc[0]
        ptime = f"{int(p.time_ms.iloc[0])}" if len(p) else "--"
        ptdr = pm(p.tdr.iloc[0], p.tdr_sd.iloc[0]) if len(p) else "--"
        ours_ms = f"{lat.loc[k, 'mean']:.0f}" if lat is not None and k in lat.index else "--"
        ours_tdr = pm(r.tdr02, r.tdr02_boot_sd if k == 0 else r.tdr02_sd)
        lab = "All" if k == 0 else str(k)
        lines.append(f"{lab} & {ptime} & {ptdr} & {r.evals:.1f} & {ours_ms} & {ours_tdr} & {r.tdr1:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.results, "table_main.tex"), "\n".join(lines) + "\n")

    # Table B: ablations at each k (model seed 0)
    lines = [r"\begin{tabular}{r rrrr}", r"\toprule",
             r"$k$ & centroid (paper) & unweighted & nearest minutia & random \\", r"\midrule"]
    for k in cfg["ks"]:
        get = lambda m, ru: s0[(s0.method == m) & (s0.k == k) & (s0.rule == ru)].iloc[0]  # noqa: E731
        a, b, n, c = get("kmeans", "w"), get("kmeans", "u"), get("nearest", "w"), get("random", "w")
        lines.append(f"{k} & {pm(a.tdr02, a.tdr02_sd)} & {pm(b.tdr02, b.tdr02_sd)} & {pm(n.tdr02, n.tdr02_sd)} & "
                     f"{pm(c.tdr02, c.tdr02_sd)} \\\\")
    lines += [r"\midrule", f"All & \\multicolumn{{4}}{{c}}{{{pm(base.tdr02, base.tdr02_boot_sd)} (bootstrap s.d.)}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.results, "table_ablation.tex"), "\n".join(lines) + "\n")

    # Table C: training seeds
    lines = [r"\begin{tabular}{c rrr}", r"\toprule",
             r"Model seed & All, TDR@0.2 & $k{=}10$, TDR@0.2 & $k{=}10$, fixed thr. TDR / FDR \\", r"\midrule"]
    for ms, g in summ.groupby("model_seed"):
        a = g[g.method == "all"].iloc[0]
        t = g[(g.method == "kmeans") & (g.k == 10) & (g.rule == "w")]
        if not len(t):
            continue
        t = t.iloc[0]
        lines.append(f"{ms} & {pm(a.tdr02, a.tdr02_boot_sd)} & {pm(t.tdr02, t.tdr02_sd)} & "
                     f"{t.fixed_tdr:.1f} / {t.fixed_fdr:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.results, "table_seeds.tex"), "\n".join(lines) + "\n")

    if timing is not None:
        g = timing.groupby("k")[["n_patches", "minutiae_ms", "select_ms", "patch_ms", "cnn_ms", "total_ms"]].mean()
        lines = [r"\begin{tabular}{r rrrrrr}", r"\toprule",
                 r"$k$ & patches & minutiae & select & cut & CNN & total \\", r"\midrule"]
        for k, r in g.iterrows():
            lines.append(f"{'All' if k == 0 else k} & {r.n_patches:.1f} & {r.minutiae_ms:.0f} & {r.select_ms:.1f} & "
                         f"{r.patch_ms:.1f} & {r.cnn_ms:.0f} & {r.total_ms:.0f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(os.path.join(args.results, "table_timing.tex"), "\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{l r rr}", r"\toprule", r"Material & $n$ & All & $k{=}10$ \\", r"\midrule"]
    for r in mat.itertuples():
        lines.append(f"{r.material.replace('_', ' ')} & {r.n} & {r.all:.1f} & {r.kmeans:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(args.results, "table_material.tex"), "\n".join(lines) + "\n")


def figures(summ, timing, cfg, args):
    s0 = summ[summ.model_seed == summ.model_seed.min()]
    base = s0[s0.method == "all"].iloc[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for method, colour, marker, lab in (("kmeans", BLUE, "o", "Ours: k-means centroids"),
                                        ("nearest", AQUA, "D", "Ours: nearest minutia"),
                                        ("random", ORANGE, "^", "Ours: random minutiae")):
        d = s0[(s0.method == method) & (s0.rule == "w")].sort_values("k")
        ax.errorbar(d.k, d.tdr02, yerr=d.tdr02_sd, color=colour, marker=marker, ms=4, capsize=2, label=lab)
    pk = PAPER[PAPER.k > 0]
    ax.errorbar(pk.k, pk.tdr, yerr=pk.tdr_sd, color=INK, marker="s", ms=3.5, ls="--", lw=1, capsize=2,
                label="Paper (Table 2)")
    ax.axhline(base.tdr02, color=BLUE, lw=1, ls=":")
    ax.axhline(95.7, color=INK, lw=1, ls=":")
    xmax = max(cfg["ks"] + [30])
    ours_above = base.tdr02 >= 95.7  # put each baseline label on the side away from the other line
    ax.text(xmax + 1, base.tdr02, "ours, all", va="bottom" if ours_above else "top", fontsize=6.5, color=INK)
    ax.text(xmax + 1, 95.7, "paper, all", va="top" if ours_above else "bottom", fontsize=6.5, color=INK)
    ax.set_xlim(2, xmax + 9)
    ax.set_xlabel("Patches per image, $k$")
    ax.set_ylabel("TDR (%) @ FDR = 0.2%")
    ax.set_xticks(sorted(set(cfg["ks"]) | set(PAPER.k[PAPER.k > 0])))
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "tdr_vs_k.pdf"))
    plt.close(fig)

    if timing is not None:
        g = timing.groupby("k")[["n_patches", "cnn_ms", "total_ms"]].mean()
        allrow = g.loc[0]
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        x = np.array([0, PAPER_ALL_MINUTIAE * 1.05])
        ax.plot(x / PAPER_ALL_MINUTIAE, x / PAPER_ALL_MINUTIAE, color=MUTED, lw=1, ls=":",
                label="Proportional to patch count")
        pn = PAPER.k.replace(0, PAPER_ALL_MINUTIAE) / PAPER_ALL_MINUTIAE
        ax.plot(pn, PAPER.time_ms / 1554, color=INK, marker="s", ms=3.5, ls="--", lw=1, label="Paper, total")
        on = g.n_patches / allrow.n_patches
        ax.plot(on, g.total_ms / allrow.total_ms, color=BLUE, marker="o", ms=4, label="Ours, total")
        ax.plot(on, g.cnn_ms / allrow.cnn_ms, color=ORANGE, marker="^", ms=4, label="Ours, CNN only")
        ax.set_xlabel("Patches, relative to all minutiae")
        ax.set_ylabel("Latency, relative to all minutiae")
        ax.legend(fontsize=6.5, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures, "relative_latency.pdf"))
        plt.close(fig)

    # Figure 9 analogue on a SYNTHETIC print: real dataset images must not be published
    from .minutiae import extract
    from .patches import select_points
    from .synthetic import fingerprint
    img = fingerprint(np.random.default_rng(3), size=360)
    feat = extract(img)
    c, _, w = select_points(feat, "kmeans", 10, 0)
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for a in axs:
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.axis("off")
    axs[0].scatter(feat["xy"][:, 0], feat["xy"][:, 1], s=6, facecolors="none", edgecolors=ORANGE, lw=0.8)
    axs[0].set_title(f"{len(feat['xy'])} minutiae", fontsize=7)
    for (x, y), wi in zip(c, w):
        axs[1].add_patch(plt.Rectangle((x - 48, y - 48), 96, 96, fill=False, ec=BLUE, lw=0.8))
        axs[1].text(x, y, f"{int(wi)}", color=INK, fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    axs[1].set_title("$k=10$ centroids, weight = cluster size", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "clusters_synthetic.pdf"))
    plt.close(fig)


def main():
    args, cfg = get_args(__doc__)
    scores = pd.read_csv(os.path.join(args.results, "scores.csv.gz"))
    tpath = os.path.join(args.results, "timing.csv")
    timing = pd.read_csv(tpath) if os.path.exists(tpath) else None
    summ = summarise(scores, cfg)
    mat = per_material(scores, cfg)
    summ.to_csv(os.path.join(args.results, "summary.csv"), index=False, float_format="%.3f")
    tables(summ, timing, mat, args, cfg)
    figures(summ, timing, cfg, args)

    s0 = summ[summ.model_seed == summ.model_seed.min()]
    base = s0[s0.method == "all"].iloc[0]
    k10 = s0[(s0.method == "kmeans") & (s0.k == 10) & (s0.rule == "w")].iloc[0]
    counts = pd.read_csv(os.path.join(args.results, "minutiae_counts.csv")).n_minutiae
    headline = dict(
        n_test_bona=int(base.n_bona), n_test_spoof=int(base.n_spoof),
        minutiae_mean=float(counts.mean()), minutiae_median=float(counts.median()),
        tdr02_all=float(base.tdr02), tdr02_all_boot_sd=float(base.tdr02_boot_sd),
        tdr02_k10=float(k10.tdr02), tdr02_k10_sd=float(k10.tdr02_sd),
        delta_k10=float(k10.tdr02 - base.tdr02),
        research_question_within_1pp=bool(k10.tdr02 >= base.tdr02 - 1.0),
        cnn_calls_all=float(base.evals), cnn_calls_k10=float(k10.evals),
    )
    if timing is not None:
        g = timing.groupby("k").total_ms.mean()
        headline["speedup_k10_total"] = float(g.loc[0] / g.loc[10]) if 10 in g.index else None
    with open(os.path.join(args.results, "headline.json"), "w") as f:
        json.dump(headline, f, indent=1)
    # LaTeX macros so the write-up text quotes these numbers directly instead of retyping them
    r10 = s0[(s0.method == "random") & (s0.k == 10)].iloc[0]
    u10 = s0[(s0.method == "kmeans") & (s0.k == 10) & (s0.rule == "u")].iloc[0]
    n10 = s0[(s0.method == "nearest") & (s0.k == 10)].iloc[0]
    macros = dict(NTestBona=f"{base.n_bona:d}" if isinstance(base.n_bona, int) else f"{int(base.n_bona)}",
                  NTestSpoof=f"{int(base.n_spoof)}", MinutiaeMean=f"{counts.mean():.1f}",
                  TDRall=f"{base.tdr02:.1f}", TDRallBootSD=f"{base.tdr02_boot_sd:.1f}",
                  TDRallOne=f"{base.tdr1:.1f}", TDRkTen=f"{k10.tdr02:.1f}", TDRkTenSD=f"{k10.tdr02_sd:.1f}",
                  TDRkTenOne=f"{k10.tdr1:.1f}", DeltaKTen=f"{k10.tdr02 - base.tdr02:+.1f}",
                  TDRrandTen=f"{r10.tdr02:.1f}", TDRrandTenSD=f"{r10.tdr02_sd:.1f}",
                  TDRunwTen=f"{u10.tdr02:.1f}", TDRunwTenSD=f"{u10.tdr02_sd:.1f}",
                  TDRnearTen=f"{n10.tdr02:.1f}", TDRnearTenSD=f"{n10.tdr02_sd:.1f}",
                  FixedTDRkTen=f"{k10.fixed_tdr:.1f}", FixedFDRkTen=f"{k10.fixed_fdr:.2f}",
                  CallsAll=f"{base.evals:.1f}")
    seeds = summ[(summ.method == "all")].set_index("model_seed").tdr02
    macros["TDRallSeedOne"] = f"{seeds.iloc[1]:.1f}" if len(seeds) > 1 else "--"
    if timing is not None:
        g = timing.groupby("k")[["total_ms", "cnn_ms", "minutiae_ms"]].mean()
        macros.update(MsAll=f"{g.loc[0, 'total_ms']:.0f}", MsKTen=f"{g.loc[10, 'total_ms']:.0f}",
                      SpeedupTotal=f"{g.loc[0, 'total_ms'] / g.loc[10, 'total_ms']:.1f}",
                      SpeedupCNN=f"{g.loc[0, 'cnn_ms'] / g.loc[10, 'cnn_ms']:.1f}",
                      MsMinutiae=f"{g.loc[0, 'minutiae_ms']:.0f}")
    write(os.path.join(args.results, "numbers.tex"),
          "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
