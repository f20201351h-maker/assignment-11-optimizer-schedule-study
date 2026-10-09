"""
Experiment 5: LR sweep at widths 256 / 512 / 1024, minima, and the LR for width 4096.

Selection metric: validation loss (whole val split) at the end of the run (step 300, after the WSD decay).
For each width:
  * grid minimum   = the LR with the lowest mean val loss over the seeds run at that LR;
  * vertex         = minimum of a parabola through the grid minimum and its two neighbours in log2(LR)
                     (only reported when the grid minimum is not at the edge of the grid);
  * uncertainty    = bootstrap: resample seeds at every LR (where >1 seed exists), add the seed-noise, refit.
Width -> LR*: least squares in log space, log2 LR* = a + b log2 width, extrapolated to 4096, with the same bootstrap.
"""
import csv
import math
import random
import statistics as st
from collections import defaultdict

import numpy as np

from common import FIG, RES, load_group, plt, save_json

FINAL = 300
WIDTHS = [256, 512, 1024]
TARGET = 4096


def collect():
    pts = defaultdict(lambda: defaultdict(dict))  # width -> lr -> seed -> record
    meta = {}
    for r in load_group("sweep"):
        c, m = r["config"], r["main"]
        div = m["diverged_at"]
        val = m["evals"][FINAL]["val"] if (div is None and FINAL in m["evals"]) else float("nan")
        pts[c["width"]][c["peak_lr"]][c["seed"]] = {"val": val, "diverged_at": div, "gpu": r["env"]["gpu_name"],
                                                     "train_eval": m["evals"].get(FINAL, {}).get("train_eval"),
                                                     "seconds": r["main"]["seconds"]}
        meta[c["width"]] = {"n_params": r["meta"]["n_params"], "n_head": c["n_head"], "n_layer": c["n_layer"],
                            "head_dim": c["head_dim"], "block_size": c["block_size"]}
        cfg_cmp = {k: v for k, v in c.items() if k not in ("width", "n_head", "peak_lr", "seed", "data_seed")}
        meta.setdefault("_shared", cfg_cmp)
        assert meta["_shared"] == cfg_cmp, "sweep runs differ in something other than width / LR / seed"
    return pts, meta


def vertex(lrs, vals):
    """Parabola through three points in log2(lr); returns (lr*, curvature) or (None, None) if not convex."""
    x = np.log2(lrs)
    a, b, c = np.polyfit(x, vals, 2)
    if a <= 0:
        return None, None
    return float(2 ** (-b / (2 * a))), float(a)


def width_minimum(grid):
    """grid: lr -> list of val losses (nan = failed). Returns dict with grid min, vertex and neighbours."""
    lrs = sorted(grid)
    means = [np.nanmean(grid[l]) if not all(math.isnan(v) for v in grid[l]) else float("nan") for l in lrs]
    finite = [i for i, m in enumerate(means) if not math.isnan(m)]
    i = min(finite, key=lambda j: means[j])
    out = {"grid_min_lr": lrs[i], "grid_min_val": means[i], "at_edge": i in (0, len(lrs) - 1)}
    if not out["at_edge"] and not math.isnan(means[i - 1]) and not math.isnan(means[i + 1]):
        v, a = vertex([lrs[i - 1], lrs[i], lrs[i + 1]], [means[i - 1], means[i], means[i + 1]])
        out.update(vertex_lr=v, curvature_per_log2lr2=a, neighbours=[lrs[i - 1], lrs[i + 1]])
        # a vertex outside the bracketing neighbours means the 3 points are not a well-formed bowl
        if v is not None and not (lrs[i - 1] <= v <= lrs[i + 1]):
            out["vertex_lr"] = None
    return out


def fit_loglog(widths, lrs):
    b, a = np.polyfit(np.log2(widths), np.log2(lrs), 1)
    return float(a), float(b), float(2 ** (a + b * math.log2(TARGET)))


def main():
    pts, meta = collect()
    rows = []
    for w in WIDTHS:
        for lr in sorted(pts[w]):
            for s, rec in sorted(pts[w][lr].items()):
                rows.append({"width": w, "lr": lr, "seed": s, **rec})
    with open(f"{RES}/lr_sweep_runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # seed noise: pooled sd of val loss among LRs with >1 seed that did not fail
    sds = [st.stdev([v["val"] for v in pts[w][lr].values()]) for w in WIDTHS for lr in pts[w]
           if len(pts[w][lr]) > 1 and all(not math.isnan(v["val"]) for v in pts[w][lr].values())]
    seed_sd = float(np.sqrt(np.mean(np.square(sds)))) if sds else None

    mins, curves = {}, {}
    for w in WIDTHS:
        grid = {lr: [v["val"] for v in pts[w][lr].values()] for lr in pts[w]}
        curves[w] = grid
        mins[w] = width_minimum(grid)
        mins[w]["n_runs"] = sum(len(v) for v in grid.values())
        mins[w]["lrs_tried"] = sorted(grid)
        mins[w]["failed_lrs"] = sorted(lr for lr in grid if all(math.isnan(v) for v in grid[lr]))
        lrs = sorted(grid)
        j = lrs.index(mins[w]["grid_min_lr"])
        mins[w]["grid_resolution_bracket"] = [math.sqrt(lrs[j - 1] * lrs[j]) if j > 0 else None,
                                              math.sqrt(lrs[j + 1] * lrs[j]) if j < len(lrs) - 1 else None]

    # bootstrap: resample seeds where available; add N(0, seed_sd) to single-seed points
    rng = random.Random(0)
    boot = defaultdict(list)
    boot_target = []
    for _ in range(2000):
        est = {}
        for w in WIDTHS:
            g = {}
            for lr, vals in curves[w].items():
                good = [v for v in vals if not math.isnan(v)]
                if not good:
                    g[lr] = [float("nan")]
                elif len(good) > 1:
                    g[lr] = [rng.choice(good) for _ in good]
                else:
                    g[lr] = [good[0] + (rng.gauss(0, seed_sd) if seed_sd else 0.0)]
            m = width_minimum(g)
            est[w] = m.get("vertex_lr") or m["grid_min_lr"]
            boot[w].append(est[w])
        boot_target.append(fit_loglog(WIDTHS, [est[w] for w in WIDTHS])[2])
    for w in WIDTHS:
        q = np.percentile(np.log2(boot[w]), [5, 16, 50, 84, 95])
        mins[w]["bootstrap_lr_star_pct_5_16_50_84_95"] = [float(2 ** x) for x in q]

    point = [mins[w].get("vertex_lr") or mins[w]["grid_min_lr"] for w in WIDTHS]
    a, b, lr4096 = fit_loglog(WIDTHS, point)
    a_g, b_g, lr4096_grid = fit_loglog(WIDTHS, [mins[w]["grid_min_lr"] for w in WIDTHS])
    qt = np.percentile(np.log2(boot_target), [5, 16, 50, 84, 95])
    resid = [math.log2(p) - (a + b * math.log2(w)) for w, p in zip(WIDTHS, point)]
    # leave-one-out: does a fit on the two smaller widths predict the third?
    b_loo, a_loo = np.polyfit(np.log2(WIDTHS[:2]), np.log2(point[:2]), 1)
    loo_pred = float(2 ** (a_loo + b_loo * math.log2(WIDTHS[2])))
    out = {
        "metric": f"val loss at step {FINAL} (WSD 300, decayed), whole val split",
        "shared_config": meta["_shared"], "model_per_width": {w: meta[w] for w in WIDTHS},
        "seed_noise_sd_val": seed_sd, "n_seed_sd_estimates": len(sds),
        "minima": mins,
        "fit": {"log2_lr_star = a + b log2 width": {"a": a, "b": b}, "lr_star_used": dict(zip(WIDTHS, point)),
                "residuals_log2": dict(zip(WIDTHS, resid)), "lr_at_4096": lr4096,
                "lr_at_4096_from_grid_minima_only": lr4096_grid, "slope_from_grid_minima_only": b_g,
                "bootstrap_lr_at_4096_pct_5_16_50_84_95": [float(2 ** x) for x in qt],
                "course_inverse_width_rule_from_1024": point[2] * 1024 / TARGET,
                "course_inverse_width_rule_from_256": point[0] * 256 / TARGET,
                "leave_one_out": {"slope_256_512": float(b_loo), "predicted_1024": loo_pred, "measured_1024": point[2],
                                  "ratio_pred_over_measured": loo_pred / point[2]},
                "pairwise_slopes": {"256->512": math.log2(point[1] / point[0]), "512->1024": math.log2(point[2] / point[1])},
                "course_table_value_4096": 1.9e-4},
    }
    save_json("lr_sweep_summary.json", out)
    plot(curves, mins, out)
    for w in WIDTHS:
        m = mins[w]
        print(f"width {w:>5} ({meta[w]['n_params']:,} params): grid min {m['grid_min_lr']:.3g} (val {m['grid_min_val']:.4f}), "
              f"vertex {m.get('vertex_lr')}, edge={m['at_edge']}, failed={m['failed_lrs']}, boot={['%.3g' % x for x in m['bootstrap_lr_star_pct_5_16_50_84_95']]}")
    print("LOO", out["fit"]["leave_one_out"], "pairwise", out["fit"]["pairwise_slopes"])
    print(f"slope b = {b:.3f} (grid-only {b_g:.3f}); LR@4096 = {lr4096:.3g} (grid-only {lr4096_grid:.3g}); "
          f"bootstrap 5/16/50/84/95% {['%.3g' % x for x in out['fit']['bootstrap_lr_at_4096_pct_5_16_50_84_95']]}; seed sd {seed_sd}")


def plot(curves, mins, out):
    col = {256: "#2e86c1", 512: "#8e44ad", 1024: "#c0392b"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.3, 1.1, 1]})
    ax = [axs[0], axs[2]]
    zoom = axs[1]
    for w in WIDTHS:
        lrs = [l for l in sorted(curves[w]) if not all(math.isnan(v) for v in curves[w][l])]
        mu = [np.nanmean(curves[w][l]) for l in lrs]
        sd = [np.nanstd(curves[w][l], ddof=1) if len(curves[w][l]) > 1 else 0 for l in lrs]
        zoom.errorbar(lrs, mu, yerr=sd, fmt="o-", color=col[w], ms=3.5, lw=1.2, capsize=2, label=f"width {w}")
        m = mins[w]
        zoom.plot(m["grid_min_lr"], m["grid_min_val"], "*", color=col[w], ms=14, mec="k", mew=0.6, zorder=5)
        if m.get("vertex_lr"):
            zoom.axvline(m["vertex_lr"], color=col[w], lw=0.8, ls=":")
    zoom.set(xscale="log", ylim=(1.66, 1.86), xlabel="peak learning rate", ylabel=f"val loss at step {FINAL}",
             title="zoom near the minima (mean ± sd over seeds;\ndotted = parabola vertex)")
    for w in WIDTHS:
        lrs = sorted(curves[w])
        good = [(l, np.nanmean(curves[w][l])) for l in lrs if not all(math.isnan(v) for v in curves[w][l])]
        ax[0].plot(*zip(*good), "o-", color=col[w], ms=3.5, lw=1.3, label=f"width {w}")
        for l in lrs:
            vals = [v for v in curves[w][l] if not math.isnan(v)]
            if len(vals) > 1:
                ax[0].plot([l] * len(vals), vals, ".", color=col[w], ms=3, alpha=0.5)
        bad = [l for l in lrs if all(math.isnan(v) for v in curves[w][l])]
        if bad:
            ax[0].plot(bad, [2.75] * len(bad), "x", color=col[w], ms=6)
        m = mins[w]
        ax[0].plot(m["grid_min_lr"], m["grid_min_val"], "*", color=col[w], ms=14, mec="k", mew=0.6, zorder=5)
        ax[0].annotate(f"{m['grid_min_lr']:.3g}", (m["grid_min_lr"], m["grid_min_val"]), xytext=(0, -14),
                       textcoords="offset points", ha="center", fontsize=8, color=col[w])
    ax[0].set(xscale="log", xlabel="peak learning rate", ylabel=f"val loss at step {FINAL}",
              title="LR sweep (★ = measured minimum; dots = extra seeds; × = failed run)", ylim=(None, 2.8))
    ax[0].legend(frameon=False, fontsize=8)
    f = out["fit"]
    pts = [f["lr_star_used"][w] for w in WIDTHS]
    for w, p in zip(WIDTHS, pts):
        lo, _, _, _, hi = mins[w]["bootstrap_lr_star_pct_5_16_50_84_95"]
        ax[1].errorbar(w, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col[w], capsize=3, ms=6)
    a, b = f["log2_lr_star = a + b log2 width"]["a"], f["log2_lr_star = a + b log2 width"]["b"]
    xs = np.array([200, TARGET * 1.2])
    ax[1].plot(xs, 2 ** (a + b * np.log2(xs)), "-", color="k", lw=1, label=f"fit: LR* ∝ width^{b:.2f}")
    ax[1].plot(xs, pts[0] * 256 / xs, ":", color="grey", lw=1, label="course rule LR* ∝ 1/width (through 256)")
    lo, _, _, _, hi = f["bootstrap_lr_at_4096_pct_5_16_50_84_95"]
    ax[1].errorbar(TARGET, f["lr_at_4096"], yerr=[[f["lr_at_4096"] - lo], [hi - f["lr_at_4096"]]], fmt="D", color="k",
                   mfc="white", capsize=3, label=f"4096 extrapolation {f['lr_at_4096']:.2g} (90% boot.)")
    ax[1].set(xscale="log", yscale="log", xlabel="width (d_model)", ylabel="best LR", title="best LR vs width")
    import matplotlib.ticker as mt
    ax[1].xaxis.set_minor_formatter(mt.NullFormatter())
    ax[1].set_xticks([256, 512, 1024, 2048, 4096])
    ax[1].set_xticklabels(["256", "512", "1024", "2048", "4096"])
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/lr_sweep_widths.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
