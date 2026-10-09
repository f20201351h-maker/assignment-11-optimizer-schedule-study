"""
Experiment 3: per-tensor update-to-weight ratio  r_t = ||W_t - W_{t-1}||_F / ||W_{t-1}||_F  through warm-up.

Runs (all the base model, WSD planned for 300 steps, the WSD-tuned peak LR, seed 1337, same batches):
  main      : warm-up 30 (the schedule-comparison run itself)
  controls  : warm-up 0, warm-up 60, and the same three with bias correction switched off (warm-up 30 / 0)

"When does warm-up stop changing the ratio?" is answered two ways, both fixed before the control runs existed:
  A  within the main run: s_t = 9-step centred rolling median of r_t; reference R = median r over steps 100-180
     (flat LR, before any decay). t_A = first step after which |s_t / R - 1| <= 20% holds through step 180.
  B  against the no-warm-up control: t_B = first step after which s_t(warm-up 30) and s_t(warm-up 0) stay within
     a factor 1.2 of each other through step 180, i.e. the step after which having had warm-up no longer shows.
Sensitivity: 10% / 30% bands. Reported per tensor, then as the median over tensors and per family.

Added after seeing the controls (post hoc, labelled as such everywhere): the warm-up-0 run never left the loss
plateau, so B mostly never settles. B2 therefore compares warm-up 30 with warm-up 60, two runs that both train
normally: the step after which the two ratios stay within a factor 1.2 through step 180.
"""
import csv
import math
import statistics as st

import numpy as np

from common import FIG, RES, load_group, plt, save_json

REF = (100, 180)
END = 180
BAND = 0.2
BANDS = [0.1, 0.2, 0.3]
SMOOTH = 9


def smooth(x, k=SMOOTH):
    h = k // 2
    return np.array([np.median(x[max(0, i - h): i + h + 1]) for i in range(len(x))])


def first_settled(ok):
    """ok[i] for step i+1; first step from which ok holds through the end of the array."""
    t = None
    for i in range(len(ok) - 1, -1, -1):
        if ok[i]:
            t = i + 1
        else:
            break
    return t


def ratio_matrix(run, key="ratio"):
    return np.array([rec[key] for rec in run["main"]["log"]])  # [steps, tensors]


def find(runs, warmup, bc):
    m = [r for r in runs if r["config"]["warmup"] == warmup and r["config"]["bias_correction"] == bc]
    assert len(m) == 1, (warmup, bc, len(m))
    return m[0]


def main(lr):
    sched = [r for r in load_group("sched") if r["config"]["schedule"] == "wsd" and r["config"]["peak_lr"] == lr
             and r["config"]["seed"] == 1337]
    assert len(sched) == 1
    runs = sched + load_group("warmup")
    runs = [r for r in runs if r["config"]["peak_lr"] == lr and r["config"]["seed"] == 1337]
    main_run = find(runs, 30, True)
    ctrl = {(wu, bc): find(runs, wu, bc) for wu, bc in [(0, True), (60, True), (30, False), (0, False)]}
    for r in ctrl.values():
        assert r["meta"]["init_hash"] == main_run["meta"]["init_hash"], "controls must start from the same weights"
        n = min(len(r["main"]["log"]), len(main_run["main"]["log"]))
        assert [x["data_hash"] for x in r["main"]["log"][:n]] == [x["data_hash"] for x in main_run["main"]["log"][:n]]
    tensors = main_run["meta"]["tensors"]
    labels = [t["label"] for t in tensors]
    fams = [t["family"] for t in tensors]
    R = ratio_matrix(main_run)
    R0 = ratio_matrix(ctrl[(0, True)])
    R60 = ratio_matrix(ctrl[(60, True)])
    assert np.isfinite(R).all() and np.isfinite(R0).all()
    lr_curve = np.array([x["lr"] for x in main_run["main"]["log"]])
    A = ratio_matrix(main_run, "adam_rms_over_lr")
    A0 = ratio_matrix(ctrl[(0, True)], "adam_rms_over_lr")

    per_tensor = []
    for j, lab in enumerate(labels):
        s, s0, s60 = smooth(R[:, j]), smooth(R0[:, j]), smooth(R60[:, j])
        ref = float(np.median(R[REF[0] - 1:REF[1], j]))
        row = {"tensor": lab, "family": fams[j], "numel": tensors[j]["numel"], "ref_ratio_100_180": ref,
               "ratio_step1": float(R[0, j]), "ratio_step30": float(R[29, j]), "ratio_step200": float(R[199, j]),
               "max_ratio_wu30": float(R[:END, j].max()), "argmax_wu30": int(R[:END, j].argmax()) + 1,
               "max_ratio_wu0": float(R0[:END, j].max()), "argmax_wu0": int(R0[:END, j].argmax()) + 1}
        for band in BANDS:
            row[f"tA_band{band}"] = first_settled(np.abs(s[:END] / ref - 1) <= band)
            row[f"tB_band{band}"] = first_settled(np.abs(np.log(s[:END] / s0[:END])) <= math.log(1 + band))
            row[f"tB60_band{band}"] = first_settled(np.abs(np.log(s60[:END] / s0[:END])) <= math.log(1 + band))
            row[f"tB2_band{band}"] = first_settled(np.abs(np.log(s[:END] / s60[:END])) <= math.log(1 + band))
        per_tensor.append(row)
    with open(f"{RES}/update_ratio_per_tensor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tensor[0].keys()))
        w.writeheader()
        w.writerows(per_tensor)
    # full per-step log for the main run, every tensor
    with open(f"{RES}/update_ratio_main_run_every_step.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "loss"] + labels)
        for i, rec in enumerate(main_run["main"]["log"]):
            w.writerow([rec["step"], rec["lr"], rec["loss"]] + [f"{v:.6g}" for v in R[i]])

    def med(key, rows=per_tensor):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return {"median": None, "min": None, "max": None, "n": 0, "n_never": len(rows)}
        return {"median": float(np.median(v)), "min": int(min(v)), "max": int(max(v)), "n": len(v), "n_never": len(rows) - len(v)}

    fam_names = list(dict.fromkeys(fams))
    summary = {
        "lr": lr, "warmup_end_configured": 30, "criterion": __doc__.strip().split("\n\n")[2],
        "tA": {str(b): med(f"tA_band{b}") for b in BANDS}, "tB": {str(b): med(f"tB_band{b}") for b in BANDS},
        "tB_warmup60_vs_0": {str(b): med(f"tB60_band{b}") for b in BANDS},
        "tB2_posthoc_warmup30_vs_60": {str(b): med(f"tB2_band{b}") for b in BANDS},
        "per_family": {fam: {"tA": med(f"tA_band{BAND}", [r for r in per_tensor if r["family"] == fam]),
                             "tB": med(f"tB_band{BAND}", [r for r in per_tensor if r["family"] == fam]),
                             "tB2_posthoc": med(f"tB2_band{BAND}", [r for r in per_tensor if r["family"] == fam]),
                             "ref_ratio_median": float(np.median([r["ref_ratio_100_180"] for r in per_tensor if r["family"] == fam])),
                             "max_ratio_wu30_median": float(np.median([r["max_ratio_wu30"] for r in per_tensor if r["family"] == fam])),
                             "max_ratio_wu0_median": float(np.median([r["max_ratio_wu0"] for r in per_tensor if r["family"] == fam]))}
                       for fam in fam_names},
        "largest_ratio_any_tensor_first180": {"warmup30": float(R[:END].max()), "warmup0": float(R0[:END].max()),
                                              "warmup60": float(R60[:END].max())},
        "median_over_tensors_ratio_by_step": {s: float(np.median(R[s - 1])) for s in (1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300)},
        "median_over_tensors_ratio_by_step_warmup0": {s: float(np.median(R0[s - 1])) for s in (1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200)},
        "median_over_matrices_ratio_by_step": {s: float(np.median(R[s - 1, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]]))
                                               for s in (1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100, 120, 150, 180, 200, 250, 300)},
        "median_over_matrices_ratio_by_step_warmup0": {s: float(np.median(R0[s - 1, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]]))
                                                       for s in (1, 5, 10, 20, 30, 50, 100, 150, 200)},
        "median_over_matrices_ratio_by_step_warmup60": {s: float(np.median(R60[s - 1, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]]))
                                                        for s in (1, 10, 20, 30, 40, 50, 60, 75, 100, 150, 200)},
        "extremes_median_over_matrices": {
            "peak_step_1_60": int(np.median(R[:60, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]], axis=1).argmax()) + 1,
            "peak_value": float(np.median(R[:60, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]], axis=1).max()),
            "trough_step_30_100": int(np.median(R[29:100, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]], axis=1).argmin()) + 30,
            "trough_value": float(np.median(R[29:100, [j for j, f in enumerate(fams) if f != "LayerNorm gain"]], axis=1).min()),
            "adam_rms_over_lr_min_step_1_200": int(np.median(A[:200], axis=1).argmin()) + 1,
            "adam_rms_over_lr_min_value": float(np.median(A[:200], axis=1).min())},
        "adam_rms_over_lr_median_by_step": {s: float(np.median(A[s - 1])) for s in (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 100, 150, 200)},
        "adam_rms_over_lr_median_by_step_warmup0": {s: float(np.median(A0[s - 1])) for s in (1, 2, 5, 10, 20, 30, 50, 100, 200)},
        "controls_val": {f"wu{wu}_bc{int(bc)}": {"val200": r["main"]["evals"].get(200, {}).get("val"),
                                                 "val300": r["main"]["evals"].get(300, {}).get("val"),
                                                 "diverged_at": r["main"]["diverged_at"],
                                                 "max_grad_norm_first50": max(x["grad_norm"] for x in r["main"]["log"][:50])}
                         for (wu, bc), r in {**ctrl, (30, True): main_run}.items()},
    }
    # every seed of every warm-up / bias-correction setting (the warm-up-30 + BC runs are the schedule-comparison runs)
    allruns = [r for r in load_group("warmup") if r["config"]["peak_lr"] == lr] +               [r for r in load_group("sched") if r["config"]["schedule"] == "wsd" and r["config"]["peak_lr"] == lr]
    seeds_table = {}
    for r in allruns:
        k = f"wu{r['config']['warmup']}_bc{int(r['config']['bias_correction'])}"
        ev = r["main"]["evals"]
        seeds_table.setdefault(k, {})[r["config"]["seed"]] = {
            "val200": ev[200]["val"], "val300": ev[300]["val"],
            "max_grad_norm_first50": max(x["grad_norm"] for x in r["main"]["log"][:50]),
            "max_ratio_any_tensor_first50": max(max(x["ratio"]) for x in r["main"]["log"][:50])}
    summary["controls_all_seeds"] = {k: {"per_seed": v, "val300_mean": float(np.mean([x["val300"] for x in v.values()])),
                                         "val300_min": float(min(x["val300"] for x in v.values())),
                                         "val300_max": float(max(x["val300"] for x in v.values()))}
                                     for k, v in sorted(seeds_table.items())}
    save_json("update_ratio_summary.json", summary)
    plot(R, R0, R60, A, A0, lr_curve, labels, fams, per_tensor, summary, ctrl, main_run)
    print(json.dumps({k: summary[k] for k in ("tA", "tB", "tB_warmup60_vs_0", "tB2_posthoc_warmup30_vs_60", "largest_ratio_any_tensor_first180")}, indent=1))
    for fam, v in summary["per_family"].items():
        print(f"{fam:>32}: tA {v['tA']}  tB2 {v['tB2_posthoc']} tB never {v['tB']['n_never']}  ref {v['ref_ratio_median']:.2e}  max wu30 {v['max_ratio_wu30_median']:.2e} wu0 {v['max_ratio_wu0_median']:.2e}")
    print("median ratio by step", {k: f"{v:.2e}" for k, v in summary["median_over_tensors_ratio_by_step"].items()})
    print("adam rms/lr", {k: round(v, 3) for k, v in summary["adam_rms_over_lr_median_by_step"].items()})
    for k, v in summary["controls_all_seeds"].items():
        print(k, round(v["val300_mean"], 4), [round(x["val300"], 4) for x in v["per_seed"].values()],
              "max gn", [round(x["max_grad_norm_first50"], 1) for x in v["per_seed"].values()],
              "max ratio", [round(x["max_ratio_any_tensor_first50"], 3) for x in v["per_seed"].values()])


def plot(R, R0, R60, A, A0, lr_curve, labels, fams, per_tensor, summary, ctrl, main_run):
    # ---- heatmap: every tensor x every step ----
    order = sorted(range(len(labels)), key=lambda j: (["embedding (tied wte = lm_head)", "position embedding", "attn qkv",
                                                        "attn out proj", "mlp up", "mlp down", "LayerNorm gain"].index(fams[j]), j))
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(np.log10(R[:, order].T), aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=[0.5, R.shape[0] + 0.5, len(order) - 0.5, -0.5], vmin=-4.5, vmax=-1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[j] for j in order], fontsize=6.5)
    for x, txt in ((30, "warm-up ends (30)"), (270, "WSD decay starts (271)"), (200, "step 200")):
        ax.axvline(x + 0.5, color="w", lw=1, ls="--")
        ax.text(x + 2, len(order) - 1.2, txt, fontsize=7, va="bottom", color="w")
    tA = summary["tA"][str(BAND)]["median"]
    ax.axvline(tA, color="#ff6b6b", lw=1.2)
    ax.text(tA + 2, len(order) - 3.2, f"median t_A = {tA:.0f}", color="#ff6b6b", fontsize=7, va="bottom")
    ax.set(xlabel="optimizer step", title="log10 update-to-weight ratio ||ΔW|| / ||W||, every tensor, every step "
                                          "(WSD, warm-up 30, tuned LR)")
    fig.colorbar(im, ax=ax, label="log10 ratio", fraction=0.025)
    fig.tight_layout()
    fig.savefig(f"{FIG}/update_ratio_heatmap.png")
    plt.close(fig)

    # ---- representative tensors ----
    reps = ["wte=lm_head", "wpe", "h.0.attn.c_attn", "h.3.mlp.c_fc", "h.5.mlp.c_proj", "h.2.ln_1"]
    cols = ["#1f5aa6", "#16a085", "#c0392b", "#8e44ad", "#e67e22", "#7f8c8d"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = np.arange(1, R.shape[0] + 1)
    for lab, c in zip(reps, cols):
        j = labels.index(lab)
        ax[0].plot(steps[:200], smooth(R[:, j])[:200], color=c, lw=1.4, label=lab)
        ax[0].plot(steps[:200], smooth(R0[:, j])[:200], color=c, lw=0.9, ls=":")
    ax[0].axvline(30, color="k", lw=0.8, ls="--")
    ax[0].axvspan(REF[0], REF[1], color="grey", alpha=0.12, lw=0)
    ax[0].axhline(1e-3, color="grey", lw=0.6)
    ax[0].text(195, 1.1e-3, "course's 1e-3 reference", ha="right", fontsize=7, color="grey")
    ax[0].set(yscale="log", xlabel="step", ylabel="||ΔW|| / ||W|| (9-step median)",
              title="representative tensors: solid = warm-up 30, dotted = no warm-up")
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    axb = ax[0].twinx()
    axb.plot(steps[:200], lr_curve[:200] * 1e3, color="k", lw=0.8, alpha=0.4)
    axb.set_ylabel("LR ×1e-3 (grey)", fontsize=8)
    axb.spines["top"].set_visible(False)
    # ratio divided by LR: how much of the ratio is the schedule and how much is Adam / the weights
    mats = [j for j, f in enumerate(fams) if f not in ("LayerNorm gain",)]
    ax[1].plot(steps[:200], np.median(A[:200], axis=1), color="#c0392b", lw=1.4, label="warm-up 30")
    ax[1].plot(steps[:200], np.median(A0[:200], axis=1), color="#1f5aa6", lw=1.2, ls=":", label="no warm-up")
    ax[1].axvline(30, color="k", lw=0.8, ls="--")
    ax[1].axhline(1, color="grey", lw=0.6)
    ax[1].set(xlabel="step", ylabel="RMS(Adam step) / LR", ylim=(0, 1.05),
              title="Adam's step in units of LR (median over all 39 tensors)")
    ax[1].legend(fontsize=7, frameon=False)
    # median ratio over all tensors, three warm-up settings
    for M, lab, c, ls in ((R0, "warm-up 0", "#1f5aa6", ":"), (R, "warm-up 30", "#c0392b", "-"), (R60, "warm-up 60", "#27ae60", "--")):
        ax[2].plot(steps[:200], smooth(np.median(M[:200, mats], axis=1)), color=c, ls=ls, lw=1.4, label=lab)
    for x in (30, 60):
        ax[2].axvline(x, color="k", lw=0.6, ls="--")
    tB2 = summary["tB2_posthoc_warmup30_vs_60"][str(BAND)]["median"]
    ax[2].axvline(tB2, color="#27ae60", lw=1)
    ax[2].text(tB2 - 3, 4.4e-3,  "warm-up 30 vs 60 agree\nfrom ~%.0f (post hoc)" % tB2, fontsize=7, color="#27ae60", ha="right")
    ax[2].set(yscale="log", xlabel="step", ylabel="median ratio over weight matrices",
              title="does warm-up still show? (median over matrices)")
    ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/update_ratio_warmup.png")
    plt.close(fig)


if __name__ == "__main__":
    import json
    import sys
    main(float(sys.argv[1]))
