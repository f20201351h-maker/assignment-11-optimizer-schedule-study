"""
Experiment 4: cosine vs WSD, both planned for 300 steps, inspected at step 200.

Tuning protocol (identical for both schedules):
  1. seed 1337 over the same nine peak LRs (sqrt(2) grid 1e-3..8e-3 plus 2.38e-3 and 3.36e-3 around the optimum);
  2. seeds 2024 and 7 at the three LRs around the optimum (2e-3, 2.38e-3, 2.83e-3), for both schedules;
  3. each schedule keeps the LR with the lowest 3-seed mean validation loss at step 200.
Warm-up (30), min LR (0.1 x peak), betas, weight decay, clipping, batch and data are shared, not tuned.
"""
import csv
import statistics as st
from collections import defaultdict

from common import FIG, RES, load_group, lr_at, plt, save_json, train_window_mean

STOP = 200


def collect():
    rows = []
    for r in load_group("sched"):
        c = r["config"]
        m = r["main"]
        kind = f"{c['schedule']}{c['total_steps']}"
        base = dict(lr=c["peak_lr"], seed=c["seed"], init_hash=r["meta"]["init_hash"], gpu=r["env"]["gpu_name"])
        ev = m["evals"]
        rows.append(dict(base, kind=kind, val200=ev[STOP]["val"], train_eval200=ev[STOP]["train_eval"],
                         train_last10_200=train_window_mean(m["log"], STOP), lr_at_200=lr_at(m["log"], STOP),
                         val300=ev.get(300, {}).get("val"), train_eval300=ev.get(300, {}).get("train_eval"),
                         data_hash_200=next(x["data_hash"] for x in m["log"] if x["step"] == STOP),
                         curve=sorted((s, e["val"]) for s, e in ev.items()), lr_curve=[x["lr"] for x in m["log"]]))
        for bname, b in r["branches"].items():
            bev = b["evals"]
            full_log = [x for x in m["log"] if x["step"] <= b["log"][0]["step"] - 1] + b["log"]
            rows.append(dict(base, kind="wsd_branch180", val200=bev[STOP]["val"], train_eval200=bev[STOP]["train_eval"],
                             train_last10_200=train_window_mean(full_log, STOP), lr_at_200=lr_at(b["log"], STOP),
                             val300=None, train_eval300=None, data_hash_200=b["log"][-1]["data_hash"],
                             curve=sorted([(s, e["val"]) for s, e in ev.items() if s <= 180] + [(s, e["val"]) for s, e in bev.items()]),
                             lr_curve=[x["lr"] for x in full_log]))
    return rows


def main():
    rows = collect()
    kinds = ["cosine300", "wsd300", "wsd_branch180", "cosine200"]
    # ---------------- integrity checks ----------------
    by_seed = defaultdict(set)
    for r in rows:
        by_seed[r["seed"]].add(r["init_hash"])
    assert all(len(v) == 1 for v in by_seed.values()), "runs with the same seed must share an init"
    for seed in by_seed:
        h = {r["data_hash_200"] for r in rows if r["seed"] == seed}
        assert len(h) == 1, f"seed {seed}: not every run saw the same first 200 batches"
    import math
    assert all(math.isfinite(r["val200"]) for r in rows)
    effort = {k: len([r for r in rows if r["kind"] == k]) for k in kinds}
    assert effort["cosine300"] == effort["wsd300"], effort

    # ---------------- tuning table ----------------
    table = []
    for k in kinds:
        for lr in sorted({r["lr"] for r in rows if r["kind"] == k}):
            rs = [r for r in rows if r["kind"] == k and r["lr"] == lr]
            v = [r["val200"] for r in rs]
            table.append(dict(kind=k, lr=lr, n_seeds=len(rs), val200_mean=st.mean(v), val200_sd=st.stdev(v) if len(v) > 1 else None,
                              val200_seed1337=next(r["val200"] for r in rs if r["seed"] == 1337),
                              train_eval200_mean=st.mean(r["train_eval200"] for r in rs),
                              train_last10_200_mean=st.mean(r["train_last10_200"] for r in rs),
                              lr_at_200=rs[0]["lr_at_200"],
                              val300_mean=st.mean(r["val300"] for r in rs) if rs[0]["val300"] is not None else None))
    with open(f"{RES}/schedule_tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    selected = {}
    for k in kinds:
        cands = [t for t in table if t["kind"] == k and t["n_seeds"] == 3]
        best = min(cands, key=lambda t: t["val200_mean"])
        selected[k] = best
    # the edge check: the selected LR must not be the largest or smallest LR tried
    for k, t in selected.items():
        lrs = sorted({x["lr"] for x in table if x["kind"] == k})
        t["at_grid_edge"] = t["lr"] in (lrs[0], lrs[-1])
    # also: what seed-1337-only tuning would have picked, and what selecting on step 300 would have picked
    alt = {k: min((t for t in table if t["kind"] == k), key=lambda t: t["val200_seed1337"])["lr"] for k in kinds}
    alt300 = {k: min((t for t in table if t["kind"] == k and t["n_seeds"] == 3), key=lambda t: t["val300_mean"])["lr"]
              for k in ("cosine300", "wsd300")}

    # ---------------- paired comparison at the selected LRs ----------------
    def per_seed(kind, lr):
        return {r["seed"]: r for r in rows if r["kind"] == kind and r["lr"] == lr}
    cos, wsd = per_seed("cosine300", selected["cosine300"]["lr"]), per_seed("wsd300", selected["wsd300"]["lr"])
    br, c200 = per_seed("wsd_branch180", selected["wsd_branch180"]["lr"]), per_seed("cosine200", selected["cosine200"]["lr"])
    seeds = sorted(cos)
    paired = {
        "wsd300_minus_cosine300_val200": [wsd[s]["val200"] - cos[s]["val200"] for s in seeds],
        "wsd_branch_minus_cosine300_val200": [br[s]["val200"] - cos[s]["val200"] for s in seeds],
        "cosine200_minus_cosine300_val200": [c200[s]["val200"] - cos[s]["val200"] for s in seeds],
        "wsd300_minus_cosine300_val300": [wsd[s]["val300"] - cos[s]["val300"] for s in seeds],
    }
    paired_summary = {k: {"per_seed": dict(zip(seeds, v)), "mean": st.mean(v), "sd": st.stdev(v),
                          "all_same_sign": all(x > 0 for x in v) or all(x < 0 for x in v)} for k, v in paired.items()}
    seed_sd = {k: selected[k]["val200_sd"] for k in kinds}
    out = {"stop_step": STOP, "planned_steps": 300, "runs_per_schedule": effort, "selected": selected,
           "selection_if_seed1337_only": alt, "selection_if_tuned_on_step300": alt300,
           "paired_at_selected_lrs": paired_summary, "seed_sd_val200_at_selected": seed_sd,
           "per_seed_at_selected": {k: {s: {kk: v[s][kk] for kk in ("val200", "train_eval200", "train_last10_200", "lr_at_200", "val300")}
                                        for s in seeds} for k, v in (("cosine300", cos), ("wsd300", wsd), ("wsd_branch180", br), ("cosine200", c200))}}
    out["literal_stop200_reruns"] = final_runs(selected, rows)
    save_json("schedules_summary.json", out)
    plot(rows, table, selected)
    for k in kinds:
        t = selected[k]
        print(f"{k:>14}: lr {t['lr']:.3g}  val@200 {t['val200_mean']:.4f} ± {t['val200_sd']:.4f}  train_eval@200 {t['train_eval200_mean']:.4f}  "
              f"train(last10)@200 {t['train_last10_200_mean']:.4f}  lr@200 {t['lr_at_200']:.3g}  val@300 {t['val300_mean']}  edge={t['at_grid_edge']}")
    for k, v in paired_summary.items():
        print(f"{k}: mean {v['mean']:+.4f} sd {v['sd']:.4f} per seed {['%+.4f' % x for x in v['per_seed'].values()]}")
    for k, v in out["literal_stop200_reruns"].items():
        print("final", k, {kk: vv for kk, vv in v.items() if kk != "checkpoint"})
    print("seed-1337-only picks", alt, "| step-300 picks", alt300, "| effort", effort)


def final_runs(selected, rows):
    """The two tuned configs re-run with stop_step=200 (checkpoints saved, reloaded and re-evaluated on GPU),
    plus the WSD step-200 checkpoint finished with a 22-step decay."""
    import json
    import os
    out = {}
    ver = json.load(open(os.path.join(RES, "checkpoint_verification.json")))
    for r in load_group("final"):
        c, m = r["config"], r["main"]
        k = f"{c['schedule']}{c['total_steps']}"
        assert m["log"][-1]["step"] == STOP == c["stop_step"], "the final runs must stop at 200"
        tuning = next(x for x in rows if x["kind"] == k and x["lr"] == c["peak_lr"] and x["seed"] == c["seed"])
        assert r["meta"]["init_hash"] == tuning["init_hash"]
        assert m["log"][-1]["data_hash"] == tuning["data_hash_200"], "rerun must see the same 200 batches"
        v = next(x for x in ver if x["run_id"] == r["run_id"] and x["segment"] == "main")
        assert v["sha_match"] and v["optimizer_steps"] == [STOP] and v["abs_val_diff"] < 1e-4
        out[k] = {"run_id": r["run_id"], "peak_lr": c["peak_lr"], "val200": m["evals"][STOP]["val"],
                  "train_eval200": m["evals"][STOP]["train_eval"], "train_last10_200": train_window_mean(m["log"], STOP),
                  "lr_at_200": lr_at(m["log"], STOP), "gpu": r["env"]["gpu_name"],
                  "tuning_run_val200_same_seed": tuning["val200"], "tuning_run_gpu": tuning["gpu"],
                  "rerun_minus_tuning_val200": m["evals"][STOP]["val"] - tuning["val200"],
                  "checkpoint": {"path_on_volume": m["ckpts"][str(STOP)]["path"] if str(STOP) in m["ckpts"] else m["ckpts"][STOP]["path"],
                                 "weights_sha": v["weights_sha_logged"], "reloaded_val": v["reloaded_val"]}}
        for bname, b in r["branches"].items():
            vb = next(x for x in ver if x["run_id"] == r["run_id"] and x["segment"] == bname)
            out[bname] = {"from_step": 200, "end_step": b["log"][-1]["step"], "val_end": b["evals"][b["log"][-1]["step"]]["val"],
                          "val210": b["evals"].get(210, {}).get("val"), "train_eval_end": b["evals"][b["log"][-1]["step"]]["train_eval"],
                          "lr_at_end": b["log"][-1]["lr"], "reloaded_val": vb["reloaded_val"], "sha_match": vb["sha_match"]}
    return out


def plot(rows, table, selected):
    col = {"cosine300": "#1f5aa6", "wsd300": "#c0392b", "wsd_branch180": "#e67e22", "cosine200": "#7f8c8d"}
    lab = {"cosine300": "cosine, planned 300", "wsd300": "WSD, planned 300", "wsd_branch180": "WSD branch: decay 180→200",
           "cosine200": "cosine, planned 200 (rerun)"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
    for k in col:
        r = next(x for x in rows if x["kind"] == k and x["lr"] == selected[k]["lr"] and x["seed"] == 1337)
        ax[0].plot(range(1, len(r["lr_curve"]) + 1), [v * 1e3 for v in r["lr_curve"]], color=col[k], lw=1.6,
                   ls="--" if k in ("wsd_branch180", "cosine200") else "-", label=f"{lab[k]} (peak {selected[k]['lr']:.3g})")
        s, v = zip(*[(a, b) for a, b in r["curve"] if a >= 120])
        ax[1].plot(s, v, color=col[k], lw=1.4, ls="--" if k in ("wsd_branch180", "cosine200") else "-", label=lab[k])
    for a in ax[:2]:
        a.axvline(STOP, color="k", lw=0.8, ls=":")
    ax[0].set(title="learning rate (each at its tuned peak)", xlabel="step", ylabel="LR (×1e-3)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set(title="validation loss, seed 1337 (full val split)", xlabel="step", ylabel="val loss", ylim=(1.68, 2.25))
    ax[1].legend(fontsize=7, frameon=False)
    for k in col:
        t = sorted([x for x in table if x["kind"] == k], key=lambda x: x["lr"])
        ax[2].plot([x["lr"] for x in t], [x["val200_seed1337"] for x in t], "o-" if k in ("cosine300", "wsd300") else "o--",
                   color=col[k], ms=3, lw=1, label=lab[k])
        t3 = [x for x in t if x["n_seeds"] == 3]
        ax[2].errorbar([x["lr"] for x in t3], [x["val200_mean"] for x in t3], yerr=[x["val200_sd"] for x in t3], fmt="none",
                       ecolor=col[k], capsize=2, lw=1)
        ax[2].plot(selected[k]["lr"], selected[k]["val200_mean"], "*", color=col[k], ms=11, mec="k", mew=0.5)
    ax[2].set(xscale="log", title="tuning: val loss at step 200 vs peak LR\n(line: seed 1337; bars: 3-seed mean ± sd; ★ chosen)",
              xlabel="peak LR", ylabel="val loss @ 200", ylim=(1.85, 2.6))
    ax[2].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/schedules_step200.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
