"""
Pass-1 technical evidence audit. Recomputes the headline numbers from raw artifacts with code that does NOT import
the analysis scripts (formulas are re-implemented here), and checks run-file integrity.

    python audit/evidence_audit.py   -> results/evidence_audit.json (raises on any failure)
"""
import csv
import glob
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("[ok]  " if ok else "[FAIL]") + f" {name}  {detail}")


def load(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. Adam by hand
rows = list(csv.DictReader(open(R("adam_by_hand.csv"))))
g = [float(r["g"]) for r in rows]
b1, b2, lr, eps, w = 0.9, 0.999, 1e-3, 1e-8, 1.0
worst = 0.0
for t in range(1, 6):
    m = sum((1 - b1) * b1 ** (t - i) * g[i - 1] for i in range(1, t + 1))
    v = sum((1 - b2) * b2 ** (t - i) * g[i - 1] ** 2 for i in range(1, t + 1))
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    w -= lr * mh / (math.sqrt(vh) + eps)
    r = rows[t - 1]
    for k, val in (("m", m), ("v", v), ("m_hat", mh), ("v_hat", vh), ("w", w)):
        worst = max(worst, abs(float(r[k]) - val))
check("Adam table recomputed from the summation form", worst < 1e-12, f"max abs diff {worst:.1e}")
aj = load(R("adam_by_hand.json"))
e64 = max(aj["max_abs_err_by_hand_vs"]["torch_float64"].values())
e32 = max(aj["max_abs_err_by_hand_vs"]["torch_float32"].values())
check("by hand vs torch float64 / float32 within stated tolerance", e64 < 1e-13 and e32 < 5e-7, f"{e64:.1e} / {e32:.1e}")
check("course table reproduced at printed precision", aj["course_table_matches_at_printed_precision"])

# ---------------------------------------------------------------- 2. bias correction t*
def tstar(beta2, tol=0.05, window=10, n=20000):
    ok = [abs((1 - 0.9 ** t) / math.sqrt(1 - beta2 ** t) - 1) < tol for t in range(1, n + 1)]
    for i in range(n - window):
        if all(ok[i:i + window]):
            return i + 1
bc = load(R("bias_correction.json"))
for case in bc["cases"]:
    exp = tstar(case["beta2"])
    check(f"t* {case['sequence']} beta2={case['beta2']} equals the eps-free closed form", case["t_star"] == exp, f"{case['t_star']} vs {exp}")
s1 = [c["step1_ratio_noBC_over_BC"] for c in bc["cases"] if c["beta2"] == 0.999]
check("step-1 ratio without BC is 0.1/sqrt(0.001) for beta2=0.999 (eps=1e-8 shifts it ~1e-6 relative)",
      all(abs(x / (0.1 / math.sqrt(0.001)) - 1) < 1e-5 for x in s1), f"{s1} vs {0.1 / math.sqrt(0.001):.7f}")

# ---------------------------------------------------------------- 3. every logged LR against an independent schedule formula
def lr_ref(s, sched, peak, wu, total, frac=0.1, mr=0.1):
    lo = peak * mr
    if wu and s <= wu:
        return peak * s / wu
    if sched == "cosine":
        return lo if s >= total else lo + (peak - lo) * (1 + math.cos(math.pi * (s - wu) / (total - wu))) / 2
    start = total - int(round(frac * total))
    if s <= start:
        return peak
    return lo if s >= total else peak - (peak - lo) * (s - start) / (total - start)

runs = {p: load(p) for p in glob.glob(R("runs", "*", "*.json"))}
worst, n = 0.0, 0
for p, r in runs.items():
    c = r["config"]
    for x in r["main"]["log"]:
        worst = max(worst, abs(x["lr"] - lr_ref(x["step"], c["schedule"], c["peak_lr"], c["warmup"], c["total_steps"], c["decay_frac"], c["min_ratio"])) / c["peak_lr"])
        n += 1
    for bc_ in r.get("branches_config", []):
        for x in r["branches"][bc_["name"]]["log"]:
            worst = max(worst, abs(x["lr"] - lr_ref(x["step"], bc_["schedule"], c["peak_lr"], c["warmup"], bc_["total_steps"], bc_["decay_frac"], c["min_ratio"])) / c["peak_lr"])
            n += 1
check("every logged LR matches the schedule formula", worst < 1e-5, f"{n} steps, max rel err {worst:.1e} (values stored to 6 s.f.)")

# ---------------------------------------------------------------- 4. update-to-weight ratio: analytic step-1 value
# At step 1 Adam moves every coordinate by lr*g/(|g|+eps) ~ lr, so ||dW||/||W|| ~ lr_1 / rms(W_0): 0.02 for N(0,0.02)
# tensors, 0.02/sqrt(2*6) for residual projections, 1 for LayerNorm gains. A ratio pipeline bug would break this.
main = next(r for p, r in runs.items() if "sched__w384__wsd300__lr0.002__wu30__bc1__s1337" in p)
lr1 = main["main"]["log"][0]["lr"]
worst = 0.0
for t, ratio in zip(main["meta"]["tensors"], main["main"]["log"][0]["ratio"]):
    rms0 = 1.0 if "ln" in t["label"] else (0.02 / math.sqrt(12) if t["label"].endswith("c_proj") else 0.02)
    worst = max(worst, abs(ratio / (lr1 / rms0) - 1))
check("step-1 ratio = lr_1 / init RMS for all 39 tensors", worst < 0.05, f"max rel dev {worst:.3f} (sampling noise of the init RMS + tiny-|g| entries)")
allr = [v for r in runs.values() if r["config"]["log_ratios"] for x in r["main"]["log"] for v in x["ratio"]]
check("all logged ratios finite and positive", all(math.isfinite(v) and v > 0 for v in allr), f"{len(allr):,} values")

# ---------------------------------------------------------------- 5. schedule comparison
ss = load(R("schedules_summary.json"))
sched = [r for r in runs.values() if r["group"] == "sched"]
for seed in {r["config"]["seed"] for r in sched}:
    rs = [r for r in sched if r["config"]["seed"] == seed]
    check(f"seed {seed}: all {len(rs)} schedule runs share one init", len({r["meta"]["init_hash"] for r in rs}) == 1)
    h200 = {next(x["data_hash"] for x in r["main"]["log"] if x["step"] == 200) for r in rs}
    check(f"seed {seed}: all schedule runs saw identical batches 1..200", len(h200) == 1)
cnt = {k: sum(1 for r in sched if f"{r['config']['schedule']}{r['config']['total_steps']}" == k) for k in ("cosine300", "wsd300")}
check("equal tuning effort (runs) for cosine and WSD", cnt["cosine300"] == cnt["wsd300"], str(cnt))
for k in ("cosine300", "wsd300"):
    sel = ss["selected"][k]
    vals = [r["main"]["evals"]["200"]["val"] for r in sched if f"{r['config']['schedule']}{r['config']['total_steps']}" == k and r["config"]["peak_lr"] == sel["lr"]]
    check(f"{k}: selected-LR val@200 mean recomputed", abs(sum(vals) / len(vals) - sel["val200_mean"]) < 1e-9 and len(vals) == 3, f"{sum(vals)/len(vals):.4f}")
    means = {}
    for r in sched:
        if f"{r['config']['schedule']}{r['config']['total_steps']}" == k:
            means.setdefault(r["config"]["peak_lr"], []).append(r["main"]["evals"]["200"]["val"])
    best = min((lr_ for lr_, v in means.items() if len(v) == 3), key=lambda lr_: sum(means[lr_]) / 3)
    check(f"{k}: selected LR is the 3-seed argmin", best == sel["lr"], f"{best}")
ver = load(R("checkpoint_verification.json"))
check("step-200 checkpoints reload to the logged val loss, weights hash and optimizer step",
      all(v["sha_match"] and abs(v["reloaded_val"] - v["logged_val"]) < 1e-4 and v["optimizer_steps"] == [v["step"]] for v in ver),
      "; ".join(f"{v['run_id'].split('__')[2]}/{v['segment']}@{v['step']}: {v['reloaded_val']:.5f}" for v in ver))

# ---------------------------------------------------------------- 6. LR sweep minima and the 4096 extrapolation
sw = load(R("lr_sweep_summary.json"))
sweep = [r for r in runs.values() if r["group"] == "sweep"]
for wd in (256, 512, 1024):
    by = {}
    for r in sweep:
        if r["config"]["width"] == wd:
            by.setdefault(r["config"]["peak_lr"], []).append(r["main"]["evals"]["300"]["val"])
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    best = min(mean, key=mean.get)
    lrs = sorted(mean)
    check(f"width {wd}: grid minimum recomputed", best == sw["minima"][str(wd)]["grid_min_lr"], f"{best} val {mean[best]:.4f} ({len(by[best])} seeds)")
    check(f"width {wd}: minimum not at grid edge", best not in (lrs[0], lrs[-1]), f"grid {lrs[0]}..{lrs[-1]}")
    i = lrs.index(best)
    xs = [math.log2(lrs[j]) for j in (i - 1, i, i + 1)]
    ys = [mean[lrs[j]] for j in (i - 1, i, i + 1)]
    # vertex of the parabola through 3 points, closed form
    d = (xs[0] - xs[1]) * (xs[0] - xs[2]) * (xs[1] - xs[2])
    A = (xs[2] * (ys[1] - ys[0]) + xs[1] * (ys[0] - ys[2]) + xs[0] * (ys[2] - ys[1])) / d
    B = (xs[2] ** 2 * (ys[0] - ys[1]) + xs[1] ** 2 * (ys[2] - ys[0]) + xs[0] ** 2 * (ys[1] - ys[2])) / d
    vtx = 2 ** (-B / (2 * A))
    check(f"width {wd}: parabola vertex recomputed", abs(vtx / sw["minima"][str(wd)]["vertex_lr"] - 1) < 1e-6, f"{vtx:.4g}")
    seeds = {r["config"]["seed"] for r in sweep if r["config"]["width"] == wd}
    shared = {json.dumps({k: v for k, v in r["config"].items() if k not in ("width", "n_head", "peak_lr", "seed", "data_seed")}, sort_keys=True)
              for r in sweep}
    check(f"width {wd}: sweep runs differ only in width/LR/seed", len(shared) == 1)
pts = [sw["fit"]["lr_star_used"][str(wd)] for wd in (256, 512, 1024)]
X = [math.log2(x) for x in (256, 512, 1024)]
Y = [math.log2(p) for p in pts]
xb, yb = sum(X) / 3, sum(Y) / 3
slope = sum((x - xb) * (y - yb) for x, y in zip(X, Y)) / sum((x - xb) ** 2 for x in X)
pred = 2 ** (yb + slope * (12 - xb))
check("log-log slope and width-4096 LR recomputed by hand", abs(slope - sw["fit"]["log2_lr_star = a + b log2 width"]["b"]) < 1e-9
      and abs(pred / sw["fit"]["lr_at_4096"] - 1) < 1e-9, f"slope {slope:.3f}, LR(4096) {pred:.3g}")
params = [sw["model_per_width"][str(wd)]["n_params"] for wd in (256, 512, 1024)]
check("parameter count rises with width", params[0] < params[1] < params[2], str(params))

# ---------------------------------------------------------------- 7. run-file integrity (parallel workers)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ledger = [json.loads(l) for l in open(R("modal_ledger.jsonl"))]
ok_ids = [x["run_id"] for x in ledger if x["status"] == "ok"]
check("ledger has no duplicate successful run ids", len(ok_ids) == len(set(ok_ids)), f"{len(ok_ids)} jobs")
names_ok = all(os.path.basename(p)[:-5] == r["run_id"] and os.path.basename(os.path.dirname(p)) == r["group"] for p, r in runs.items())
check("every result file holds the run its name claims", names_ok, f"{len(runs)} files")
check("every successful job has exactly one result file", set(ok_ids) == {r["run_id"] for r in runs.values()})
task_ids = [r["env"]["modal_task_id"] for r in runs.values()]
check("each result came from a recorded Modal task", all(task_ids), f"{len(set(task_ids))} distinct containers for {len(task_ids)} runs")
digests = [hashlib.sha256(json.dumps(r["main"]["log"][-1], sort_keys=True).encode()).hexdigest() for r in runs.values()]
check("no two runs share an identical final log record (no overwrite / crossed outputs)", len(digests) == len(set(digests)))
import plans  # noqa: E402
planned = plans.schedules(lrs=plans.SCHED_LRS + [0.00238, 0.00336]) + plans.schedules(lrs=[0.002, 0.00238, 0.00283], seeds=[2024, 7]) + \
          plans.sweep() + plans.sweep(widths=[256], lrs=[0.00283, 0.00566]) + plans.sweep(widths=[512], lrs=[0.000707, 0.00141]) + \
          plans.sweep(widths=[1024], lrs=[0.000354, 0.000707]) + plans.sweep_seeds() + plans.warmup_controls(0.002, seeds=[1337, 2024, 7]) + \
          plans.final_stop200(0.00238, 0.002)
have = {r["run_id"]: r for r in runs.values()}
missing = [j["run_id"] for j in planned if j["run_id"] not in have]
check("every planned candidate has a result", not missing, f"{len(planned)} planned, missing {missing}")
cfg_match = all(all(have[j["run_id"]]["config"][k] == (list(v) if isinstance(v, tuple) else v) for k, v in j["cfg"].items() if k != "branches")
                for j in planned)
check("stored config of every run equals its planned config", cfg_match)
finite = all(math.isfinite(e["val"]) for r in runs.values() for seg in [r["main"], *r["branches"].values()] for e in seg["evals"].values())
check("all validation losses finite", finite)
check("no run diverged (non-finite) anywhere", all(r["main"]["diverged_at"] is None for r in runs.values()))

failed = [c for c in CHECKS if not c["passed"]]
with open(R("evidence_audit.json"), "w") as f:
    json.dump({"n_checks": len(CHECKS), "n_failed": len(failed), "checks": CHECKS}, f, indent=1)
print(f"\n{len(CHECKS)} checks, {len(failed)} failed")
assert not failed
