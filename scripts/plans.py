"""
Every GPU run of the study, as data. A plan is a list of jobs {run_id, group, cfg, ckpt}.
The run_id encodes everything that varies, so two jobs can never write the same result file.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- shared settings (everything not listed takes s11lab.runner.DEFAULTS) ---------------------------------
WARMUP = 30          # 10% of the planned 300 steps
TOTAL = 300
STOP = 200           # the early stop at step 200
DECAY_FRAC = 0.1     # WSD decays over the last 10% of whatever horizon it is given (course: "last ~10%")

# Schedule comparison (base model, width 384): one shared sqrt(2)-spaced LR grid for BOTH schedules
SCHED_LRS = [1e-3, 1.41e-3, 2e-3, 2.83e-3, 4e-3, 5.66e-3, 8e-3]
# Width sweep, coarse: factor-2 grid, same for all widths
SWEEP_WIDTHS = [256, 512, 1024]
SWEEP_LRS = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]


def rid(group, width, sched, total, lr, warmup=WARMUP, bc=True, seed=1337, extra=""):
    return f"{group}__w{width}__{sched}{total}__lr{lr:.3g}__wu{warmup}__bc{int(bc)}__s{seed}{extra}"


def job(group, cfg, ckpt=False, extra=""):
    c = dict(cfg)
    r = rid(group, c.get("width", 384), c["schedule"], c["total_steps"], c["peak_lr"], c.get("warmup", WARMUP),
            c.get("bias_correction", True), c.get("seed", 1337), extra)
    return {"run_id": r, "group": group, "cfg": c, "ckpt": ckpt}


def bench():
    return [job("bench", dict(width=w, schedule="wsd", total_steps=TOTAL, stop_step=40, peak_lr=1e-3, eval_every=20))
            for w in (384, 1024)]


def schedules(lrs=SCHED_LRS, seeds=(1337,)):
    """Cosine vs WSD, both planned for 300 steps, same model/init/data, each tuned over the same LR grid.
    Per LR: cosine-300, WSD-300 (+ a branch that starts decaying at 180 so it finishes at 200), cosine-200.
    A seed sets both the init and the data order, and is shared by the three runs (paired comparison)."""
    jobs = []
    for lr, seed in [(lr, s) for lr in lrs for s in seeds]:
        base = dict(width=384, peak_lr=lr, warmup=WARMUP, eval_every=10, seed=seed, data_seed=seed)
        jobs.append(job("sched", dict(base, schedule="cosine", total_steps=TOTAL, stop_step=TOTAL)))
        jobs.append(job("sched", dict(base, schedule="wsd", total_steps=TOTAL, stop_step=TOTAL, decay_frac=DECAY_FRAC,
                                      branches=[{"name": "wsd_branch_decay180to200", "from_step": 180, "schedule": "wsd",
                                                 "total_steps": STOP, "decay_frac": DECAY_FRAC,
                                                 "eval_steps": [190, 200]}])))
        jobs.append(job("sched", dict(base, schedule="cosine", total_steps=STOP, stop_step=STOP)))
    return jobs


def final_stop200(cos_lr, wsd_lr):
    """The two tuned configs re-run with a literal stop at step 200 and checkpoints saved there.
    The WSD run also branches from its own step-200 checkpoint: a WSD schedule planned for 222 steps is flat
    through step 200 and decays over 201..222, so it is the step-200 checkpoint "finished" with a 10% decay."""
    out = []
    for sched, lr in (("cosine", cos_lr), ("wsd", wsd_lr)):
        cfg = dict(width=384, schedule=sched, total_steps=TOTAL, stop_step=STOP, peak_lr=lr,
                   warmup=WARMUP, decay_frac=DECAY_FRAC, eval_every=10, ckpt_steps=[STOP])
        if sched == "wsd":
            cfg["branches"] = [{"name": "wsd_finish_from200", "from_step": STOP, "schedule": "wsd", "total_steps": 222,
                                "decay_frac": DECAY_FRAC, "eval_steps": [210, 222], "ckpt_steps": [222]}]
        out.append(job("final", cfg, ckpt=True))
    return out


def warmup_controls(lr, seeds=(1337,)):
    """Same WSD-300 run as the schedule comparison, with warm-up and bias correction switched."""
    out = []
    for (wu, bc), s in [(x, s) for x in ((0, True), (60, True), (WARMUP, False), (0, False)) for s in seeds]:
        out.append(job("warmup", dict(width=384, schedule="wsd", total_steps=TOTAL, stop_step=TOTAL, peak_lr=lr,
                                      warmup=wu, bias_correction=bc, decay_frac=DECAY_FRAC, eval_every=10,
                                      allow_divergence=True, seed=s, data_seed=s)))
    return out


def sweep_seeds(seeds=(2024, 7)):
    """Extra seeds around each width's minimum (chosen after the coarse + sqrt(2) refinement)."""
    near = {256: [2e-3, 2.83e-3, 4e-3, 5.66e-3], 512: [1e-3, 1.41e-3, 2e-3], 1024: [2.5e-4, 3.54e-4, 5e-4, 7.07e-4]}
    return [j for w, lrs in near.items() for j in sweep(widths=[w], lrs=lrs, seeds=seeds)]


def sweep(widths=SWEEP_WIDTHS, lrs=SWEEP_LRS, seeds=(1337,), group="sweep"):
    out = []
    for w in widths:
        for lr in lrs:
            for s in seeds:
                out.append(job(group, dict(width=w, schedule="wsd", total_steps=TOTAL, stop_step=TOTAL, peak_lr=lr,
                                           warmup=WARMUP, decay_frac=DECAY_FRAC, eval_every=50, seed=s, data_seed=s,
                                           allow_divergence=True)))
    return out


def from_file(path):
    with open(path) as f:
        return json.load(f)


PLANS = {"bench": bench, "schedules": schedules}
