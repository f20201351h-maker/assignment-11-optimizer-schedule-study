"""
Modal runner for every GPU experiment.

    modal run scripts/modal_app.py --plan schedules
    modal run scripts/modal_app.py --plan sweep --args '{"widths": [256], "lrs": [0.003]}'

Each job runs on its own A10G container (max 4 at once, to leave room for other apps in the workspace).
Results come back as return values and are written locally to results/runs/<group>/<run_id>.json; a job whose
file already exists is skipped. Checkpoints (only for the final step-200 runs) go to the Volume
era-v5-s11-optim under /<run_id>/, one directory per run, so concurrent workers never share a path.
"""
import json
import os
import sys
import time

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

APP_NAME = "era-v5-s11-optimizers"
app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "numpy==2.1.3")
         .add_local_dir(os.path.join(ROOT, "s11lab"), "/root/s11lab")
         .add_local_file(os.path.join(ROOT, "data", "input.txt"), "/root/data/input.txt"))
vol = modal.Volume.from_name("era-v5-s11-optim", create_if_missing=True)
GPU = "A10G"


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=3600, max_containers=4)
def run_job(job):
    t_start = time.time()
    sys.path.insert(0, "/root")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from s11lab.data import CharData, load_text
    from s11lab.runner import Trainer
    text, sha, _ = load_text("/root/data")
    data = CharData(text)
    p = torch.cuda.get_device_properties(0)
    env = {"gpu_name": torch.cuda.get_device_name(0), "gpu_mem_GiB": round(p.total_memory / 2**30, 2),
           "compute_capability": f"{p.major}.{p.minor}", "torch": torch.__version__, "cuda": torch.version.cuda,
           "data_sha256": sha, "modal_task_id": os.environ.get("MODAL_TASK_ID"), "requested_gpu": GPU}
    out_dir = f"/vol/{job['run_id']}" if job.get("ckpt") else None
    res = Trainer(job["cfg"], data, "cuda", out_dir).run()
    if out_dir:
        vol.commit()
    res.update(run_id=job["run_id"], group=job["group"], env=env, wall_seconds=time.time() - t_start)
    return res


def _round(x, sig=6):
    if isinstance(x, float):
        return float(f"{x:.{sig}g}")
    if isinstance(x, list):
        return [_round(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    return x


@app.local_entrypoint()
def main(plan: str, args: str = "{}"):
    if plan == "verify":
        return verify()
    import plans
    jobs = getattr(plans, plan)(**json.loads(args))
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate run ids in plan"
    todo = []
    for j in jobs:
        path = os.path.join(ROOT, "results", "runs", j["group"], j["run_id"] + ".json")
        if os.path.exists(path):
            print("skip (exists)", j["run_id"])
        else:
            todo.append((j, path))
    print(f"plan {plan}: {len(jobs)} jobs, {len(todo)} to run on {GPU}")
    ledger = os.path.join(ROOT, "results", "modal_ledger.jsonl")
    t0 = time.time()
    for (j, path), res in zip(todo, run_job.map([j for j, _ in todo], return_exceptions=True, order_outputs=True)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(res, Exception):
            print("FAILED", j["run_id"], repr(res))
            with open(ledger, "a") as f:
                f.write(json.dumps({"run_id": j["run_id"], "plan": plan, "status": "failed", "error": repr(res)}) + "\n")
            continue
        assert res["run_id"] == j["run_id"], (res["run_id"], j["run_id"])  # no crossed outputs
        if j["group"] == "sweep" or j["group"].startswith("sweep"):
            for rec in res["main"]["log"]:  # keep the per-tensor ratio, drop the bulkier diagnostics
                rec.pop("w_norm", None)
                rec.pop("adam_rms_over_lr", None)
        with open(path, "w") as f:
            json.dump(_round(res), f)
        ev = res["main"]["evals"]
        last = max(ev, key=int) if ev else None
        print(f"done {j['run_id']}  {res['wall_seconds']:.0f}s  {res['env']['gpu_name']}  "
              f"val@{last}={ev[last]['val'] if last else float('nan'):.4f}  diverged={res['main']['diverged_at']}")
        with open(ledger, "a") as f:
            f.write(json.dumps({"run_id": j["run_id"], "plan": plan, "status": "ok", "gpu": res["env"]["gpu_name"],
                                "wall_seconds": round(res["wall_seconds"], 1),
                                "train_seconds": round(res["main"]["seconds"], 1),
                                "modal_task_id": res["env"]["modal_task_id"]}) + "\n")
    print(f"plan {plan} finished in {time.time() - t0:.0f}s")


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=900)
def verify_ckpts(items):
    """Reload saved checkpoints, rebuild the model from the stored config, re-evaluate on the whole val split."""
    sys.path.insert(0, "/root")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from s11lab.data import CharData, load_text
    from s11lab.model import GPT, GPTConfig
    from s11lab.runner import eval_loss, state_hash, val_windows
    text, _, _ = load_text("/root/data")
    data = CharData(text)
    out = []
    for it in items:
        ck = torch.load(it["path"], map_location="cpu", weights_only=False)
        c = ck["config"]
        m = GPT(GPTConfig(block_size=c["block_size"], vocab_size=data.vocab_size, n_layer=c["n_layer"], n_head=c["n_head"],
                          n_embd=c["width"], dropout=0.0, bias=False))
        m.load_state_dict(ck["model"])
        opt_steps = {int(float(s["step"])) for s in ck["opt"]["state"].values()}
        Xv, Yv = val_windows(data, c["block_size"])
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        val = eval_loss(m.cuda(), Xv, Yv, ctx, "cuda")
        out.append({**it, "reloaded_val": val, "stored_step": ck["step"], "optimizer_steps": sorted(opt_steps),
                    "weights_sha_reloaded": state_hash(ck["model"]), "gpu": torch.cuda.get_device_name(0)})
    return out


def verify():
    import glob
    items = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "final", "*.json"))):
        r = json.load(open(p))
        segs = {"main": r["main"], **r["branches"]}
        for name, seg in segs.items():
            for step, info in seg["ckpts"].items():
                items.append({"run_id": r["run_id"], "segment": name, "step": int(step), "path": info["path"],
                              "logged_val": seg["evals"][str(step)]["val"], "weights_sha_logged": info["weights_sha"]})
    res = verify_ckpts.remote(items)
    for x in res:
        x["abs_val_diff"] = abs(x["reloaded_val"] - x["logged_val"])
        x["sha_match"] = x["weights_sha_reloaded"] == x["weights_sha_logged"]
        print(x["run_id"], x["segment"], x["step"], "logged", round(x["logged_val"], 5), "reloaded", round(x["reloaded_val"], 5),
              "opt steps", x["optimizer_steps"], "sha match", x["sha_match"])
    with open(os.path.join(ROOT, "results", "checkpoint_verification.json"), "w") as f:
        json.dump(res, f, indent=1)
