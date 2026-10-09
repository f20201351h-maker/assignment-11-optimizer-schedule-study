"""
CPU checks for the support code. Run:  python tests/test_core.py
Writes results/tests_core.json. Any failure raises.
"""
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from s11lab.data import CharData, load_text  # noqa: E402
from s11lab.model import GPT, GPTConfig  # noqa: E402
from s11lab.optim import ManualAdamW  # noqa: E402
from s11lab.runner import Trainer, full_config, state_hash  # noqa: E402
from s11lab.schedules import check_boundaries, wsd_decay_start  # noqa: E402

REPORT = []


def check(name, cond, detail=""):
    REPORT.append({"check": name, "passed": bool(cond), "detail": str(detail)})
    assert cond, f"FAILED {name}: {detail}"
    print(f"[ok] {name} {detail}")


def tiny_gpt(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32, bias=False))


def test_manual_adamw_matches_torch():
    """ManualAdamW(bias_correction=True) vs torch.optim.AdamW on the same model, same grads, 15 steps, fp64."""
    torch.set_default_dtype(torch.float64)
    try:
        a, b = tiny_gpt(), tiny_gpt()
        check("two tiny models start identical", state_hash(a.state_dict()) == state_hash(b.state_dict()))
        groups = lambda m: [{"params": [p for p in m.parameters() if p.dim() >= 2], "weight_decay": 0.1},
                            {"params": [p for p in m.parameters() if p.dim() < 2], "weight_decay": 0.0}]
        oa = torch.optim.AdamW(groups(a), lr=3e-3, betas=(0.9, 0.99), eps=1e-8, foreach=False)
        ob = ManualAdamW(groups(b), lr=3e-3, betas=(0.9, 0.99), eps=1e-8, bias_correction=True)
        g = torch.Generator().manual_seed(1)
        worst = 0.0
        for t in range(15):
            x = torch.randint(0, 65, (4, 16), generator=g)
            for m, o in ((a, oa), (b, ob)):
                lr = 3e-3 * min(1.0, (t + 1) / 5)
                for grp in o.param_groups:
                    grp["lr"] = lr
                loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 65), x[:, 1:].reshape(-1))
                loss.backward()
                o.step()
                o.zero_grad()
            worst = max(worst, max(float((p - q).detach().abs().max()) for p, q in zip(a.parameters(), b.parameters())))
        check("ManualAdamW(bias_correction=True) == torch AdamW over 15 steps (fp64)", worst < 1e-12, f"max |diff| {worst:.2e}")
    finally:
        torch.set_default_dtype(torch.float32)


def test_no_bias_correction_first_step():
    """At t=1 without bias correction the step is lr*(1-b1)*g/(sqrt(1-b2)*|g|) = 3.162*lr for betas (0.9, 0.999)."""
    w = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    o = ManualAdamW([w], lr=1e-3, betas=(0.9, 0.999), eps=0.0, bias_correction=False)
    w.grad = torch.tensor([0.5], dtype=torch.float64)
    o.step()
    step = 1.0 - float(w)
    check("no-BC first step = 0.1/sqrt(0.001) * lr", abs(step / 1e-3 - 0.1 / math.sqrt(0.001)) < 1e-9, f"{step / 1e-3:.6f} lr")


def test_schedules():
    out = check_boundaries(peak=1e-3, warmup=30, total=300, decay_frac=0.1, min_ratio=0.1)
    check("schedule boundary values", True, out)
    check("WSD planned 300 decays over steps 271..300", wsd_decay_start(300, 0.1) == 270)
    check("WSD planned 200 decays over steps 181..200", wsd_decay_start(200, 0.1) == 180)


def tiny_run_cfg(**kw):
    c = dict(width=64, n_layer=2, head_dim=32, block_size=32, micro_batch=4, grad_accum=2, total_steps=24,
             stop_step=24, warmup=4, peak_lr=3e-3, eval_every=8, n_train_eval_seqs=8, precision="fp32")
    c.update(kw)
    return c


def test_trainer(data):
    r_cos = Trainer(tiny_run_cfg(schedule="cosine"), data, "cpu").run()
    r_wsd = Trainer(tiny_run_cfg(schedule="wsd", branches=[
        {"name": "decay_to_20", "from_step": 18, "schedule": "wsd", "total_steps": 20, "decay_frac": 0.1}]),
        data, "cpu").run()
    check("cosine and WSD runs start from identical weights", r_cos["meta"]["init_hash"] == r_wsd["meta"]["init_hash"],
          r_cos["meta"]["init_hash"])
    check("cosine and WSD runs see identical batches",
          [x["data_hash"] for x in r_cos["main"]["log"]] == [x["data_hash"] for x in r_wsd["main"]["log"]])
    check("identical step-1 loss (same init, same batch)", r_cos["main"]["log"][0]["loss"] == r_wsd["main"]["log"][0]["loss"])
    check("identical LR through warm-up", all(a["lr"] == b["lr"] for a, b in zip(r_cos["main"]["log"][:4], r_wsd["main"]["log"][:4])))
    br = r_wsd["branches"]["decay_to_20"]
    main_hashes = [x["data_hash"] for x in r_wsd["main"]["log"]]
    check("branch replays the main run's batches for steps 19-20", [x["data_hash"] for x in br["log"]] == main_hashes[18:20])
    check("branch ends at the min LR", abs(br["log"][-1]["lr"] - 3e-4) < 1e-15, br["log"][-1]["lr"])
    ratios = [r for x in r_wsd["main"]["log"] for r in x["ratio"]]
    check("all update-to-weight ratios finite and positive", all(math.isfinite(r) and r > 0 for r in ratios), f"{len(ratios)} values")
    check("one ratio per parameter tensor", len(r_wsd["main"]["log"][0]["ratio"]) == len(r_wsd["meta"]["tensors"]))


def test_ratio_is_the_actual_update(data):
    """Independent re-measurement: run one step by hand and compare ||dW||/||W|| with what the Trainer logged."""
    cfg = tiny_run_cfg(stop_step=1, total_steps=24)
    tr = Trainer(cfg, data, "cpu")
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    rec, _ = tr.one_step(1, 1e-3)
    worst = 0.0
    for (n, p), logged, adam_rms in zip(tr.model.named_parameters(), rec["ratio"], rec["adam_rms_over_lr"]):
        d = (p.detach() - before[n]).double()
        mine = float(d.norm() / before[n].double().norm())
        worst = max(worst, abs(mine - logged) / mine)
    check("logged ratio equals an independent ||W_after - W_before|| / ||W_before||", worst < 1e-5, f"max rel err {worst:.1e}")
    # at step 1 Adam's step is lr * g/(|g| + eps) in every coordinate, i.e. +-lr except where |g| is comparable
    # to eps (a few tiny gradients pull the RMS slightly under 1, hence the 1% tolerance)
    vals = rec["adam_rms_over_lr"]
    check("step 1: Adam part of the update has RMS ~= lr in every tensor", all(abs(v - 1) < 1e-2 for v in vals),
          f"min {min(vals):.5f} max {max(vals):.5f}")


def test_width_family():
    rows = []
    for w in (256, 384, 512, 1024):
        c = full_config({"width": w})
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=256, vocab_size=65, n_layer=c["n_layer"], n_head=c["n_head"], n_embd=w))
        n = sum(p.numel() for p in m.parameters())
        formula = 65 * w + 256 * w + c["n_layer"] * (12 * w * w + 2 * w) + w
        rows.append((w, c["n_head"], n))
        check(f"width {w}: params match 65C + 256C + L(12C^2 + 2C) + C", n == formula, f"{n:,} heads={c['n_head']}")
        others = {k: v for k, v in c.items() if k not in ("width", "n_head")}
        base = {k: v for k, v in full_config({"width": 256}).items() if k not in ("width", "n_head")}
        check(f"width {w}: nothing but width/heads differs from the width-256 config", others == base)
    check("parameter count increases with width", all(a[2] < b[2] for a, b in zip(rows, rows[1:])), rows)


if __name__ == "__main__":
    text, sha, _ = load_text(os.path.join(ROOT, "data"))
    data = CharData(text)
    check("Tiny Shakespeare: 65 symbols", data.vocab_size == 65, sha[:16])
    test_manual_adamw_matches_torch()
    test_no_bias_correction_first_step()
    test_schedules()
    test_trainer(data)
    test_ratio_is_the_actual_update(data)
    test_width_family()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "tests_core.json"), "w") as f:
        json.dump({"torch": torch.__version__, "checks": REPORT}, f, indent=1)
    print(f"\n{len(REPORT)} checks passed")
