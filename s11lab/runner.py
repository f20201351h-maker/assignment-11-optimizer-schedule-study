"""
One training run of the base model, with everything the optimizer study needs logged.

What is reused from the training-loop audit (s11lab/model.py and s11lab/data.py are byte-for-byte copies of s10lab/):
  * nanoGPT GPT: pre-LN blocks, tied wte/lm_head, N(0, 0.02) init with residual projections at 0.02/sqrt(2L),
    no biases, no dropout;
  * Tiny Shakespeare, character level, 65 symbols, first 90% train / last 10% validation;
  * AdamW via GPT.configure_optimizer: fused on CUDA, weight decay only on tensors with dim >= 2;
  * global batch 32 x 2 micro-batches x 256 positions = 16,384 tokens per optimizer step, bf16 autocast,
    grad-norm clipping at 1.0 with the norm logged before clipping.

What is new:
  * batches are a pure function of (data_seed, step), so two runs with the same data_seed see identical tokens
    in identical order whatever their model or schedule is (checked through a running hash);
  * the per-tensor update-to-weight ratio ||W_t - W_{t-1}|| / ||W_{t-1}|| is measured from the ACTUAL parameter
    change made by opt.step(), every step, for every tensor;
  * validation is the whole validation split cut into non-overlapping 256-token windows, so a val loss has no
    sampling noise and is identical across runs;
  * a run can snapshot (model, optimizer) at a step and later "branch" from it with a different LR tail.
"""
import copy
import hashlib
import math
import os
import time

import torch
from torch.nn import functional as F

from .data import CharData
from .model import GPT, GPTConfig
from .optim import ManualAdamW
from .schedules import lr_fn

DEFAULTS = dict(
    width=384, n_layer=6, head_dim=64, block_size=256,
    micro_batch=32, grad_accum=2,
    total_steps=300, stop_step=300,
    schedule="wsd", peak_lr=1e-3, warmup=30, decay_frac=0.1, min_ratio=0.1,
    betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1, grad_clip=1.0, bias_correction=True,
    seed=1337, data_seed=1337, precision="bf16",
    eval_steps=None, eval_every=25, n_train_eval_seqs=128,
    snapshot_steps=(), ckpt_steps=(), branches=(), log_ratios=True,
    allow_divergence=False, divergence_loss=8.0,
)


def full_config(cfg):
    c = dict(DEFAULTS)
    unknown = set(cfg) - set(DEFAULTS) - {"run_id", "tag", "group"}
    assert not unknown, f"unknown config keys {unknown}"
    c.update(cfg)
    c["betas"] = tuple(c["betas"])
    assert c["width"] % c["head_dim"] == 0
    c["n_head"] = c["width"] // c["head_dim"]
    assert 1 <= c["stop_step"] <= c["total_steps"]
    return c


def tensor_hash(tensors, n=16):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:n]


def state_hash(sd):
    return tensor_hash([sd[k] for k in sorted(sd)])


class Batches:
    """Training batches as a pure function of (data_seed, step)."""

    def __init__(self, data, B, accum, T, data_seed):
        self.train = data.ids["train"]
        self.B, self.accum, self.T, self.seed = B, accum, T, data_seed

    def indices(self, step):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        return torch.randint(len(self.train) - self.T - 1, (self.B * self.accum,), generator=g)

    def get(self, step):
        ix = self.indices(step)
        chunk = torch.stack([self.train[i: i + self.T + 1] for i in ix.tolist()])
        xs, ys = chunk[:, :-1], chunk[:, 1:]
        return [(xs[k * self.B:(k + 1) * self.B], ys[k * self.B:(k + 1) * self.B]) for k in range(self.accum)], ix


def val_windows(data, T):
    """Whole validation split, non-overlapping windows of T predictions (stride T)."""
    v = data.ids["val"]
    n = (len(v) - 1) // T
    chunk = torch.stack([v[i * T: i * T + T + 1] for i in range(n)])
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


def train_eval_windows(data, T, n, seed=4242):
    g = torch.Generator().manual_seed(seed)
    t = data.ids["train"]
    ix = torch.randint(len(t) - T - 1, (n,), generator=g)
    chunk = torch.stack([t[i: i + T + 1] for i in ix.tolist()])
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


@torch.no_grad()
def eval_loss(model, X, Y, ctx, device, bs=64):
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
        with ctx:
            logits = model(x)
        tot += float(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        n += y.numel()
    model.train(was)
    return tot / n


def tensor_labels(model):
    """(name, label, family, decayed) for every unique parameter tensor, in model.parameters() order."""
    out = []
    for name, p in model.named_parameters():
        short = name.replace("transformer.", "")
        if short == "wte.weight":
            fam = "embedding (tied wte = lm_head)"
            short = "wte=lm_head"
        elif short == "wpe.weight":
            fam = "position embedding"
            short = "wpe"
        elif "ln" in short:
            fam = "LayerNorm gain"
        elif "attn.c_attn" in short:
            fam = "attn qkv"
        elif "attn.c_proj" in short:
            fam = "attn out proj"
        elif "mlp.c_fc" in short:
            fam = "mlp up"
        elif "mlp.c_proj" in short:
            fam = "mlp down"
        else:
            fam = "other"
        short = short.replace(".weight", "")
        out.append({"name": name, "label": short, "family": fam, "numel": p.numel(), "shape": list(p.shape),
                    "decayed": p.dim() >= 2})
    return out


def make_optimizer(model, c, device):
    if c["bias_correction"]:
        return model.configure_optimizer(c["peak_lr"], c["weight_decay"], c["betas"], device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return ManualAdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                        {"params": no_decay, "weight_decay": 0.0}],
                       lr=c["peak_lr"], betas=c["betas"], eps=c["eps"], bias_correction=False)


def opt_step_count(opt):
    steps = {int(float(s["step"])) for s in opt.state.values()}
    assert len(steps) == 1, steps
    return steps.pop()


class Trainer:
    def __init__(self, cfg, data, device, out_dir=None):
        self.c = c = full_config(cfg)
        self.device = device
        self.out_dir = out_dir
        self.ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16) if c["precision"] == "bf16"
                    else torch.autocast(device_type=device, enabled=False))
        torch.manual_seed(c["seed"])
        mcfg = GPTConfig(block_size=c["block_size"], vocab_size=data.vocab_size, n_layer=c["n_layer"],
                         n_head=c["n_head"], n_embd=c["width"], dropout=0.0, bias=False)
        model = GPT(mcfg)  # built on CPU from the seed, so the init is identical on any device
        self.init_hash = state_hash(model.state_dict())
        self.model = model.to(device)
        self.opt = make_optimizer(self.model, c, device)
        self.params = list(self.model.parameters())
        self.labels = tensor_labels(self.model)
        self.decay_mask = [l["decayed"] for l in self.labels]
        self.numel = [l["numel"] for l in self.labels]
        self.batches = Batches(data, c["micro_batch"], c["grad_accum"], c["block_size"], c["data_seed"])
        self.Xv, self.Yv = val_windows(data, c["block_size"])
        self.Xt, self.Yt = train_eval_windows(data, c["block_size"], c["n_train_eval_seqs"])
        self.n_params = sum(p.numel() for p in self.params)
        self.n_params_non_emb = self.n_params - data.vocab_size * c["width"] - c["block_size"] * c["width"]
        self.data_hash = hashlib.sha256()
        self.lr_main = lr_fn(c["schedule"], c["peak_lr"], c["warmup"], c["total_steps"], c["decay_frac"],
                             c["min_ratio"])

    # ------------------------------------------------------------------------------------------------------
    def evaluate(self):
        return {"val": eval_loss(self.model, self.Xv, self.Yv, self.ctx, self.device),
                "train_eval": eval_loss(self.model, self.Xt, self.Yt, self.ctx, self.device)}

    def one_step(self, step, lr):
        c = self.c
        for g in self.opt.param_groups:
            g["lr"] = lr
        window, ix = self.batches.get(step)
        N = sum(y.numel() for _, y in window)
        loss_sum = 0.0
        for x, y in window:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self.ctx:
                logits = self.model(x)
            s = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            (s / N).backward()
            loss_sum = loss_sum + s.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, c["grad_clip"])
        prev = [p.detach().clone() for p in self.params] if c["log_ratios"] else None
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        rec = {"step": step, "lr": lr}
        if prev is not None:
            with torch.no_grad():
                delta = torch._foreach_sub([p.detach() for p in self.params], prev)
                d_norm = torch.stack(torch._foreach_norm(delta))
                w_norm = torch.stack(torch._foreach_norm(prev))
                # Adam's own part of the step: add back the decoupled decay (lr*wd*W_prev) on decayed tensors
                wd_lr = lr * c["weight_decay"]
                adam_part = [d + wd_lr * w if dm else d for d, w, dm in zip(delta, prev, self.decay_mask)]
                a_norm = torch.stack(torch._foreach_norm(adam_part))
                numel = torch.tensor(self.numel, device=d_norm.device, dtype=torch.float32)
                stats = torch.stack([d_norm / w_norm, a_norm / numel.sqrt() / lr, w_norm]).float().cpu()
                scal = torch.stack([(loss_sum / N).float(), gnorm.float()]).cpu()
            rec["ratio"] = stats[0].tolist()
            rec["adam_rms_over_lr"] = stats[1].tolist()
            rec["w_norm"] = stats[2].tolist()
            rec["loss"] = float(scal[0])
            rec["grad_norm"] = float(scal[1])
        else:
            rec["loss"] = float(loss_sum / N)
            rec["grad_norm"] = float(gnorm)
        rec["clipped"] = rec["grad_norm"] > c["grad_clip"]
        return rec, ix

    def snapshot(self):
        return {"model": copy.deepcopy(self.model.state_dict()), "opt": copy.deepcopy(self.opt.state_dict()),
                "data_hash": self.data_hash.copy()}

    def restore(self, snap):
        self.model.load_state_dict(snap["model"])
        self.opt.load_state_dict(snap["opt"])
        self.data_hash = snap["data_hash"].copy()

    def save_ckpt(self, step, tag):
        sd = self.model.state_dict()
        info = {"step": step, "opt_step": opt_step_count(self.opt), "weights_sha": state_hash(sd)}
        assert info["opt_step"] == step, f"checkpoint at {step} but optimizer has taken {info['opt_step']} steps"
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"{tag}_step{step}.pt")
            torch.save({"model": sd, "opt": self.opt.state_dict(), "config": self.c, "step": step}, path)
            info["path"] = path
        return info

    # ------------------------------------------------------------------------------------------------------
    def run_segment(self, first, last, lr_of, eval_steps, tag, snapshot_steps=(), ckpt_steps=()):
        c = self.c
        log, evals, snaps, ckpts = [], {}, {}, {}
        diverged = None
        cuda = self.device == "cuda"
        t0 = time.perf_counter()
        for step in range(first, last + 1):
            lr = lr_of(step)
            rec, ix = self.one_step(step, lr)
            self.data_hash.update(ix.numpy().tobytes())
            rec["data_hash"] = self.data_hash.hexdigest()[:12]
            log.append(rec)
            bad = (not math.isfinite(rec["loss"])) or rec["loss"] > c["divergence_loss"] and step > c["warmup"]
            if c["log_ratios"] and not all(math.isfinite(r) for r in rec["ratio"]):
                bad = True
            if bad:
                diverged = step
                if not c["allow_divergence"]:
                    raise FloatingPointError(f"{tag}: non-finite or diverged at step {step}: loss {rec['loss']}")
                break
            if step in eval_steps:
                evals[step] = self.evaluate()
                assert all(math.isfinite(v) for v in evals[step].values()), evals[step]
            if step in snapshot_steps:
                assert opt_step_count(self.opt) == step
                snaps[step] = self.snapshot()
            if step in ckpt_steps:
                ckpts[step] = self.save_ckpt(step, tag)
        if cuda:
            torch.cuda.synchronize()
        return {"log": log, "evals": evals, "snaps": snaps, "ckpts": ckpts, "diverged_at": diverged,
                "seconds": time.perf_counter() - t0}

    def run(self):
        c = self.c
        stop = c["stop_step"]
        eval_steps = set(c["eval_steps"] or [])
        eval_steps |= set(range(c["eval_every"], stop + 1, c["eval_every"])) | {1, stop}
        eval_steps = {s for s in eval_steps if s <= stop}
        snap_steps = set(c["snapshot_steps"]) | {b["from_step"] for b in c["branches"]}
        init_eval = self.evaluate()
        main = self.run_segment(1, stop, self.lr_main, eval_steps, "main", snap_steps, set(c["ckpt_steps"]))
        result = {
            "config": {k: v for k, v in c.items() if k != "branches"}, "branches_config": list(c["branches"]),
            "meta": {"n_params": self.n_params, "n_params_non_embedding": self.n_params_non_emb,
                     "init_hash": self.init_hash, "n_val_tokens": int(self.Yv.numel()),
                     "n_train_eval_tokens": int(self.Yt.numel()), "tensors": self.labels},
            "init_eval": init_eval,
            "main": {k: v for k, v in main.items() if k != "snaps"},
            "final_data_hash": self.data_hash.hexdigest()[:16],
            "branches": {},
        }
        for b in c["branches"]:
            if main["diverged_at"] is not None:
                break
            lr_b = lr_fn(b["schedule"], c["peak_lr"], c["warmup"], b["total_steps"], b.get("decay_frac", c["decay_frac"]),
                         c["min_ratio"])
            # a branch is only a branch if it agrees with the main schedule up to the branch point
            for s in range(1, b["from_step"] + 1):
                assert abs(lr_b(s) - self.lr_main(s)) < 1e-15, f"branch {b['name']} differs from main at step {s}"
            self.restore(main["snaps"][b["from_step"]])
            assert opt_step_count(self.opt) == b["from_step"]
            seg = self.run_segment(b["from_step"] + 1, b["total_steps"], lr_b, set(b.get("eval_steps", [])) | {b["total_steps"]},
                                   b["name"], ckpt_steps=set(b.get("ckpt_steps", [])))
            seg["data_hash_at_end"] = self.data_hash.hexdigest()[:16]
            result["branches"][b["name"]] = seg
        return result
