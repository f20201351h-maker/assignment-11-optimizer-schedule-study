"""
Learning-rate schedules. `step` is the 1-based optimizer step, i.e. the LR used for the k-th call to opt.step().

cosine : linear warm-up to `peak` over `warmup` steps, then cosine down to min_ratio*peak, reached exactly at `total`.
         This is the nanoGPT get_lr, with the horizon `total` baked into every value after warm-up.
wsd    : linear warm-up, flat at `peak`, then a linear decay to min_ratio*peak over the last `decay_frac` of `total`.
         The flat phase never looks at `total`, which is the property this study cares about.
constant: warm-up then flat forever (used for the "no decay" view and for the warm-up controls).
"""
import math


def warmup_lr(step, peak, warmup):
    return peak * step / warmup


def cosine_lr(step, peak, warmup, total, min_ratio=0.1):
    lo = peak * min_ratio
    if warmup > 0 and step <= warmup:
        return warmup_lr(step, peak, warmup)
    if step >= total:
        return lo
    r = (step - warmup) / (total - warmup)
    return lo + 0.5 * (1.0 + math.cos(math.pi * r)) * (peak - lo)


def wsd_decay_start(total, decay_frac):
    """Last step that still uses the peak LR. Decay steps are decay_start+1 .. total."""
    return total - int(round(decay_frac * total))


def wsd_lr(step, peak, warmup, total, decay_frac=0.1, min_ratio=0.1):
    lo = peak * min_ratio
    if warmup > 0 and step <= warmup:
        return warmup_lr(step, peak, warmup)
    start = wsd_decay_start(total, decay_frac)
    if step <= start:
        return peak
    if step >= total:
        return lo
    r = (step - start) / (total - start)
    return peak + r * (lo - peak)


def constant_lr(step, peak, warmup, total=None, **_):
    if warmup > 0 and step <= warmup:
        return warmup_lr(step, peak, warmup)
    return peak


SCHEDULES = {"cosine": cosine_lr, "wsd": wsd_lr, "constant": constant_lr}


def lr_fn(name, peak, warmup, total, decay_frac=0.1, min_ratio=0.1):
    if name == "cosine":
        return lambda s: cosine_lr(s, peak, warmup, total, min_ratio)
    if name == "wsd":
        return lambda s: wsd_lr(s, peak, warmup, total, decay_frac, min_ratio)
    if name == "constant":
        return lambda s: constant_lr(s, peak, warmup)
    raise ValueError(name)


def check_boundaries(peak=1e-3, warmup=30, total=300, decay_frac=0.1, min_ratio=0.1):
    """Assertions on the values a schedule must take at its boundaries. Returns a dict of what was checked."""
    tol = 1e-15
    c = lambda s: cosine_lr(s, peak, warmup, total, min_ratio)
    w = lambda s: wsd_lr(s, peak, warmup, total, decay_frac, min_ratio)
    start = wsd_decay_start(total, decay_frac)
    out = {
        "cosine_step1": c(1), "cosine_at_warmup_end": c(warmup), "cosine_at_total": c(total),
        "cosine_midpoint": c(warmup + (total - warmup) // 2),
        "wsd_step1": w(1), "wsd_at_warmup_end": w(warmup), "wsd_decay_start": start,
        "wsd_at_decay_start": w(start), "wsd_first_decay_step": w(start + 1), "wsd_at_total": w(total),
    }
    assert abs(c(1) - peak / warmup) < tol and abs(w(1) - peak / warmup) < tol
    assert abs(c(warmup) - peak) < tol and abs(w(warmup) - peak) < tol
    assert abs(c(total) - peak * min_ratio) < tol and abs(w(total) - peak * min_ratio) < tol
    assert abs(c(warmup + (total - warmup) / 2) - (peak + peak * min_ratio) / 2) < 1e-12
    assert abs(w(start) - peak) < tol and w(start + 1) < peak
    assert all(c(s) <= c(s - 1) + tol for s in range(warmup + 1, total + 1)), "cosine must be non-increasing"
    assert all(w(s) <= w(s - 1) + tol for s in range(warmup + 1, total + 1)), "wsd must be non-increasing"
    assert all(w(s) == peak for s in range(warmup, start + 1)), "wsd must be flat in its stable phase"
    return out
