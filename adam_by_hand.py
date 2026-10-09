"""
Experiments 1 and 2 (CPU only).

1. Adam by hand on one weight and five gradients, verified three ways:
     (a) `adam_by_hand`   : the recursive update in plain Python floats, no torch anywhere;
     (b) `adam_closed_form`: the same quantities from the non-recursive sums m_t = (1-b1) sum b1^(t-i) g_i, in exact
                            rational arithmetic (fractions) with a 50-digit Decimal square root;
     (c) torch.optim.Adam  : the gradient arrives through loss.backward(), m and v are read from optimizer.state,
                            and the step is the observed change of the parameter. float64 and float32.
   Also checked against the five-step table printed in the Session 11 lesson, at the precision it was printed.

2. Bias correction off vs on, same weight, same gradients, first 20 steps, plus a long tail to find out when the
   difference stops mattering under the criterion fixed below (written down before any of the numbers were made).

    python adam_by_hand.py      -> results/adam_*.csv|json, results/bias_correction*.csv|json, figures/*.png
"""
import csv
import json
import math
import os
import random
from decimal import Decimal, getcontext
from fractions import Fraction

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

# ---- the worked example of Session 11, Section 6 -------------------------------------------------------------
GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]
W0, LR, B1, B2, EPS = 1.0, 1e-3, 0.9, 0.999, 1e-8
# the lesson's table: t, g, m, v, m_hat, v_hat, step, w   (with the number of decimals it was printed with)
COURSE_TABLE = [
    (1, 0.50, 0.0500, 0.000250, 0.5000, 0.2500, -0.001000, 0.999000),
    (2, 0.40, 0.0850, 0.000410, 0.4474, 0.2050, -0.000988, 0.998012),
    (3, 0.60, 0.1365, 0.000769, 0.5037, 0.2567, -0.000994, 0.997018),
    (4, 0.45, 0.1678, 0.000971, 0.4881, 0.2431, -0.000990, 0.996028),
    (5, 0.55, 0.2061, 0.001273, 0.5032, 0.2550, -0.000996, 0.995031),
]
COURSE_DECIMALS = {"m": 4, "v": 6, "m_hat": 4, "v_hat": 4, "update": 6, "w": 6}

# ---- Experiment 2: the criterion, fixed before looking -------------------------------------------------------
# u_t is the change of the weight at step t. The difference "stops mattering" at the first step t* such that
#   |u_t(no BC) - u_t(BC)| / |u_t(BC)| < TOL   for t*, t*+1, ..., t*+WINDOW-1.
# The step is what bias correction changes directly; the weight gap it leaves behind is reported separately
# because it is cumulative and never closes.
TOL, WINDOW = 0.05, 10
TOL_SENSITIVITY = [0.01, 0.02, 0.05, 0.10]
N_PLOT, N_LONG = 20, 10_000

QUANTITIES = ["m", "v", "bc1", "bc2", "m_hat", "v_hat", "denom", "update", "w"]


def adam_by_hand(grads, w0=W0, lr=LR, b1=B1, b2=B2, eps=EPS, bias_correction=True):
    """Plain-float Adam. bc1/bc2 are the bias-correction denominators 1 - beta^t (1.0 when switched off)."""
    rows, m, v, w = [], 0.0, 0.0, w0
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1 = 1 - b1 ** t if bias_correction else 1.0
        bc2 = 1 - b2 ** t if bias_correction else 1.0
        m_hat, v_hat = m / bc1, v / bc2
        denom = math.sqrt(v_hat) + eps
        update = -lr * m_hat / denom
        w = w + update
        rows.append(dict(t=t, g=g, m=m, v=v, bc1=bc1, bc2=bc2, m_hat=m_hat, v_hat=v_hat, denom=denom,
                         update=update, w=w))
    return rows


def adam_closed_form(grads, w0=W0, lr=LR, b1=B1, b2=B2, eps=EPS):
    """Exact rationals for m, v, m_hat, v_hat from the summation form; Decimal(50 digits) for the square root."""
    getcontext().prec = 50
    F = lambda x: Fraction(str(x))
    g = [F(x) for x in grads]
    b1, b2, lr_, eps_ = F(b1), F(b2), F(lr), F(eps)
    D = lambda q: Decimal(q.numerator) / Decimal(q.denominator)
    rows, w = [], D(F(w0))
    for t in range(1, len(g) + 1):
        m = (1 - b1) * sum(b1 ** (t - i) * g[i - 1] for i in range(1, t + 1))
        v = (1 - b2) * sum(b2 ** (t - i) * g[i - 1] ** 2 for i in range(1, t + 1))
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        m_hat, v_hat = m / bc1, v / bc2
        denom = D(v_hat).sqrt() + D(eps_)
        update = -D(lr_) * D(m_hat) / denom
        w = w + update
        rows.append(dict(t=t, g=float(g[t - 1]), m=float(m), v=float(v), bc1=float(bc1), bc2=float(bc2),
                         m_hat=float(m_hat), v_hat=float(v_hat), denom=float(denom), update=float(update), w=float(w)))
    return rows


def adam_pytorch(grads, dtype_name, foreach=False):
    import torch
    dtype = getattr(torch, dtype_name)
    w = torch.nn.Parameter(torch.tensor(W0, dtype=dtype))
    opt = torch.optim.Adam([w], lr=LR, betas=(B1, B2), eps=EPS, foreach=foreach)
    rows = []
    for g in grads:
        opt.zero_grad()
        loss = w * torch.tensor(g, dtype=dtype)       # d loss / d w = g, delivered by autograd
        loss.backward()
        assert float(w.grad) == float(torch.tensor(g, dtype=dtype))
        before = w.detach().clone()
        opt.step()
        st = opt.state[w]
        t = int(st["step"])
        m, v = float(st["exp_avg"]), float(st["exp_avg_sq"])
        bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
        m_hat, v_hat = m / bc1, v / bc2
        rows.append(dict(t=t, g=g, m=m, v=v, bc1=bc1, bc2=bc2, m_hat=m_hat, v_hat=v_hat,
                         denom=math.sqrt(v_hat) + EPS, update=float(w.detach() - before), w=float(w)))
    return rows, torch.__version__


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def experiment_1():
    hand = adam_by_hand(GRADS)
    exact = adam_closed_form(GRADS)
    t64, torch_version = adam_pytorch(GRADS, "float64")
    t64f, _ = adam_pytorch(GRADS, "float64", foreach=True)
    t32, _ = adam_pytorch(GRADS, "float32")
    refs = {"closed_form_exact": exact, "torch_float64": t64, "torch_float64_foreach": t64f, "torch_float32": t32}
    comp, max_err = [], {}
    for name, rows in refs.items():
        max_err[name] = {}
        for q in QUANTITIES:
            errs = [abs(h[q] - r[q]) for h, r in zip(hand, rows)]
            max_err[name][q] = max(errs)
    for h, e, a, b in zip(hand, exact, t64, t32):
        for q in QUANTITIES:
            comp.append({"t": h["t"], "quantity": q, "by_hand": h[q], "closed_form_exact": e[q], "torch_float64": a[q],
                         "torch_float32": b[q], "abs_err_vs_exact": abs(h[q] - e[q]),
                         "abs_err_vs_torch64": abs(h[q] - a[q]), "abs_err_vs_torch32": abs(h[q] - b[q])})
    # tolerances: float64 paths agree to ~1e-15; float32 to its own precision (w is ~1, so ulp ~1.2e-7)
    tol = {"closed_form_exact": 1e-13, "torch_float64": 1e-13, "torch_float64_foreach": 1e-13, "torch_float32": 5e-7}
    for name, errs in max_err.items():
        for q, e in errs.items():
            scale = max(abs(r[q]) for r in hand)
            assert e <= tol[name] * max(1.0, scale), f"{name} disagrees on {q}: {e:.3e}"
    # the lesson's table, at the precision it was printed
    course_check = []
    for row, h in zip(COURSE_TABLE, hand):
        t, g, m, v, mh, vh, step, w = row
        for q, val in zip(["m", "v", "m_hat", "v_hat", "update", "w"], [m, v, mh, vh, step, w]):
            d = COURSE_DECIMALS[q]
            ok = abs(round(h[q], d) - val) < 10 ** -(d + 3)
            course_check.append({"t": t, "quantity": q, "course_printed": val, "ours_rounded": round(h[q], d), "match": ok})
    assert all(c["match"] for c in course_check), [c for c in course_check if not c["match"]]
    write_csv(os.path.join(RES, "adam_by_hand.csv"), hand)
    write_csv(os.path.join(RES, "adam_vs_pytorch.csv"), comp)
    summary = {"setup": {"w0": W0, "lr": LR, "beta1": B1, "beta2": B2, "eps": EPS, "grads": GRADS,
                         "source": "Session 11 lesson, Section 6 worked example"},
               "torch_version": torch_version, "tolerances": tol, "max_abs_err_by_hand_vs": max_err,
               "course_table_matches_at_printed_precision": all(c["match"] for c in course_check),
               "course_table_check": course_check}
    with open(os.path.join(RES, "adam_by_hand.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return hand, max_err


def first_stop(rel, tol=TOL, window=WINDOW):
    for i in range(len(rel) - window + 1):
        if all(r < tol for r in rel[i:i + window]):
            return i + 1  # 1-based step
    return None


def experiment_2():
    rng = random.Random(0)
    seqs = {
        "course_pattern": [GRADS[i % 5] for i in range(N_LONG)],          # the five gradients, repeated
        "zero_mean_noise": [rng.gauss(0.0, 0.5) for _ in range(N_LONG)],  # Section 9's "noisy, averaging to zero"
    }
    betas2 = {"0.999 (course / PyTorch default)": 0.999, "0.99 (our GPT runs)": 0.99}
    summary, rows20, curves = {"criterion": {"definition": "first t* with |u_noBC - u_BC|/|u_BC| < tol for `window` consecutive steps",
                                             "tol": TOL, "window": WINDOW, "fixed_before_results": True}, "cases": []}, [], {}
    for sname, seq in seqs.items():
        for bname, b2 in betas2.items():
            on = adam_by_hand(seq, b2=b2, bias_correction=True)
            off = adam_by_hand(seq, b2=b2, bias_correction=False)
            # same starting point, same gradients: the m and v trajectories must be identical
            assert all(a["m"] == b["m"] and a["v"] == b["v"] for a, b in zip(on, off))
            rel = [abs(b["update"] - a["update"]) / abs(a["update"]) for a, b in zip(on, off)]
            analytic = [(1 - B1 ** t) / math.sqrt(1 - b2 ** t) for t in range(1, N_LONG + 1)]
            ratio = [b["update"] / a["update"] for a, b in zip(on, off)]
            worst_vs_analytic = max(abs(r - q) / q for r, q in zip(ratio, analytic))
            case = {"sequence": sname, "beta2": b2, "beta2_label": bname,
                    "step1_ratio_noBC_over_BC": ratio[0],
                    "max_ratio_first20": max(ratio[:N_PLOT]), "argmax_ratio_first20": 1 + ratio[:N_PLOT].index(max(ratio[:N_PLOT])),
                    "ratio_at_step20": ratio[N_PLOT - 1],
                    "stopped_mattering_within_20_steps": first_stop(rel[:N_PLOT]) is not None,
                    "t_star": first_stop(rel),
                    "t_star_sensitivity": {str(tol): first_stop(rel, tol) for tol in TOL_SENSITIVITY},
                    "t_star_window5": first_stop(rel, TOL, 5), "t_star_window20": first_stop(rel, TOL, 20),
                    "max_rel_dev_from_analytic_ratio": worst_vs_analytic,
                    "weight_gap_at_20": off[N_PLOT - 1]["w"] - on[N_PLOT - 1]["w"],
                    "weight_gap_at_t_star": (off[first_stop(rel) - 1]["w"] - on[first_stop(rel) - 1]["w"]) if first_stop(rel) else None,
                    "weight_gap_at_10000": off[-1]["w"] - on[-1]["w"]}
            summary["cases"].append(case)
            curves[(sname, b2)] = (on, off, ratio, analytic)
            for a, b in zip(on[:N_PLOT], off[:N_PLOT]):
                rows20.append({"sequence": sname, "beta2": b2, "t": a["t"], "g": a["g"], "m": a["m"], "v": a["v"],
                               "update_BC": a["update"], "update_noBC": b["update"],
                               "update_BC_over_lr": a["update"] / LR, "update_noBC_over_lr": b["update"] / LR,
                               "w_BC": a["w"], "w_noBC": b["w"], "rel_step_diff": abs(b["update"] - a["update"]) / abs(a["update"])})
    write_csv(os.path.join(RES, "bias_correction_20_steps.csv"), rows20)
    with open(os.path.join(RES, "bias_correction.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot_bias_correction(curves, summary)
    return summary


def plot_bias_correction(curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    t20 = list(range(1, N_PLOT + 1))
    colors = {0.999: "#1f5aa6", 0.99: "#c0392b"}
    for b2 in (0.999, 0.99):
        on, off, ratio, analytic = curves[("course_pattern", b2)]
        ax[0].plot(t20, [r["w"] for r in on[:N_PLOT]], "-o", ms=3, color=colors[b2], label=f"BC on, β2={b2}")
        ax[0].plot(t20, [r["w"] for r in off[:N_PLOT]], "--s", ms=3, color=colors[b2], mfc="white", label=f"BC off, β2={b2}")
        ax[1].plot(t20, [-r["update"] / LR for r in on[:N_PLOT]], "-o", ms=3, color=colors[b2], label=f"BC on, β2={b2}")
        ax[1].plot(t20, [-r["update"] / LR for r in off[:N_PLOT]], "--s", ms=3, color=colors[b2], mfc="white", label=f"BC off, β2={b2}")
        tt = list(range(1, N_LONG + 1))
        ax[2].plot(tt, ratio, color=colors[b2], lw=1.6, label=f"β2={b2}")
        noisy = curves[("zero_mean_noise", b2)][2]
        ax[2].plot(tt, noisy, ":", color="k", lw=0.8)
        ts = next(c["t_star"] for c in summary["cases"] if c["beta2"] == b2 and c["sequence"] == "course_pattern")
        ax[2].axvline(ts, color=colors[b2], lw=0.8, ls="--")
        ax[2].annotate(f"t* = {ts}", (ts, 1.6 if b2 == 0.999 else 2.4), color=colors[b2], fontsize=8, ha="right",
                       xytext=(-3, 0), textcoords="offset points")
    ax[0].set(title="weight, first 20 steps\n(course gradients, repeated)", xlabel="step t", ylabel="w")
    ax[1].set(title="size of each step, in units of lr\n(BC on: both β2 lie on 1.0)", xlabel="step t", ylabel="|Δw| / lr", ylim=(0, None))
    ax[1].axhline(1, color="grey", lw=0.6)
    ax[2].axhspan(1 - TOL, 1 + TOL, color="grey", alpha=0.2, lw=0, label=f"±{TOL:.0%} band")
    ax[2].set(xscale="log", title="step without BC ÷ step with BC\n(dotted black: zero-mean noisy gradients)", xlabel="step t (log)",
              ylabel="ratio", ylim=(0.8, 7))
    ax[2].axvspan(1, N_PLOT, color="#f3d27a", alpha=0.25, lw=0)
    ax[2].text(1.3, 6.4, "first 20 steps", fontsize=8)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].legend(fontsize=7, frameon=False, ncol=2)
    ax[2].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bias_correction.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    hand, max_err = experiment_1()
    print(f"{'t':>2} {'g':>5} {'m':>9} {'v':>10} {'m_hat':>8} {'v_hat':>8} {'update':>11} {'w':>9}")
    for r in hand:
        print(f"{r['t']:>2} {r['g']:>5.2f} {r['m']:>9.6f} {r['v']:>10.7f} {r['m_hat']:>8.5f} {r['v_hat']:>8.5f} "
              f"{r['update']:>11.8f} {r['w']:>9.6f}")
    for name, errs in max_err.items():
        print(f"max |by_hand - {name}| over all quantities: {max(errs.values()):.2e}  (update {errs['update']:.2e}, w {errs['w']:.2e})")
    s = experiment_2()
    for c in s["cases"]:
        print(f"{c['sequence']:>16} beta2={c['beta2']}: step-1 ratio {c['step1_ratio_noBC_over_BC']:.4f}, "
              f"max in 20 steps {c['max_ratio_first20']:.3f} (t={c['argmax_ratio_first20']}), at 20 {c['ratio_at_step20']:.3f}; "
              f"t*={c['t_star']} sens={c['t_star_sensitivity']} w5={c['t_star_window5']} w20={c['t_star_window20']}; "
              f"gap@20={c['weight_gap_at_20']:.5f} gap@10k={c['weight_gap_at_10000']:.4f}")
