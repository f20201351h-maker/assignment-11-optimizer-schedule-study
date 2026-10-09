import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})


def load_group(group):
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "runs", group, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        for seg in [r["main"], *r.get("branches", {}).values()]:
            seg["evals"] = {int(k): v for k, v in seg["evals"].items()}
        runs.append(r)
    return runs


def save_json(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def train_window_mean(log, step, n=10):
    """Mean minibatch training loss over the n steps ending at `step` (smoother than one batch)."""
    xs = [r["loss"] for r in log if step - n < r["step"] <= step]
    assert len(xs) == n, (step, len(xs))
    return sum(xs) / n


def lr_at(log, step):
    return next(r["lr"] for r in log if r["step"] == step)
