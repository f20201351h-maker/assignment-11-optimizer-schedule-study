"""
Trace every number in README.md to a value in the result summaries (results/*.json, results/*.csv).

A README number matches when some artifact value, or its negation, rounds to it at the precision the README printed
it with (relative tolerance of half a unit in the last printed digit). Numbers that are configuration (grid values,
step counts, widths) are matched against the stored configs as well. Anything left over is printed for manual review.
    python audit/trace_readme_numbers.py  -> results/readme_number_trace.json
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flatten(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(k, out)
            flatten(v, out)
    elif isinstance(x, list):
        for v in x:
            flatten(v, out)
    elif isinstance(x, bool):
        return
    elif isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, str):
        try:
            out.append(float(x))
        except ValueError:
            for m in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", x):
                out.append(float(m))


vals = []
for p in glob.glob(os.path.join(ROOT, "results", "*.json")):
    flatten(json.load(open(p)), vals)
for p in glob.glob(os.path.join(ROOT, "results", "*.csv")):
    for row in csv.reader(open(p)):
        flatten(row, vals)
# run configs (grids, steps, seeds) and one sample run's metadata
for p in glob.glob(os.path.join(ROOT, "results", "runs", "*", "*.json")):
    r = json.load(open(p))
    flatten(r["config"], vals)
    flatten(r["meta"]["n_params"], vals)
vals = sorted(set(abs(v) for v in vals))

text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
text = re.sub(r"```.*?```", "", text, flags=re.S)          # code blocks
text = re.sub(r"`[^`]*`", "", text)                         # inline code (paths, names)
text = re.sub(r"\]\([^)]*\)", "]", text)                    # link targets
tokens = re.findall(r"(?<![\w.])(\d[\d,]*\.?\d*(?:e-?\d+)?)", text)

import bisect


def matches(tok):
    s = tok.replace(",", "")
    x = float(s)
    if "e" in s:
        mant = s.split("e")[0]
        dec = len(mant.split(".")[1]) if "." in mant else 0
        tol = 0.5 * 10 ** (-dec) * 10 ** int(s.split("e")[1])
    else:
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec)
    lo, hi = x - tol * 1.0001, x + tol * 1.0001
    i = bisect.bisect_left(vals, lo)
    return i < len(vals) and vals[i] <= hi


res = {"matched": [], "unmatched": []}
for t in dict.fromkeys(tokens):
    (res["matched"] if matches(t) else res["unmatched"]).append(t)
with open(os.path.join(ROOT, "results", "readme_number_trace.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"{len(res['matched'])} distinct numbers matched; {len(res['unmatched'])} need a manual look:")
print(res["unmatched"])
