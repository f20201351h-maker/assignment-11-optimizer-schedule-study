"""
GPU usage and cost of this study.

  * ledger: results/modal_ledger.jsonl, one line per GPU job, with the in-container wall time;
  * billing: `modal billing report --for today --show-resources --json`, filtered to this app's name, which also
    covers container start-up and the checkpoint verification call that the ledger does not.
Writes results/compute_cost.json.
"""
import json
import os
import subprocess
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = "era-v5-s11-optimizers"
A10G_PER_SEC = 0.000306  # modal billing rates: Nvidia A10G $1.10 / hour

ledger = [json.loads(l) for l in open(os.path.join(ROOT, "results", "modal_ledger.jsonl"))]
ok = [x for x in ledger if x["status"] == "ok"]
wall = sum(x["wall_seconds"] for x in ok)
out = {"jobs_ok": len(ok), "jobs_failed": len(ledger) - len(ok), "gpu_reported": dict(Counter(x["gpu"] for x in ok)),
       "jobs_by_plan": dict(Counter(x["plan"] for x in ok)), "in_container_gpu_seconds": round(wall, 1),
       "in_container_gpu_hours": round(wall / 3600, 3), "ledger_cost_usd_at_1.10_per_hour": round(wall * A10G_PER_SEC, 3),
       "max_concurrent_containers_configured": 4}
try:
    rows = []
    for period in ("yesterday", "today"):  # UTC days; the runs fall on 2026-10-01 UTC
        raw = subprocess.run(["modal", "billing", "report", "--for", period, "--show-resources", "--json"],
                             capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"}).stdout
        rows += [r for r in json.loads(raw[raw.index("["):]) if r["description"] == APP]
    by_res = Counter()
    for r in rows:
        by_res[r["resource"]] += float(r["cost"])
    out["billing_report"] = {"app_runs": len({r["object_id"] for r in rows}), "cost_by_resource_usd": {k: round(v, 4) for k, v in by_res.items()},
                             "total_usd": round(sum(by_res.values()), 4), "source": "modal billing report --show-resources --json"}
except Exception as e:  # billing can lag; the ledger number stands on its own
    out["billing_report"] = {"error": repr(e)}
with open(os.path.join(ROOT, "results", "compute_cost.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
