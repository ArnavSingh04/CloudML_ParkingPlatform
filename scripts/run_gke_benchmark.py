"""Orchestrate Locust against GKE at 1, 2, 4 and 8 API replicas.

Usage (from the project root, with locust on PATH or in .venv):
    python scripts/run_gke_benchmark.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

NS = "smartpark"
DEPLOY = "smartpark-api"
HOST = "http://34.129.33.23"
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _locust_bin() -> Path | None:
    found = shutil.which("locust")
    if found:
        return Path(found)
    for candidate in (
        ROOT / ".venv" / "Scripts" / "locust.exe",
        ROOT / ".venv" / "bin" / "locust",
    ):
        if candidate.exists():
            return candidate
    return None


LOCUST = _locust_bin()


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, cwd=ROOT)


def kubectl_json(args: list[str]):
    out = subprocess.check_output(["kubectl", *args, "-o", "json"], cwd=ROOT)
    return json.loads(out)


def scale_and_wait(replicas: int, timeout_s: int = 420) -> int:
    run(["kubectl", "scale", f"deploy/{DEPLOY}", "-n", NS, f"--replicas={replicas}"])
    deadline = time.time() + timeout_s
    ready = 0
    while time.time() < deadline:
        dep = kubectl_json(["get", "deploy", DEPLOY, "-n", NS])
        ready = int(dep.get("status", {}).get("readyReplicas") or 0)
        avail = int(dep.get("status", {}).get("availableReplicas") or 0)
        print(f"  readyReplicas={ready} availableReplicas={avail} want={replicas}", flush=True)
        if ready >= replicas:
            time.sleep(15)  # extra settle after model load
            return ready
        time.sleep(8)
    print(f"WARNING: timed out with {ready}/{replicas} ready", flush=True)
    return ready


def run_locust(label: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    csv_prefix = REPORTS / label
    cmd = [
        str(LOCUST),
        "-f",
        str(ROOT / "locustfile_bench.py"),
        "--host",
        HOST,
        "--headless",
        "--csv",
        str(csv_prefix),
        "--html",
        str(REPORTS / f"{label}.html"),
        "--csv-full-history",
        "--only-summary",
    ]
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode not in (0, 1):
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def summarise(label: str) -> dict:
    stats = REPORTS / f"{label}_stats.csv"
    rows = list(csv.DictReader(stats.open(encoding="utf-8")))
    aggregated = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1] if rows else {})
    return {
        "label": label,
        "requests": aggregated.get("Request Count"),
        "failures": aggregated.get("Failure Count"),
        "avg_ms": aggregated.get("Average Response Time"),
        "p50_ms": aggregated.get("Median Response Time"),
        "p95_ms": aggregated.get("95%"),
        "rps": aggregated.get("Requests/s"),
    }


def main() -> int:
    if LOCUST is None:
        print("locust not installed — pip install locust==2.32.6", file=sys.stderr)
        return 1
    summaries = []
    for n in (1, 2, 4, 8):
        print(f"\n===== {n} replica(s) =====", flush=True)
        ready = scale_and_wait(n)
        label = f"pods-{n}"
        t0 = time.time()
        run_locust(label)
        elapsed = round(time.time() - t0, 1)
        s = summarise(label)
        s["ready_pods"] = ready
        s["wanted_pods"] = n
        s["locust_seconds"] = elapsed
        summaries.append(s)
        print("summary:", s, flush=True)
        (REPORTS / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\nALL DONE", json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
