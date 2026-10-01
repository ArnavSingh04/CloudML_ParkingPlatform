"""Build the two assignment plots from Locust stats_history CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
USERS = (5, 10, 20, 40)
POD_FILES = {
    1: REPORTS / "pods-1_stats_history.csv",
    2: REPORTS / "pods-2_stats_history.csv",
    4: REPORTS / "pods-4_stats_history.csv",
    8: REPORTS / "pods-8_stats_history.csv",
}


def aggregated_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Type") in ("", None) and row.get("Name") in ("Aggregated", ""):
                rows.append(row)
    return rows


def stage_points(path: Path) -> dict[int, dict]:
    """Incremental average latency and RPS for each held user count."""
    rows = aggregated_rows(path)
    last_by_users: dict[int, dict] = {}
    for row in rows:
        users = int(row["User Count"])
        if users in USERS:
            last_by_users[users] = row

    prev_n = 0
    prev_sum = 0.0
    out: dict[int, dict] = {}
    for users in USERS:
        row = last_by_users.get(users)
        if row is None:
            continue
        n = int(row["Total Request Count"])
        avg = float(row["Total Average Response Time"] or 0)
        inc_n = n - prev_n
        inc_avg = ((n * avg) - prev_sum) / inc_n if inc_n else None
        out[users] = {
            "incremental_avg_ms": inc_avg,
            "rps": float(row["Requests/s"] or 0),
            "n": inc_n,
        }
        prev_n, prev_sum = n, n * avg
    return out


def main() -> None:
    series = {pods: stage_points(path) for pods, path in POD_FILES.items() if path.exists()}

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    for pods, points in series.items():
        xs = [u for u in USERS if u in points and points[u]["incremental_avg_ms"]]
        ys = [points[u]["incremental_avg_ms"] / 1000 for u in xs]
        label = f"{pods} pod" + ("s" if pods != 1 else "")
        if pods == 8:
            label = "8 desired / 6 ready"
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Concurrent Locust users")
    ax.set_ylabel("Average response time (seconds)")
    ax.set_title("Latency vs concurrent users (CORE APIs, cache off)")
    ax.set_xticks(list(USERS))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "latency_vs_users.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    for pods, points in series.items():
        xs = [u for u in USERS if u in points]
        ys = [points[u]["rps"] for u in xs]
        label = f"{pods} pod" + ("s" if pods != 1 else "")
        if pods == 8:
            label = "8 desired / 6 ready"
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Concurrent Locust users")
    ax.set_ylabel("Throughput (requests / second)")
    ax.set_title("Throughput vs concurrent users (CORE APIs, cache off)")
    ax.set_xticks(list(USERS))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "throughput_vs_users.png")
    plt.close(fig)

    print("wrote", REPORTS / "latency_vs_users.png")
    print("wrote", REPORTS / "throughput_vs_users.png")
    for pods, points in series.items():
        print(pods, {u: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()} for u, p in points.items()})


if __name__ == "__main__":
    main()
