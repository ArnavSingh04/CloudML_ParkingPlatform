# Lab notes — GKE Locust run (16 Sep 2026)

Companion to `benchmark-report.md` / `benchmark-report.pdf`. This file is the
command and data log, not the 500-word Moodle submission.

## Cluster at the time of the run

| Item | Value |
|---|---|
| Project | `smartpark-508006` |
| Cluster | `smartpark-cluster` (zonal, `australia-southeast2-a`) |
| Created | 2026-09-09T07:48:37Z |
| Namespace | `smartpark` |
| API Service | LoadBalancer **34.129.33.23** (port 80 → 8000) |
| Camera Service | ClusterIP `camera-service:8081` |
| Node pool | `default-pool`, `e2-standard-4`, autoscaler min=1 max=3 |
| Allocatable / node | 3920 mCPU, ~13.6 Gi |
| API image | `australia-southeast2-docker.pkg.dev/smartpark-508006/smartpark-repo/smartpark-api:v2` |
| Camera image | `.../camera-service:v2` |
| Cache | `CACHE_TTL_SECONDS=0` (disabled for raw inference) |
| Locust | 2.32.6 on the laptop (not in-cluster) |
| Billing account | Education `01C867-4FAD17-B8FFE7` |

## Commands actually executed

```text
# 1. Push the missing API image (v2 had been built locally but not pushed;
#    camera set-image had previously run with empty $REG/$TAG)
docker push australia-southeast2-docker.pkg.dev/smartpark-508006/smartpark-repo/smartpark-api:v2

# 2. Point YAML at v2, apply, restart
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/camera-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl rollout restart deploy/smartpark-api -n smartpark
kubectl rollout restart deploy/smartpark-camera -n smartpark
kubectl rollout status deploy/smartpark-camera -n smartpark --timeout=180s
kubectl rollout status deploy/smartpark-api -n smartpark --timeout=300s

# 3. Smoke test (CBD ids, cache disabled)
curl -s http://34.129.33.23/api/health
curl -s "http://34.129.33.23/api/find-carparks?uuid=smoke&n=2"
curl -s http://34.129.33.23/api/operations/cache-stats
# health: model_loaded true, num_carparks 24
# find-carparks: CBD_010 / CBD_008, speed_inference 2583.3 ms, cached false
# cache-stats: enabled false, ttl_seconds 0.0

# 4. Locust (laptop). requirements-dev.txt fails on Python 3.14, so:
pip install locust==2.32.6
python scripts/run_gke_benchmark.py
# locustfile_bench.py shape: 5 users 70s, 10 users 70s, 20 users 70s, 40 users 70s
# Equivalent per replica count:
#   kubectl scale deploy/smartpark-api -n smartpark --replicas=N
#   locust -f locustfile_bench.py --host http://34.129.33.23 --headless \
#          --csv reports/pods-N --html reports/pods-N.html --csv-full-history

# 5. Scale-down (cost)
kubectl scale deploy/smartpark-api -n smartpark --replicas=1
gcloud container node-pools update default-pool --cluster=smartpark-cluster \
  --zone=australia-southeast2-a --enable-autoscaling --min-nodes=1 --max-nodes=1
# Extra nodes from the 8-pod attempt were drained so the autoscaler can delete them.
# For the 20-minute demo, raise max-nodes back to 3 before scaling the API.
```

Locust host: `http://34.129.33.23`
Locust files: `locustfile.py` (user behaviour) + `locustfile_bench.py` (stepped shape)

The first driver stopped after two pods because Locust exits 1 on any failure
(the single 502). `run_gke_benchmark.py` was then set to treat exit 1 as OK and
the 4-pod and 8-pod stages were rerun.

## Raw Locust summaries (whole 280 s shape)

From `reports/pods-*_stats.csv` Aggregated row:

| File | reqs | fails | mean ms | median ms | p95 ms | req/s | ready/desired |
|---|---:|---:|---:|---:|---:|---:|---|
| pods-1 | 88 | 0 | 35256 | 32000 | 63000 | 0.32 | 1/1 |
| pods-2 | 115 | 1 | 30582 | 26000 | 63000 | 0.41 | 2/2 |
| pods-4 | 245 | 0 | 16975 | 13000 | 38000 | 0.88 | 4/4 |
| pods-8 | 360 | 0 | 10885 | 8100 | 31000 | 1.29 | 6/8 |

Per-endpoint (from the same CSVs):

| Run | find-carparks reqs / mean s | annotate-carpark reqs / mean s |
|---|---|---|
| 1 pod | 71 / 35.8 | 17 / 32.9 |
| 2 pods | 97 / 31.6 | 18 / 25.3 (1 × 502) |
| 4 pods | 200 / 16.9 | 45 / 17.3 |
| 6/8 pods | 296 / 11.4 | 64 / 8.7 |

pods-2 single failure: `GET /api/annotate-carpark` HTTP 502
`Camera fetch failed`. Locust `--headless` returns exit 1 when any request fails.

## Incremental mean latency (s) by user plateau

Derived by `scripts/plot_benchmark.py` from `*_stats_history.csv`
(running totals converted to per-stage means):

| Users | 1 pod | 2 pods | 4 pods | 6 ready / 8 desired |
|---:|---:|---:|---:|---:|
| 5 | 12.9 | 10.4 | 6.4 | 4.1 |
| 10 | 25.1 | 35.6 | 9.3 | 5.7 |
| 20 | 42.2 | 30.0 | 16.6 | 10.9 |
| 40 | 57.9 | 34.3 | 28.6 | 18.0 |

The 2-pod / 10-user cell is noisy: that run’s max `find-carparks` was 125 s
(and one request completed in 88 ms). Treat 35.6 s as an outlier, not a
reversal of the 1 → 4 → 6 trend.

## Scheduling (why 8 desired became 6 ready)

```text
kubectl scale deploy/smartpark-api -n smartpark --replicas=8
# timed out: readyReplicas=6
kubectl get nodes
# 3 × e2-standard-4, allocatable 3920 mCPU each
```

Each API pod requests `cpu: 1000m` plus a GCS FUSE sidecar. Three nodes cannot
pack eight such pods once kube-system DaemonSets are counted. Cluster autoscaler
maxNodeCount is already 3, so no fourth node appeared.

During the 8-pod attempt the autoscaler added nodes:

| Node | Inserted (PDT, UTC-7) | Notes |
|---|---|---|
| `...-xo4z` | 2026-09-12 08:57 | long-lived node, deleted 15 Sep 21:10 |
| `...-2t5p` | 2026-09-15 20:49 | still up after the run |
| `...-jbz7` | 2026-09-15 20:55 | still up after the run |
| `...-x6k4` | 2026-09-15 21:13 | third node for the 8-pod attempt |

## Cost reconstruction (from `gcloud compute operations`)

e2-standard-4 VM-hours to ~21:30 PDT on 15 Sep 2026:

| Instance | Hours |
|---|---:|
| k4w2, 0wq6, n5tf, nn4v (cluster bring-up 9 Sep) | 5.1 |
| xsbr (9 Sep 04:41 → 12 Sep 09:00) | 76.3 |
| xo4z (12 Sep 08:57 → 15 Sep 21:10) | 84.2 |
| 2t5p + jbz7 + x6k4 (today’s extra nodes, still running at calculation time) | 1.5 |
| **Total** | **~167** |

167 × $0.1616 ≈ **$27.0** compute, before disks / LB / GKE fee. See the PDF
cost table. Confirm in the Billing console; do not leave three nodes idle.

## Artefacts

| File | What |
|---|---|
| `reports/benchmark-report.md` | ≤500-word submission body + tables + figure refs |
| `reports/benchmark-report.pdf` | Same report as a PDF (Moodle file) |
| `reports/latency_vs_users.png` | Figure 1 |
| `reports/throughput_vs_users.png` | Figure 2 |
| `reports/pods-{1,2,4,8}.html` | Locust HTML reports |
| `reports/pods-*_stats.csv` | Per-endpoint stats |
| `reports/pods-*_stats_history.csv` | 1 Hz history used for the plots |
| `reports/summaries.json` | Machine-readable whole-run totals |
| `scripts/run_gke_benchmark.py` | Scale 1/2/4/8 and invoke Locust |
| `scripts/plot_benchmark.py` | Build the two PNGs |
| `scripts/render_report_pdf.py` | Build the PDF |
| `locustfile.py` / `locustfile_bench.py` | User behaviour + stepped load shape |

Rename `reports/benchmark-report.pdf` to `firstname_lastname_studentid.pdf`
before uploading to Moodle.
