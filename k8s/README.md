# SmartPark on GKE

Manifests for the SmartPark API + camera simulator on `smartpark-cluster`
(zonal, `australia-southeast2-a`), project `smartpark-508006`. Aligned to the
deployment guide's step numbers.

**Identity model:** pods run as KSA `smartpark/smartpark-app`. IAM roles are
granted **directly to that KSA's principal** via Workload Identity Federation for
GKE — no Google service account, no annotation, no JSON key. `firestore.AsyncClient`
and the GCS FUSE driver both pick this up as Application Default Credentials.

**Assets:** one bucket `gs://smartpark-508006-assets` is mounted read-only at
`/mnt/assets` in every pod (`implicit-dirs`), so:
- API reads `/mnt/assets/models/model.pt`
- Camera reads from `/mnt/assets/images`

## Files

| File | Kind | Notes |
|------|------|-------|
| `namespace.yaml` | Namespace | `smartpark` |
| `service-account.yaml` | ServiceAccount | `smartpark-app` (no annotation — direct WI) |
| `configmap.yaml` | ConfigMap | env for both services; names match the code |
| `camera-deployment.yaml` / `camera-service.yaml` | Deployment + ClusterIP | `camera-service:8081` → container `8001` |
| `api-deployment.yaml` / `api-service.yaml` | Deployment + LoadBalancer | `:80` → container `8000` |
| `setup.sh` | script | CLI equivalent of step 12 (IAM grants) |

## Steps 11 → 17

```bash
# 11) namespace + identity
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/service-account.yaml

# 12) grant the KSA principal bucket + Firestore access
#     Option A (CLI): prints the principal string and does both grants:
bash k8s/setup.sh
#     Option B (Console): run setup.sh just to print the principal, then paste it
#     as the New principal for Storage Object Viewer (bucket) and Cloud Datastore
#     User (IAM), per the guide.

# 13) config
kubectl apply -f k8s/configmap.yaml

# 16) camera FIRST (the API depends on it)
kubectl apply -f k8s/camera-deployment.yaml
kubectl apply -f k8s/camera-service.yaml
kubectl get deployments,pods,services -n smartpark

# 17) API
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl get deployments,pods,services -n smartpark -o wide   # wait for EXTERNAL-IP
```

The API pod stays `0/1 Ready` until the YOLO model finishes loading from the
FUSE mount (readiness returns 503 until then) — that's expected; give it a minute.

## Verify Firestore is actually shared (the whole point)

```bash
IP=$(kubectl get svc smartpark-api -n smartpark -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
for u in alice bob carol; do curl -s "http://$IP/api/find-carparks?uuid=$u&n=2" >/dev/null; done
curl -s "http://$IP/api/operations/recent-uuids"   # must list all 3 UUIDs
```
Then check the Firestore console for the `carpark_status/` and `request_logs/`
collections. (You can also run `python scripts/firestore_smoke.py` locally
against the same database — needs `gcloud auth application-default login`.)

## Config note (names matter)

`configmap.yaml` uses the env-var names `app/config.py` and
`camera_service/main.py` actually read. Concurrency is `MODEL_MAX_CONCURRENCY`
(not `INFERENCE_CONCURRENCY`). The Firestore project is supplied via
`GOOGLE_CLOUD_PROJECT` (picked up by ADC).

**Caching (§4.3).** `CACHE_TTL_SECONDS` is in the ConfigMap. It is set to `"0"`
so the Locust 1/2/4/8 replica table measures raw inference, not cache hits.
After those runs, set it to `"5"` (or similar), `kubectl apply -f k8s/configmap.yaml`,
and restart the API pods to quantify the per-user cache. Hit
`/api/operations/cache-stats` to read hits / misses / hit rate.

**Large n (§4.3).** `find-carparks` queries `min(2*n, NUM_CARPARKS)` and still
returns 200 with ranked results. If `2*n` exceeds the catalogue (max 99), the
sample is capped at every configured park — it does not 400/422.

## Troubleshooting

```bash
kubectl describe pod <pod> -n smartpark      # events (mount / scheduling)
kubectl logs <pod> -n smartpark              # app logs (JSON)
kubectl logs <pod> -n smartpark -c gke-gcsfuse-sidecar -n smartpark   # FUSE mount
```
- `PermissionDenied` on Firestore/GCS → IAM not propagated yet, or the principal
  string (namespace/SA name) doesn't match `smartpark/smartpark-app`.
- Volume won't mount → missing `gke-gcsfuse/volumes: "true"` annotation, or the
  cluster lacks the GCS FUSE CSI driver / the node pool isn't on `GKE_METADATA`
  (see the commented block at the bottom of `setup.sh`).
- Camera `degraded` / no images → `IMAGES_DIR` wrong or bucket has no objects
  under `images/`.
