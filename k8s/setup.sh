#!/usr/bin/env bash
# IAM for SmartPark using Workload Identity Federation for GKE (no Google service
# account, no JSON key). This is the CLI equivalent of guide step 12 — run it, OR
# do the same two grants in the Console. Either way, the pods' KSA principal gets:
#   - Storage Object Viewer  on the assets bucket   (read model + images)
#   - Cloud Datastore User   on the project         (Firestore read/write)
#
# Prereqs: the namespace + service account already applied (steps 11), and the
# cluster already has Workload Identity + the GCS FUSE CSI driver enabled.
set -euo pipefail

PROJECT_ID="smartpark-508006"
ASSETS_BUCKET="smartpark-508006-assets"
NAMESPACE="smartpark"
KSA="smartpark-app"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

# The identity string GCP uses for this Kubernetes service account.
PRINCIPAL="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"

echo "==> KSA principal (paste this as the 'New principal' in the Console if doing step 12 by hand):"
echo "$PRINCIPAL"
echo

echo "==> Granting Storage Object Viewer on gs://${ASSETS_BUCKET}"
gcloud storage buckets add-iam-policy-binding "gs://${ASSETS_BUCKET}" \
  --member="$PRINCIPAL" --role="roles/storage.objectViewer"

echo "==> Granting Cloud Datastore User (Firestore) on ${PROJECT_ID}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="$PRINCIPAL" --role="roles/datastore.user"

# If you export traces to Cloud Trace, also grant the Trace Agent role:
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member="$PRINCIPAL" --role="roles/cloudtrace.agent"

echo
echo "Done. IAM can take a minute or two to propagate before pods can read GCP."

# ─────────────────────────────────────────────────────────────────────────
# ONLY IF the cluster doesn't already have these enabled (they usually do by
# now). These MUTATE the cluster and updating node pools RECREATES nodes:
#
# gcloud container clusters update smartpark-cluster --location australia-southeast2-a \
#   --workload-pool="${PROJECT_ID}.svc.id.goog"
# gcloud container clusters update smartpark-cluster --location australia-southeast2-a \
#   --update-addons GcsFuseCsiDriver=ENABLED
# for POOL in $(gcloud container node-pools list --cluster smartpark-cluster \
#     --location australia-southeast2-a --format='value(name)'); do
#   gcloud container node-pools update "$POOL" --cluster smartpark-cluster \
#     --location australia-southeast2-a --workload-metadata=GKE_METADATA
# done
