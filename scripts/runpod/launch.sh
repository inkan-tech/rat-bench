#!/usr/bin/env bash
# Run one RAT-Bench command on a RunPod GPU pod with the vllm backend.
#
# The repo (with benchmark/ and data/pums/) goes up with rsync, the command runs
# in a uv venv with RAT_BACKEND=runpod, benchmark/ comes back, and the pod is
# deleted. The pod is also deleted on failure, on Ctrl-C, and when the cost cap
# is reached: locally by this script, and on the pod by a watchdog, so a closed
# laptop does not leave a pod billing.
#
# Usage:
#   RUNPOD_API_KEY=... scripts/runpod/launch.sh --confirm [options] -- <command>
# Example:
#   scripts/runpod/launch.sh --confirm -- bash scripts/anonymize/level_1.sh
#
# Options:
#   --max-cost USD  cost cap for the whole run, pod creation included (default 5)
#   --cloud TYPE    SECURE or COMMUNITY (default SECURE: public IP for SSH)
#   --confirm       required to create a pod; without it the plan is printed only
#
# Secrets:
#   RUNPOD_API_KEY  Proton Pass, vault Operations, item "Runpod", field "API Key".
#                   Never in the repo. It is copied to the pod (mode 600) only so
#                   the watchdog can delete the pod.
#   HF_TOKEN        or ~/.cache/huggingface/token; the meta-llama model is gated.
#   pii_benchmark/credentials.py goes up with the repo if it exists (API attackers).
set -euo pipefail

API=https://rest.runpod.io/v1
# torch 2.9.1 wheels need a CUDA >= 12.8 host driver.
IMAGE=runpod/pytorch:1.3.3-cu1290-torch291-ubuntu2204
CUDA_VERSIONS='["12.8","12.9","13.0"]'
# 24GB is enough for Llama-3.1-8B in bf16 with an 8192 context.
GPUS='["NVIDIA GeForce RTX 4090","NVIDIA RTX A5000","NVIDIA GeForce RTX 3090"]'
REPO=$(cd "$(dirname "$0")/../.." && pwd)
REMOTE=/root/rat-bench

MAX_COST=5
CLOUD=SECURE
CONFIRM=0
while [ $# -gt 0 ]; do
  case $1 in
    --confirm) CONFIRM=1 ;;
    --max-cost) MAX_COST=$2; shift ;;
    --cloud) CLOUD=$2; shift ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[ $# -gt 0 ] || { echo "Missing command after --" >&2; exit 2; }
RUN_CMD=$(printf '%q ' "$@")
NAME="rat-bench-$(date -u +%Y%m%d%H%M%S)"

[ -n "${RUNPOD_API_KEY:-}" ] || {
  echo "RUNPOD_API_KEY is not set (Proton Pass: vault Operations, item Runpod, field 'API Key')." >&2
  exit 2
}
HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
[ -n "$HF_TOKEN" ] || { echo "No Hugging Face token: set HF_TOKEN or run 'hf auth login'." >&2; exit 2; }
[ -d "$REPO/benchmark" ] || { echo "Missing $REPO/benchmark (see README)." >&2; exit 2; }
[ -f "$REPO/data/pums/pums_pwgtp_sample_.csv" ] ||
  echo "Warning: $REPO/data/pums/pums_pwgtp_sample_.csv is missing." >&2

echo "Run:      $NAME"
echo "Command:  $RUN_CMD"
echo "GPU:      $GPUS ($CLOUD)"
echo "Cost cap: \$$MAX_COST (the pod is deleted when it is reached)"
if [ "$CONFIRM" -ne 1 ]; then
  echo "Nothing created. Add --confirm to create the pod (real spend)."
  exit 0
fi

api() {
  curl -sS --fail-with-body -X "$1" "$API$2" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    ${3:+-d "$3"}
}

WORK=$(mktemp -d)
ssh-keygen -q -t ed25519 -N "" -C "$NAME" -f "$WORK/key"
POD_ID=""
IP=""
PORT=""
SSH_READY=0

ssh_pod() {
  ssh -i "$WORK/key" -p "$PORT" -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$WORK/known_hosts" -o ConnectTimeout=15 \
    -o ServerAliveInterval=30 "root@$IP" "$@"
}

fetch_results() {
  echo "Fetching benchmark/ and the job log"
  rsync -az -e "ssh -i $WORK/key -p $PORT -o UserKnownHostsFile=$WORK/known_hosts" \
    "root@$IP:$REMOTE/benchmark/" "$REPO/benchmark/" || echo "rsync of benchmark/ failed" >&2
  ssh_pod "cat /root/job.log" > "$REPO/benchmark/runpod_$NAME.log" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$POD_ID" ]; then
    [ "$SSH_READY" -eq 1 ] && fetch_results
    if api DELETE "/pods/$POD_ID" >/dev/null; then
      echo "Pod $POD_ID deleted"
    else
      echo "FAILED to delete pod $POD_ID. Delete it at https://console.runpod.io/pods" >&2
    fi
  fi
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

BODY=$(jq -n --arg name "$NAME" --arg image "$IMAGE" --arg cloud "$CLOUD" \
  --arg pub "$(cat "$WORK/key.pub")" --argjson gpus "$GPUS" --argjson cuda "$CUDA_VERSIONS" \
  '{name: $name, imageName: $image, gpuTypeIds: $gpus, gpuTypePriority: "custom",
    gpuCount: 1, cloudType: $cloud, containerDiskInGb: 80, volumeInGb: 0,
    ports: ["22/tcp"], supportPublicIp: true, allowedCudaVersions: $cuda,
    env: {PUBLIC_KEY: $pub}}')

START=$(date +%s)
for attempt in 1 2 3; do
  if POD=$(api POST /pods "$BODY" 2>&1); then
    POD_ID=$(jq -r .id <<<"$POD")
    break
  fi
  echo "Pod creation failed (attempt $attempt/3): $POD" >&2
  case $POD in
    *"does not have the resources"*|*"Something went wrong"*|*"no longer any instances"*) sleep 15 ;;
    *) exit 1 ;;
  esac
done
[ -n "$POD_ID" ] || { echo "No GPU could be placed." >&2; exit 1; }

COST_PER_HR=$(jq -r '.costPerHr // 0' <<<"$POD")
BUDGET_S=$(awk -v c="$MAX_COST" -v r="$COST_PER_HR" 'BEGIN { if (r > 0) printf "%d", c / r * 3600; else print 0 }')
[ "$BUDGET_S" -gt 0 ] || { echo "Unknown pod price ($COST_PER_HR/h)." >&2; exit 1; }
DEADLINE=$((START + BUDGET_S))
echo "Pod $POD_ID: \$$COST_PER_HR/h, deleted after $((BUDGET_S / 60)) min at the latest"

echo "Waiting for SSH"
for _ in $(seq 1 90); do
  INFO=$(api GET "/pods/$POD_ID" || echo '{}')
  IP=$(jq -r '.publicIp // empty' <<<"$INFO")
  PORT=$(jq -r '.portMappings["22"] // empty' <<<"$INFO")
  [ -n "$IP" ] && [ -n "$PORT" ] && ssh_pod true 2>/dev/null && { SSH_READY=1; break; }
  sleep 10
done
[ "$SSH_READY" -eq 1 ] || { echo "SSH did not come up in 15 min." >&2; exit 1; }

printf '%s' "$RUNPOD_API_KEY" | ssh_pod 'umask 077; cat > /root/.runpod_key'
printf '%s' "$HF_TOKEN" | ssh_pod 'umask 077; mkdir -p ~/.cache/huggingface; cat > ~/.cache/huggingface/token'
ssh_pod 'cat > /root/watchdog.sh' <<EOF
sleep $((DEADLINE - $(date +%s) - 30))
curl -s -X DELETE "$API/pods/$POD_ID" -H "Authorization: Bearer \$(cat /root/.runpod_key)"
EOF
ssh_pod 'nohup bash /root/watchdog.sh </dev/null >/dev/null 2>&1 &'

echo "Uploading the repo"
ssh_pod 'apt-get update -qq && apt-get install -y -qq rsync >/dev/null'
rsync -az --exclude .venv --exclude .git --exclude .claude --exclude __pycache__ \
  -e "ssh -i $WORK/key -p $PORT -o UserKnownHostsFile=$WORK/known_hosts" \
  "$REPO/" "root@$IP:$REMOTE/"

ssh_pod 'cat > /root/job.sh' <<EOF
cd $REMOTE
export RAT_BACKEND=runpod VLLM_GPU_MEMORY_UTILIZATION=0.9
pip install -q uv
uv sync
source .venv/bin/activate
$RUN_CMD
EOF
ssh_pod "nohup bash -c 'bash -e /root/job.sh > /root/job.log 2>&1; echo \$? > /root/job.exit' </dev/null >/dev/null 2>&1 &"
echo "Job started. Log on the pod: /root/job.log"

FAILS=0
while :; do
  sleep 60
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "Cost cap reached: stopping." >&2
    exit 1
  fi
  if ! STATUS=$(ssh_pod 'cat /root/job.exit 2>/dev/null; tail -n 1 /root/job.log 2>/dev/null' 2>/dev/null); then
    FAILS=$((FAILS + 1))
    [ "$FAILS" -lt 5 ] || { echo "Pod unreachable 5 times in a row." >&2; SSH_READY=0; exit 1; }
    continue
  fi
  FAILS=0
  EXIT_CODE=$(head -n 1 <<<"$STATUS")
  if [[ $EXIT_CODE =~ ^[0-9]+$ ]]; then
    echo "Job finished with exit code $EXIT_CODE"
    exit "$EXIT_CODE"
  fi
  echo "[$(( ($(date +%s) - START) / 60 )) min] ${STATUS:0:200}"
done
