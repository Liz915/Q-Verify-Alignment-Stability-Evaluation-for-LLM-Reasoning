#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_qverify}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p results/analysis data/task_manifests

QVERIFY_DEVICE="${QVERIFY_DEVICE:-cpu}"
TASK_ROOT="${TASK_ROOT:-benchmarks/tasks}"
TARGET_MIN_TASKS="${TARGET_MIN_TASKS:-20}"
MANIFEST_PATH="${MANIFEST_PATH:-data/task_manifests/test.jsonl}"
SEEDS="${SEEDS:-42,43,44,45,46}"
N_TASKS="${N_TASKS:-300}"
BEST_OF_N="${BEST_OF_N:-2}"
MECH_LAYERS="${MECH_LAYERS:-4,8,12,16,20}"
MECH_NOISE_TYPES="${MECH_NOISE_TYPES:-gaussian,uniform,signflip,dropout}"
MECH_NOISE_LEVELS="${MECH_NOISE_LEVELS:-0.1,0.2,0.5}"
MECH_RESUME="${MECH_RESUME:-1}"
MECH_LOG_FILE="${MECH_LOG_FILE:-results/analysis/mechanism_progress.log}"
MECH_EXTRA_ARGS=()
if [[ "${MECH_RESUME}" == "1" ]]; then
  MECH_EXTRA_ARGS+=(--resume)
fi

echo "=========================================="
echo "Q-Verify Paper Suite"
date
echo "=========================================="
echo "Device: $QVERIFY_DEVICE"
echo "Task root: $TASK_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "Seeds: $SEEDS"

echo ""
echo "[1/4] Build task manifests"
python experiments/build_task_manifest.py \
  --task_root "$TASK_ROOT" \
  --output_dir data/task_manifests \
  --target_min_tasks "$TARGET_MIN_TASKS"

echo ""
echo "[2/4] Stability stats (multi-seed + CI + significance + fair compute control)"
python experiments/evaluate_stability_stats.py \
  --manifest_path "$MANIFEST_PATH" \
  --n_tasks "$N_TASKS" \
  --seeds "$SEEDS" \
  --device "$QVERIFY_DEVICE" \
  --best_of_n "$BEST_OF_N" \
  --include_bestofn \
  --include_base_reflexion

echo ""
echo "[3/4] Mechanism checks (layer/noise-type scans)"
python experiments/evaluate_mechanism.py \
  --manifest_path "$MANIFEST_PATH" \
  --n_tasks "$N_TASKS" \
  --seeds "$SEEDS" \
  --layers "$MECH_LAYERS" \
  --noise_types "$MECH_NOISE_TYPES" \
  --noise_levels "$MECH_NOISE_LEVELS" \
  --device "$QVERIFY_DEVICE" \
  --log_file "$MECH_LOG_FILE" \
  "${MECH_EXTRA_ARGS[@]}"

echo ""
echo "[4/4] Legacy compatibility plots (recovery/failure modes)"
python experiments/evaluate_recovery.py --episodes 50 --noise 0.2 --device "$QVERIFY_DEVICE" --manifest_path "$MANIFEST_PATH"
python experiments/evaluate_failure_modes.py --n_samples 50 --device "$QVERIFY_DEVICE" --manifest_path "$MANIFEST_PATH"

echo ""
echo "=========================================="
echo "Paper suite complete."
date
echo "Key outputs:"
echo "  data/task_manifests/*.jsonl"
echo "  results/analysis/stability_stats_*"
echo "  results/analysis/mechanism_*"
echo "  results/analysis/recovery_*"
echo "  results/analysis/failure_modes*"
echo "=========================================="
