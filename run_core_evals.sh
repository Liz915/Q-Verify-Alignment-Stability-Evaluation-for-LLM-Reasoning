#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_qverify}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p results/analysis

STABILITY_SAMPLES="${STABILITY_SAMPLES:-50}"
RECOVERY_EPISODES="${RECOVERY_EPISODES:-30}"
RECOVERY_NOISE="${RECOVERY_NOISE:-0.2}"
FAILURE_SAMPLES="${FAILURE_SAMPLES:-50}"
QVERIFY_DEVICE="${QVERIFY_DEVICE:-cpu}"
TASK_ROOT="${TASK_ROOT:-benchmarks/tasks}"
TARGET_MIN_TASKS="${TARGET_MIN_TASKS:-20}"
MANIFEST_PATH="${MANIFEST_PATH:-data/task_manifests/test.jsonl}"

echo "=========================================="
echo "Q-Verify Core Evaluation Pipeline"
date
echo "=========================================="

echo ""
echo "[0/4] Build task manifests"
python experiments/build_task_manifest.py --task_root "$TASK_ROOT" --output_dir data/task_manifests --target_min_tasks "$TARGET_MIN_TASKS"

echo ""
echo "[1/4] Stability Radius (Base / DPO / DPO+Reflexion)"
python experiments/evaluate_stability.py --n_samples "$STABILITY_SAMPLES" --device "$QVERIFY_DEVICE" --manifest_path "$MANIFEST_PATH"

echo ""
echo "[2/4] Inference-Time Recovery (includes Base+Reflexion control)"
python experiments/evaluate_recovery.py --episodes "$RECOVERY_EPISODES" --noise "$RECOVERY_NOISE" --device "$QVERIFY_DEVICE" --manifest_path "$MANIFEST_PATH"

echo ""
echo "[3/4] Failure Mode Decomposition"
python experiments/evaluate_failure_modes.py --n_samples "$FAILURE_SAMPLES" --device "$QVERIFY_DEVICE" --manifest_path "$MANIFEST_PATH"

echo ""
echo "=========================================="
echo "Pipeline complete."
date
echo "Outputs: results/analysis/*.csv, *.png, *.pdf"
echo "Samples: stability=$STABILITY_SAMPLES recovery_episodes=$RECOVERY_EPISODES failure=$FAILURE_SAMPLES"
echo "Device: $QVERIFY_DEVICE"
echo "Task root: $TASK_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "=========================================="
