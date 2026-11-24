#!/bin/bash
# Test-split evaluation of one or more Skills (40 objects x 3 sessions, no Skill Evolver).
# usage: EVO_DATASET_ROOT=<subset folder> scripts/experiments/run_test.sh <base port> <skill dir>...
#   e.g. scripts/experiments/run_test.sh 9876 skills/paper/S0_seed skills/paper/S9
# Set EXEC_MODEL to evaluate under another Executor model.
PORT=$1; shift
cd "$(dirname "$0")/../.."
: "${EVO_DATASET_ROOT:?set EVO_DATASET_ROOT to the evaluation subset folder}"
export EVO_ORACLE_RENDER_SIMS=4 EVO_SKILL_PREFLIGHT_TIMEOUT_S=600
for V in "$@"; do
  echo "===== TEST EVAL $V $(date)"
  python -m evo.engine.loop --config evo/configs/main.yaml \
    --skill-dir "$V" --split test --no-generator --max-rounds 1 --executor-reps "${REPS:-3}" \
    --parallel "${PAR:-14}" --sim-base-port "$PORT" --headless --oracle-mode objective \
    --executor-effort low --generator-effort high ${EXEC_MODEL:+--executor-model $EXEC_MODEL}
done
