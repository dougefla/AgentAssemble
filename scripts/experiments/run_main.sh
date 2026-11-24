#!/bin/bash
# Self-improvement run of the paper: 12 rounds from the seed Skill S0 on the train split, with the Evolver edit
# gate and the paired validation gate (3 sessions per validation object, 1.5 standard errors).
# usage: EVO_DATASET_ROOT=<subset folder> scripts/experiments/run_main.sh [seed=43] [base port=9876]
SEED=${1:-43}; PORT=${2:-9876}
cd "$(dirname "$0")/../.."
: "${EVO_DATASET_ROOT:?set EVO_DATASET_ROOT to the evaluation subset folder}"
export EVO_FEEDBACK_FORMAT=cluster EVO_ORACLE_RENDER_SIMS=4 EVO_SKILL_PREFLIGHT_TIMEOUT_S=600
CFG=$(mktemp -t main_XXXX).yaml
sed "s/^  seed: 42/  seed: $SEED/" evo/configs/main.yaml > "$CFG"
exec python -m evo.engine.loop --config "$CFG" \
  --skill-dir skills/paper/S0_seed --split train --gate-split val --gate-mode paired --gate-reps 3 --gate-z 1.5 \
  --max-rounds 12 --parallel "${PAR:-14}" --sim-base-port "$PORT" --headless --oracle-mode objective \
  --executor-effort low --generator-effort xhigh
