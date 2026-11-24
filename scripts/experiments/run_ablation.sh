#!/bin/bash
# Ablation runs of the paper, each twelve rounds from the seed Skill S0.
#   noreg  without the regression strategy: no Evolver edit gate, no validation gate, no rollback.
#          The Skill that the Evolver writes in round t is used in round t+1.
#   score  without diagnostics: the Evolver sees only the pass predicate and the fitness of every rollout.
#          The validation gate is unchanged.
# usage: EVO_DATASET_ROOT=<subset folder> scripts/experiments/run_ablation.sh <noreg|score> [base port=9876]
KIND=$1; PORT=${2:-9876}
cd "$(dirname "$0")/../.."
: "${EVO_DATASET_ROOT:?set EVO_DATASET_ROOT to the evaluation subset folder}"
export EVO_ORACLE_RENDER_SIMS=4 EVO_SKILL_PREFLIGHT_TIMEOUT_S=600
CFG=$(mktemp -t abl_XXXX).yaml
COMMON="--skill-dir skills/paper/S0_seed --split train --max-rounds 12 --parallel ${PAR:-14} --sim-base-port $PORT --headless --oracle-mode objective --executor-effort low --generator-effort xhigh"
case "$KIND" in
  noreg)
    export EVO_FEEDBACK_FORMAT=cluster
    sed -e "s/^  seed: 42/  seed: 45/" -e "s/^  generator_trials_per_turn: 10/  generator_trials_per_turn: 0/" evo/configs/main.yaml > "$CFG"
    exec python -m evo.engine.loop --config "$CFG" $COMMON --regression-tol -1 ;;
  score)
    export EVO_FEEDBACK_FORMAT=score
    sed "s/^  seed: 42/  seed: 46/" evo/configs/main.yaml > "$CFG"
    exec python -m evo.engine.loop --config "$CFG" $COMMON --gate-split val --gate-mode paired --gate-reps 3 --gate-z 1.5 --no-report ;;
  *) echo "usage: $0 <noreg|score> [base port]"; exit 1 ;;
esac
