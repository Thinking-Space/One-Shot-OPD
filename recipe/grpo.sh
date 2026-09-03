#!/bin/bash
# GRPO -- the reference baseline: verifier reward only, no teacher.
#
# Same entry point, data and lengths as opd.sh; the reward comes from the
# rule-based verifier instead, grouped into a GRPO advantage over N_RESPONSES
# samples per prompt. No teacher is loaded.
#
# Usage:
#     bash recipe/grpo.sh                      # full data
#     MODE=oneshot bash recipe/grpo.sh         # 1-shot prompts

# Read by _common.sh, so they are set before the source.
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESPONSES=${N_RESPONSES:-8}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$MODE" = "template" ]; then
    echo "MODE=template needs a teacher to score the generated prompts; use opd.sh" >&2
    exit 2
fi

export PROJECT_NAME=${PROJECT_NAME:-GRPO}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet"]}
if [ "$MODE" = "oneshot" ]; then
    export TRAIN_FILES=${TRAIN_FILES:-["${DATA_BASE}/math/DAPO-oneshot/t22.parquet"]}
else
    export TRAIN_FILES=${TRAIN_FILES:-["${DATA_BASE}/math/DAPO/train.parquet"]}
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-grpo_${MODE}_${ACTOR_MODEL_NAME}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Mode:  $MODE"
echo "Actor: $ACTOR_MODEL_PATH"
echo "Train: $TRAIN_FILES  (n=${N_RESPONSES} per prompt, adv=${ADV_ESTIMATOR})"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
