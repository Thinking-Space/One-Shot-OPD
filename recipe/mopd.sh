#!/bin/bash
# MOPD -- multi-teacher OPD with ability routing.
#
# Same distillation objective as opd.sh, but several frozen teachers. Each row is
# routed to a teacher by its `ability` column; rows sharing a teacher are
# forwarded as one group and scattered back into the original batch order, so the
# reward tensor keeps its input ordering. Every step logs
#     [MOPD] batch routing: {teacher: count}
#
# Usage:
#     bash recipe/mopd.sh                      # full data
#     MODE=oneshot bash recipe/mopd.sh         # 1-shot prompts
#
# Adding or removing a teacher is a parameter change: pass name:path pairs to
# multi_teacher_args and extend ABILITY_TO_TEACHER.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$MODE" = "template" ]; then
    echo "MODE=template is single-teacher only; use opd.sh" >&2
    exit 2
fi

export PROJECT_NAME=${PROJECT_NAME:-MOPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B}
export TEACHER_MATH_PATH=${TEACHER_MATH_PATH:-${MODEL_BASE}/JustRL-DeepSeek-1.5B}
export TEACHER_CODE_PATH=${TEACHER_CODE_PATH:-${MODEL_BASE}/Nemotron-Research-Reasoning-Qwen-1.5B-v2}
export TEACHER_IF_PATH=${TEACHER_IF_PATH:-${MODEL_BASE}/UltraData-IF-1.5B}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export ABILITY_TO_TEACHER=${ABILITY_TO_TEACHER:-'{math:math,code:code,instruction_following:if}'}
export STRICT_ROUTING=${STRICT_ROUTING:-True}

if [ "$MODE" = "oneshot" ]; then
    # Only math ships a 1-shot corpus. Pass CODE_TRAIN/IF_TRAIN explicitly if you
    # want this combination.
    MATH_TRAIN=${MATH_TRAIN:-${DATA_BASE}/math/DAPO-oneshot/t22.parquet}
else
    MATH_TRAIN=${MATH_TRAIN:-${DATA_BASE}/math/DAPO/train.parquet}
fi

CODE_TRAIN=${CODE_TRAIN:-${DATA_BASE}/code/TACO/train.parquet}
IF_TRAIN=${IF_TRAIN:-${DATA_BASE}/if/MultiIF/train.parquet}
export TRAIN_FILES=${TRAIN_FILES:-["$MATH_TRAIN","$CODE_TRAIN","$IF_TRAIN"]}


export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet","${DATA_BASE}/code/LCBv6/test.parquet"]}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-mopd_${MODE}_${ACTOR_MODEL_NAME}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Mode:         $MODE"
echo "Actor:        $ACTOR_MODEL_PATH"
echo "Teacher math: $TEACHER_MATH_PATH"
echo "Teacher code: $TEACHER_CODE_PATH"
echo "Teacher if:   $TEACHER_IF_PATH"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(multi_teacher_args \
        "math:${TEACHER_MATH_PATH}" \
        "code:${TEACHER_CODE_PATH}" \
        "if:${TEACHER_IF_PATH}") \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
