#!/bin/bash
# OPD -- on-policy distillation from ONE frozen teacher.
#
# The teacher scores the student's own rollouts; the per-token reverse KL
# -(logp_student - logp_teacher) becomes the token-level reward. No verifier
# reward is used for the gradient (see grpo.sh for that baseline).
#
# Usage:
#     bash recipe/opd.sh                       # math, full data
#     DOMAIN=code bash recipe/opd.sh           # code, full data
#     MODE=oneshot bash recipe/opd.sh          # 1-shot prompts
#     MODE=template bash recipe/opd.sh         # student writes its own input
#
# DOMAIN picks the teacher and the parquets; MODE picks which prompts are read.
# Anything appended on the command line passes through to Hydra untouched.

# --- domain defaults -------------------------------------------------------
# Runs BEFORE _common.sh is sourced: _common.sh resolves MAX_PROMPT_LENGTH /
# MAX_RESP_LENGTH into MAX_MODEL_LEN and SEQ_LEN_TOKENS at source time, so a
# domain needing a different budget has to say so first. Only names are chosen
# here; MODEL_BASE and DATA_BASE do not exist yet.
DOMAIN=${DOMAIN:-math}
case "$DOMAIN" in
    math)
        DOMAIN_TEACHER=JustRL-DeepSeek-1.5B
        DOMAIN_TRAIN=math/DAPO/train.parquet
        DOMAIN_VAL=math/MATH-500/test.parquet
        ;;
    code)
        DOMAIN_TEACHER=DeepCoder-1.5B-Preview
        # TACO, not LCBv6: LCBv6 is the validation set and ships no train split.
        DOMAIN_TRAIN=code/TACO/train.parquet
        DOMAIN_VAL=code/LCBv6/test.parquet
        ;;
    fc)
        # The one domain whose student is not the shared R1 distill.
        DOMAIN_ACTOR=Qwen2.5-Coder-1.5B-Instruct
        DOMAIN_TEACHER=Hammer2.1-1.5b
        DOMAIN_TRAIN=agentic/xlam/train.parquet
        DOMAIN_VAL="agentic/BFCL/simple/test.parquet
                    agentic/BFCL/multiple/test.parquet
                    agentic/BFCL/parallel/test.parquet
                    agentic/BFCL/parallel_multiple/test.parquet
                    agentic/BFCL/live_simple/test.parquet"
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
        ;;
    if)
        DOMAIN_TEACHER=UltraData-IF-1.5B
        DOMAIN_TRAIN=if/MultiIF/train.parquet
        # Empty on purpose: Multi-IF is three turns and the validation loop
        # generates once, so eval/run_if_eval.py is this domain's only source of
        # numbers.
        DOMAIN_VAL=
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
        ;;
    *)
        echo "unknown DOMAIN='$DOMAIN' (expected: math | code | fc | if)" >&2
        exit 1
        ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export PROJECT_NAME=${PROJECT_NAME:-OPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/${DOMAIN_ACTOR:-DeepSeek-R1-Distill-Qwen-1.5B}}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${MODEL_BASE}/${DOMAIN_TEACHER}}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

# DOMAIN_VAL is a whitespace-separated list of DATA_BASE-relative paths; Hydra
# wants one bracketed list. Empty stays empty -- see the `if` branch.
if [ -z "${VAL_FILES:-}" ]; then
    _val_list=
    for _p in $DOMAIN_VAL; do
        _val_list="${_val_list:+${_val_list},}${DATA_BASE}/${_p}"
    done
    export VAL_FILES="[${_val_list}]"
fi
case "$MODE" in
    full)
        export TRAIN_FILES=${TRAIN_FILES:-[${DATA_BASE}/${DOMAIN_TRAIN}]}
        ;;
    oneshot)
        # t22 is DAPO problem 22 repeated 64 times -- 64 rows, one distinct
        # prompt. Built by data/prep/prepare_dapo_oneshot.py.
        if [ -z "${TRAIN_FILES:-}" ] && [ "$DOMAIN" != "math" ]; then
            echo "MODE=oneshot has no corpus for DOMAIN=$DOMAIN -- only math" \
                 "ships one (math/DAPO-oneshot/). Pass TRAIN_FILES to override." >&2
            exit 1
        fi
        export TRAIN_FILES=${TRAIN_FILES:-[${DATA_BASE}/math/DAPO-oneshot/t22.parquet]}
        ;;
    template)
        # Prompts are generated, not read. Hydra still requires the key.
        export TRAIN_FILES=${TRAIN_FILES:-$VAL_FILES}
        ;;
esac

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-opd_${DOMAIN}_${MODE}_${ACTOR_MODEL_NAME}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Domain:  $DOMAIN"
echo "Mode:    $MODE"
echo "Actor:   $ACTOR_MODEL_PATH"
echo "Teacher: $TEACHER_MODEL_PATH"
echo "Train:   $TRAIN_FILES"
echo "Val:     $VAL_FILES"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(teacher_args "$TEACHER_MODEL_PATH") \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
