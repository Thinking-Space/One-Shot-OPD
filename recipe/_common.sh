#!/bin/bash
# Shared settings for every launch script in this recipe. Sourced, not executed.
#
#   grpo.sh   verifier reward only, no teacher            (baseline)
#   opd.sh    one frozen teacher scoring the student      MODE=full|oneshot|template
#   mopd.sh   several teachers, routed by ability column  MODE=full|oneshot
#
# All three go through `python -m verl.trainer.main_ppo` with trainer.use_v1=False.

set -x

# ============ paths ============
_OPD_RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPD_RECIPE_DIR=${OPD_RECIPE_DIR:-$_OPD_RECIPE_DIR}
export VERL_ROOT=${VERL_ROOT:-$(cd "${_OPD_RECIPE_DIR}/../verl" && pwd)}
# OPD_ROOT is where data/, logs/ and checkpoints/ live.
OPD_ROOT=${OPD_ROOT:-$(cd "${_OPD_RECIPE_DIR}/.." && pwd)}
export OPD_ROOT
export PYTHONPATH=${VERL_ROOT}:${PYTHONPATH}

export MODEL_BASE=${MODEL_BASE:-/path/to/models/opd/baselines}
export DATA_BASE=${DATA_BASE:-${OPD_ROOT}/data}

export CONDA_ENV_BIN=${CONDA_ENV_BIN:-/path/to/conda/envs/verl/bin}
export PATH="${CONDA_ENV_BIN}:$PATH"

# ============ mode ============
export MODE=${MODE:-full}
[ "$MODE" = "input" ] && export MODE=template
case "$MODE" in
    full|oneshot|template) ;;
    *)
        # exit, not return: `return` would only leave this sourced file.
        echo "MODE=${MODE} is not one of: full, oneshot, template (alias: input)" >&2
        exit 2
        ;;
esac

# ============ sequence lengths ============
# 4096 fits code/IF/agentic; the math launcher sets 1024 itself. Do not lower it
# globally: data.filter_overlong_prompts=True drops the row rather than truncating.
if [ "$MODE" = "oneshot" ]; then
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
fi
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH ))}

if [ "${MAX_MODEL_LEN}" -lt "$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))" ]; then
    echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} is shorter than the sequence it must" \
         "hold (MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} + MAX_RESP_LENGTH=" \
         "${MAX_RESP_LENGTH}). Raise MAX_VAL_RESP_LENGTH to at least" \
         "${MAX_RESP_LENGTH}, or set MAX_MODEL_LEN explicitly." >&2
    exit 1
fi

# ============ training hyper-parameters ============
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export N_RESPONSES=${N_RESPONSES:-1}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export USE_KL=${USE_KL:-False}

# ============ top-K distillation ============
#   LOG_PROB_TOP_K > 0  top-K IS the training reward (token reward becomes 3-D)
#   METRIC_TOP_K   > 0  top-K for metrics only; reward stays the per-token reverse KL
# LOG_PROB_TOP_K wins if both are set. 0/0 disables top-K entirely.
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export METRIC_TOP_K=${METRIC_TOP_K:-0}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
# Reward mode adds a candidate axis the stock `vanilla` loss cannot broadcast against.
if [ "${LOG_PROB_TOP_K}" -gt 0 ]; then
    export POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla_topk}
else
    export POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
fi

# ============ template mode (MODE=template) ============
export PREFIX_TEMPLATE=${PREFIX_TEMPLATE:-'{bos}{user}<think>\n'}
export PREFIX_SUFFIX=${PREFIX_SUFFIX:-''}
export PREFIX_MODE=${PREFIX_MODE:-template}
export MIN_CHAR_LENGTH=${MIN_CHAR_LENGTH:-20}
export TEMPLATE_TASK=${TEMPLATE_TASK:-math}
export TEMPLATE_DATASET_LENGTH=${TEMPLATE_DATASET_LENGTH:-1000000}
export TEMPLATE_DATA_SOURCE=${TEMPLATE_DATA_SOURCE:-template}
export OVERSAMPLE_RATIO=${OVERSAMPLE_RATIO:-1.0}

# ============ cluster ============
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-20}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# VAL_ONLY=True scores the current weights against data.val_files and exits.
export VAL_ONLY=${VAL_ONLY:-False}
[ "$VAL_ONLY" = "True" ] && export VAL_BEFORE_TRAIN=True

# ============ validation sampling ============
# avg@N, not greedy. Every reference number was produced under these settings.
export VAL_N=${VAL_N:-16}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
export VAL_TOP_P=${VAL_TOP_P:-0.9}

# ============ swanlab ============
export SWANLAB_PROJECT=${SWANLAB_PROJECT:-}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OPD_ROOT}/swanlab}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "${OPD_ROOT}/.swanlab_key" ]; then
    SWANLAB_API_KEY=$(tr -d ' \t\n\r' < "${OPD_ROOT}/.swanlab_key")
    export SWANLAB_API_KEY
fi

if [ -n "${SWANLAB_API_KEY:-}" ]; then
    export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','swanlab']"}
    mkdir -p "$SWANLAB_LOG_DIR"
else
    export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}
    echo "[opd] no SWANLAB_API_KEY (env or ${OPD_ROOT}/.swanlab_key); logging to console only" >&2
fi

# ============ runtime env ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export RAY_memory_usage_threshold=0.99
# ray.init's `uv run` probe walks ancestor processes and dies on NoSuchProcess.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
# `local` forces a fresh cluster: auto-discovery can attach to someone else's GCS.
export RAY_ADDRESS=${RAY_ADDRESS:-local}
# ray + vllm + FSDP exhaust the 1024 soft fd limit, and failure is silent.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
# Override to sdpa if flash-attn cannot be built for the installed torch/CUDA pair.
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
# DataLoader workers fork the trainer; with FSDP CPU offload they get SIGKILL'd.
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M)
export DATE TIME_TAG

mkdir -p "${OPD_ROOT}/logs"

# Dynamic-bsz token budget must cover one full sequence.
SEQ_LEN_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( SEQ_LEN_TOKENS > 32768 ? SEQ_LEN_TOKENS : 32768 ))}
if [ "${PPO_MAX_TOKEN_LEN_PER_GPU}" -lt "${SEQ_LEN_TOKENS}" ]; then
    echo "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU} cannot hold one" \
         "sequence (${MAX_PROMPT_LENGTH}+${MAX_RESP_LENGTH}=${SEQ_LEN_TOKENS})" >&2
    exit 1
fi
export PPO_MAX_TOKEN_LEN_PER_GPU

if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.005 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi
export KL_ARGS

# common_args -- the Hydra override block every launcher shares. Word-split on
# purpose (Hydra wants one token per override), so paths must not contain spaces.
common_args() {
    echo "\
        --config-name=ppo_trainer \
        trainer.use_v1=False \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        data.train_batch_size=$(( MINI_BATCH_SIZE * PARALLEL_SIZE )) \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESP_LENGTH} \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.return_raw_chat=True \
        data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
        actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        ++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${PARALLEL_SIZE} \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
        actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE} \
        ${KL_ARGS} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${PARALLEL_SIZE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.75} \
        actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
        actor_rollout_ref.rollout.max_num_seqs=${MINI_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${N_RESPONSES} \
        actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
        actor_rollout_ref.rollout.response_length=${MAX_RESP_LENGTH} \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
        actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_VAL_RESP_LENGTH} \
        reward.custom_reward_function.path=${VERL_ROOT}/verl/utils/reward_score/opd_mixed_verify.py \
        reward.custom_reward_function.name=reward_func \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=${TEST_FREQ} \
        trainer.val_before_train=${VAL_BEFORE_TRAIN} \
        trainer.val_only=${VAL_ONLY} \
        trainer.logger=${TRAINER_LOGGER}"
}

# mode_args -- what differs between MODE=full / oneshot / template.
#
# The prefix values are wrapped in literal single quotes: they contain `{`, `}`
# and `<`, which Hydra's override lexer would otherwise read as dict/list syntax.
mode_args() {
    if [ "$MODE" != "template" ]; then
        echo "data.shuffle=True"
        return
    fi
    echo "\
        data.shuffle=False \
        data.custom_cls.path=${VERL_ROOT}/verl/utils/dataset/template_dataset.py \
        data.custom_cls.name=TemplatePromptDataset \
        template.enabled=True \
        template.min_char_length=${MIN_CHAR_LENGTH} \
        template.prefix_template='${PREFIX_TEMPLATE}' \
        template.prefix_suffix='${PREFIX_SUFFIX}' \
        template.prefix_mode=${PREFIX_MODE} \
        template.task=${TEMPLATE_TASK} \
        template.dataset_length=${TEMPLATE_DATASET_LENGTH} \
        template.data_source=${TEMPLATE_DATA_SOURCE} \
        template.oversample_ratio=${OVERSAMPLE_RATIO}"
}

# teacher_args <path> -- single-teacher OPD reward block.
teacher_args() {
    echo "\
        $(_teacher_common_args) \
        distillation.fsdp_teacher.model_path=$1"
}

# multi_teacher_args <name:path> [name:path ...] -- MOPD reward block.
#
# teachers/ability_to_teacher are not in the structured config, so they need `++`
# and Hydra's bare-key syntax; JSON quoting fails the parser. No spaces allowed.
multi_teacher_args() {
    local entries="" spec
    for spec in "$@"; do
        entries="${entries}${entries:+,}{name:${spec%%:*},path:${spec#*:}}"
    done
    echo "\
        $(_teacher_common_args) \
        distillation.fsdp_teacher.strict_routing=${STRICT_ROUTING:-True} \
        ++distillation.fsdp_teacher.teachers=[${entries}] \
        ++distillation.fsdp_teacher.ability_to_teacher=${ABILITY_TO_TEACHER}"
}

_teacher_common_args() {
    echo "\
        distillation.fsdp_teacher.enabled=True \
        distillation.fsdp_teacher.micro_batch_size_per_gpu=${TEACHER_MICRO_BSZ:-24} \
        distillation.fsdp_teacher.dtype=${MODEL_DTYPE} \
        distillation.fsdp_teacher.teacher_temperature=${TEACHER_TEMPERATURE} \
        distillation.fsdp_teacher.log_prob_top_k=${LOG_PROB_TOP_K} \
        distillation.fsdp_teacher.metric_top_k=${METRIC_TOP_K} \
        distillation.fsdp_teacher.top_k_strategy=${TOP_K_STRATEGY} \
        distillation.fsdp_teacher.attn_implementation=${ATTN_IMPLEMENTATION} \
        distillation.fsdp_teacher.reward_weight_mode=${REWARD_WEIGHT_MODE} \
        distillation.fsdp_teacher.fsdp_config.param_offload=True"
}

# launch <experiment name> <override>... -- run main_ppo and tee the log.
#
# OPD_DRY_RUN=1 composes and prints the config instead of starting ray. `--cfg job`
# has to lead: Hydra's argparse stops collecting overrides at the first flag.
launch() {
    local exp="$1"
    shift
    set -- "$@" \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="$exp" \
        trainer.default_local_dir="${CKPT_PATH:-${OPD_ROOT}/checkpoints/${exp}}"

    if [ -n "${OPD_DRY_RUN:-}" ]; then
        python3 -m verl.trainer.main_ppo --cfg job "$@"
        return
    fi

    python3 -m verl.trainer.main_ppo "$@" \
        2>&1 | tee "${OPD_ROOT}/logs/${exp}.log"
}
