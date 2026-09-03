# `data/prep/`

Scripts that rebuild every parquet under `data/` from its public source. No
dataset ships with this repository, so these scripts are the only supported way
to obtain the data.

Outputs default to `<repo>/data/<domain>/<dataset>/`, resolved by `_paths.py`
from the script's own location and overridable with `DATA_BASE`. Run from the
repository root:

```bash
python data/prep/<script>.py [--out ...]
```

| Script | Output | External dependency |
| --- | --- | --- |
| `ms_fetch.sh` | Any ModelScope model or dataset repository | None, plain `curl` |
| `prepare_dapo_math500.py` | `math/DAPO/train.parquet`, `math/MATH-500/test.parquet` | None, fetches from ModelScope |
| `prepare_amc23_aime25.py` | `math/AMC23/test.parquet` (40 rows), `math/AIME25/test.parquet` (30 rows) | None, fetches `knoveleng/AMC-23` and `opencompass/AIME2025` from ModelScope |
| `prepare_taco_train.py` | `code/TACO/train.parquet` | None, fetches the 9 arrow shards of `BAAI/TACO` (~4.2 GB) from ModelScope into `$TACO_CACHE`; `ms_fetch.sh` fetches the same files with curl |
| `prepare_code_eval.py` | `code/{LCBv6,HumanEvalPlus,MBPPPlus}/test.parquet` | None, fetches `test6.jsonl` and evalscope's mirrors of `evalplus/{humanevalplus,mbppplus}` from ModelScope |
| `prepare_xlam_train.py` | `agentic/xlam/train.parquet` | Local copy of `Salesforce/xlam-function-calling-60k` (`--src`; ModelScope carries it, see below) |
| `prepare_bfcl_eval.py` | `agentic/BFCL/<subset>/test.parquet` | Local copy of `gorilla-llm/BFCL` |
| `make_multiif_train.py` | `if/MultiIF/train.parquet` (4501 rows, first turn) | `--src multiIF_20241018.csv` |
| `prepare_multiif_eval.py` | `if/MultiIF/eval_3turn.parquet` (4445 rows, all turns) | Same CSV |
| `prepare_wildchat_for_opd.py` | `chat/WildChat/train.parquet` | `WILDCHAT_DIR` pointing at the WildChat-1M shards (14 parquets, ~3.4 GB; ModelScope carries them, see below) |
| `prepare_dapo_oneshot.py` | `math/DAPO-oneshot/t22.parquet` | None, slices `math/DAPO/train.parquet` |
| `align_mopd_schema.py` | Schema verification and rewriting | None |

## Verification against the reference files

`prepare_dapo_math500.py --verify <parquet>` compares row by row and reports
byte-identical and equivalent counts separately.

| File | Byte-identical | Equivalent |
| --- | --- | --- |
| `math/DAPO/train.parquet` | 17917/17917 | 17917/17917 |
| `math/MATH-500/test.parquet` | 500/500 | 500/500 |
| `code/LCBv6/test.parquet` | 175/175 (prompt and ground truth) | — |

## Full rebuild

`hf-mirror.com` and `huggingface.co` are both SNI-reset in this environment
(`curl` exit 35/104, HTTP 000). ModelScope responds normally and carries every
required repository.

```bash
# Code teacher (~7.1 GB)
bash data/prep/ms_fetch.sh models agentica-org/DeepCoder-1.5B-Preview \
     "$MODEL_BASE/DeepCoder-1.5B-Preview"

# Math training set and validation set
python3 data/prep/prepare_dapo_math500.py --set all

# Code validation sets (test6.jsonl ~134 MB, 175 problems; HumanEval+ 164; MBPP+ 378)
python3 data/prep/prepare_code_eval.py --bench all

# Code training set (9 arrow shards, ~4.2 GB, downloaded on first run and
# resumed if interrupted)
export TACO_CACHE=<cache directory on persistent disk>
python3 data/prep/prepare_taco_train.py

# Function-calling training and validation sets
bash data/prep/ms_fetch.sh datasets Salesforce/xlam-function-calling-60k data/.cache/xlam
python3 data/prep/prepare_xlam_train.py --src data/.cache/xlam/xlam_function_calling_60k.json
# BFCL is not on ModelScope: --src is a copy of the huggingface.co dataset
# gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL_v3_*.json plus
# possible_answer/), fetched from a network that can reach it.
python3 data/prep/prepare_bfcl_eval.py --src <BFCL directory>

# WildChat stress-test set (14 shards, ~3.4 GB; read by no launcher)
bash data/prep/ms_fetch.sh datasets allenai/WildChat-1M data/.cache/wildchat data
WILDCHAT_DIR=data/.cache/wildchat/data python3 data/prep/prepare_wildchat_for_opd.py

# Multi-IF source CSV, used by both the training and evaluation sets
bash data/prep/ms_fetch.sh datasets facebook/Multi-IF data/.cache/multiif
python3 data/prep/make_multiif_train.py     # 4501 rows, turn 1, training
python3 data/prep/prepare_multiif_eval.py   # 4445 rows, 3 turns, evaluation

# Verify the MOPD concatenation precondition
python3 data/prep/align_mopd_schema.py
```

`ms_fetch.sh` uses `/usr/bin/curl` over plain HTTP and does not install the
`modelscope` package, whose dependency tree replaces `transformers 5.10.4`.
`/usr/bin/curl` rather than the conda copy: the latter ships a broken CA bundle
that fails every HTTPS request with exit 77. Downloads resume with `curl -C -`.

## Constraints

**`data_source` is the verifier's routing key, not a label.**
`prepare_code_eval.py` must write `livecodebench`. Any other string falls
through to the math verifier, which scores Python programs near zero without
raising.

**The six MOPD files must share one schema**, or `datasets` fails the
concatenation with "The features can't be aligned". All six generation scripts
write through `align_mopd_schema.write_aligned()`. Verify with
`python data/prep/align_mopd_schema.py`.

**Struct field ordering is not a difference.** `datasets` normalizes it when
generating splits, so TACO's `(role, content)` and DAPO's `(content, role)`
concatenate correctly. The verifier reports ordering differences without
failing on them.

**The Multi-IF training and evaluation sets overlap by construction.** Both come
from the same CSV; the training set is the first turn of the conversations in
the evaluation set. This follows the reference setup and should be stated
alongside any reported Multi-IF score.
