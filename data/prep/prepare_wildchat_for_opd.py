"""Build the WildChat set used for the off-domain stress test.

    python data/prep/prepare_wildchat_for_opd.py

Writes chat/WildChat/train.parquet (192824 rows). No launcher reads this file
by default.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_ROOT

# Directory of raw WildChat-1M parquet shards. No default: download
# allenai/WildChat-1M and point WILDCHAT_DIR at the extracted data/ directory.
DATA_DIR = Path(os.environ.get("WILDCHAT_DIR", "")) if os.environ.get("WILDCHAT_DIR") else None
OUT_PATH = DATA_ROOT / "chat" / "WildChat" / "train.parquet"


def extract_first_user_message(conv) -> str | None:
    """Extract the content of the first user message in a conversation.

    Args:
        conv: numpy.ndarray or list of {"role": ..., "content": ...} dicts

    Returns:
        Text of the first user message, or None if not found.
    """
    try:
        if hasattr(conv, '__len__') and len(conv) > 0:
            msg = conv[0]
            if isinstance(msg, dict) and msg.get('role') == 'user' and 'content' in msg:
                return msg['content']
    except Exception:
        pass
    return None


def main() -> None:
    if DATA_DIR is None:
        raise SystemExit(
            "set WILDCHAT_DIR to the directory holding WildChat-1M's parquet shards"
        )
    files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Loading {len(files)} parquet files...")

    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[
            'conversation', 'turn', 'language', 'toxic', 'redacted',
        ])
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"Raw total: {len(raw):,}")

    # Filter
    mask = (
        (raw['turn'] == 1) &
        (raw['language'] == 'English') &
        (~raw['toxic']) &
        (~raw['redacted'])
    )
    filtered = raw[mask].copy()
    print(f"After filter (turn=1, English, non-toxic): {len(filtered):,}")

    # Extract the first user message
    filtered['user_msg'] = filtered['conversation'].apply(extract_first_user_message)
    filtered = filtered.dropna(subset=['user_msg'])

    # Length filter
    msg_lens = filtered['user_msg'].str.len()
    filtered = filtered[(msg_lens >= 20) & (msg_lens <= 2000)]
    print(f"After length filter (20-2000 chars): {len(filtered):,}")

    # Build the OPD format
    records = []
    for i, (_, row) in enumerate(filtered.iterrows()):
        records.append({
            'prompt': np.array([{"role": "user", "content": row['user_msg']}], dtype=object),
            'data_source': 'WildChat',
            'ability': 'general',
            'reward_model': {"ground_truth": "", "style": "rule"},
            'extra_info': {"index": f"WildChat-{i}", "split": "train"},
        })

    out_df = pd.DataFrame(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Final dataset size: {len(out_df):,}")

    # Verify
    check = pd.read_parquet(OUT_PATH)
    print(f"\nVerification - shape: {check.shape}, columns: {list(check.columns)}")
    row0 = check.iloc[0]
    print(f"  prompt type: {type(row0['prompt'])}, len: {len(row0['prompt'])}")
    print(f"  prompt[0]: role={row0['prompt'][0]['role']}, content[:80]={row0['prompt'][0]['content'][:80]}")


if __name__ == "__main__":
    main()
