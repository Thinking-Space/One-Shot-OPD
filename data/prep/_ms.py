"""Fetch single files from a ModelScope dataset repository.

Shared by the prep scripts that need files too large or too many for the
``_fetch`` one-liner in prepare_dapo_math500.py: TACO's nine ~500 MB shards
resume after an interrupted download instead of starting over.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["MS_FILE", "fetch"]

#: Same endpoint as ms_fetch.sh and prepare_dapo_math500.py.
MS_FILE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"


def fetch(repo: str, path: str, dest: Path, url: str | None = None) -> Path:
    """Download ``path`` of ModelScope dataset ``repo`` to ``dest``; no-op if present.

    Args:
        repo: ``owner/name`` on ModelScope.
        path: file path inside the repository.
        dest: local destination. A ``.part`` file next to it holds the partial
              download and is resumed with a Range request on the next call.
        url:  full URL overriding the ModelScope one, for a different mirror.

    Returns:
        ``dest``.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest}")
        return dest
    url = url or MS_FILE.format(repo=repo, path=path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    print(f"  get   {url}\n     -> {dest}" + (f" (resuming at {have} bytes)" if have else ""))

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # nothing left to fetch: the .part is already complete
            part.rename(dest)
            return dest
        raise
    with resp:
        # 206 honours the Range; a 200 means the server sent the whole file,
        # so the partial copy is discarded rather than appended to.
        mode = "ab" if (have and resp.status == 206) else "wb"
        with part.open(mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    part.rename(dest)
    return dest
