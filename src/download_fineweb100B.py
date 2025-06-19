# Run with  uv run cached_fineweb100B.py`
# /// script
# dependencies = [
#   "huggingface_hub",
#   "etils[epath]",
#   "gcsfs",
# ]
# ///
import os
import sys

from etils import epath
from huggingface_hub import hf_hub_download

TMP_DIR = epath.Path("/tmp/fineweb100B/")
TARGET_DIR = epath.Path("gs://rail-dibya-central2/fineweb100B/")


# Download the GPT-2 tokens of Fineweb100B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    target_path = TARGET_DIR / fname
    if not target_path.exists():
        hf_hub_download(
            repo_id="kjj0/fineweb100B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=TMP_DIR,
        )
        local_path = TMP_DIR / fname
        local_path.copy(target_path)
        local_path.unlink()
    return target_path


get("fineweb_val_%06d.bin" % 0)
num_chunks = 1030  # full fineweb100B. Each chunk is 100M tokens

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(get, f"fineweb_train_{i:06d}.bin") for i in range(num_chunks)
    ]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
