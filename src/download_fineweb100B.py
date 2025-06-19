# Run with  uv run cached_fineweb100B.py`
# /// script
# dependencies = [
#   "huggingface_hub",
#   "etils[epath]",
#   "gcsfs",
#   "tyro",
# ]
# ///

import tyro
from etils import epath
from huggingface_hub import hf_hub_download


def main(target_dir: str, tmp_dir: str | None = None, num_workers: int = 10):
    target_dir = epath.Path(target_dir)
    tmp_dir = epath.Path(tmp_dir) if tmp_dir else None

    # Download the GPT-2 tokens of Fineweb100B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    def get(fname):
        target_path = target_dir / fname
        if not target_path.exists():
            hf_hub_download(
                repo_id="kjj0/fineweb100B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=tmp_dir or target_dir,
            )
            if tmp_dir:
                local_path = tmp_dir / fname
                local_path.copy(target_path)
                local_path.unlink()
        return target_path

    get("fineweb_val_%06d.bin" % 0)
    num_chunks = 1029  # full fineweb100B. Each chunk is 100M tokens

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(get, f"fineweb_train_{i:06d}.bin")
            for i in range(1, num_chunks)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    tyro.cli(main)
