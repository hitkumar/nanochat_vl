import argparse
import os
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests
from nanochat_vl.common import get_base_dir

BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_SHARD = 1822
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
BASE_DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(BASE_DATA_DIR, exist_ok=True)


def download_single_file(index):
    filename = index_to_filename(index)
    filepath = os.path.join(BASE_DATA_DIR, filename)
    print(filepath)
    if os.path.exists(filepath):
        print(f"file at {index} exists")
        return

    remote_url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(remote_url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.rename(temp_path, filepath)
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    os.removce(path)

            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"waiting {wait_time} before retry file at index {index}")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def list_parquet_files(data_dir=None):
    data_dir = BASE_DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    return [os.path.join(data_dir, p) for p in parquet_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fineweb-Edu Dataset shards")
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download, -1 means download all the files",
    )

    args = parser.parse_args()
    num_files = (
        MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    )
    ids_to_download = list(range(num_files))
    print(
        f"Downloading {len(ids_to_download)} files using {args.num_workers} workers..."
    )

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Downloaded {successful} / {len(ids_to_download)} shards to {BASE_DATA_DIR}")

# print(get_base_dir())
