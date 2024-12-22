import os
import sys
from huggingface_hub import hf_hub_download


def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'omgprot50')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="Synthyra/omg_prot50", filename=fname, repo_type="dataset", local_dir=local_dir)


get("data/valid-00000-of-00001.parquet")
get("data/test-00000-of-00001.parquet")
# Full omgprot50, which is roughly 52 billion tokens
# Each chunk is ~2.3 million sequences, ~600,000,000 tokens, 490 MB
num_chunks = 91


if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])


for i in range(1, num_chunks+1):
    get(f"data/train-{i:05d}-of-00091.parquet")
