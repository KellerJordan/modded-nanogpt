import os
import argparse
from huggingface_hub import hf_hub_download


### Download the OMGprot50 tokens (ESM2 tokenizer) from huggingface
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'omgprot50')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="Synthyra/omg_prot50_packed", filename=fname, repo_type="dataset", local_dir=local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OMGprot50 tokens from huggingface")
    parser.add_argument("-n", "--num_chunks", type=int, default=442, help="Number of chunks to download")
    # each chunk is 100M tokens
    args = parser.parse_args()
    get("omgprot50_val_%06d.bin" % 0)
    get("omgprot50_test_%06d.bin" % 0)
    for i in range(1, args.num_chunks+1):
        get("omgprot50_train_%06d.bin" % i)
