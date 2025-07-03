import os
import argparse
from huggingface_hub import hf_hub_download


### Download the data from huggingface
def get(fname, data_name):
    local_dir = os.path.join(os.path.dirname(__file__), data_name)
    if not os.path.exists(os.path.join(local_dir, fname)):
        try:
            hf_hub_download(repo_id=f"Synthyra/{data_name}_packed", filename=fname, repo_type="dataset", local_dir=local_dir)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")
    else:
        print(f"File {fname} already exists in {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from huggingface")
    parser.add_argument("-d", "--data_name", type=str, default="omgprot50", help="Name of the dataset")
    parser.add_argument("-n", "--num_chunks", type=int, default=442, help="Number of chunks to download")
    # each chunk is 100M tokens
    args = parser.parse_args()
    get(f"{args.data_name}_valid_%06d.bin" % 0, args.data_name)
    get(f"{args.data_name}_test_%06d.bin" % 0, args.data_name)
    for i in range(0, args.num_chunks+1):
        get(f"{args.data_name}_train_%06d.bin" % i, args.data_name)