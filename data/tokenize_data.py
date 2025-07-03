"""
example doc to highlight the structure of the dataset:
{
  "sequence": "MYDSNIFEKVNQYKFLYIWWLIMINVNH"
}
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import glob
from functools import partial
from transformers import EsmTokenizer
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi


def upload_folder_to_hf(folder_path, repo_id, repo_type="dataset", token=None):
    """
    Upload an entire folder to Hugging Face Hub (bulk upload to avoid rate limiting)
    
    Benefits:
    - Uploads all files in a single operation instead of individual requests
    - Automatically handles large uploads with multi-commit strategy
    - Reduces API rate limiting issues
    - More efficient for large numbers of files
    """
    if repo_id is None:
        print(f"Skipping upload for {folder_path} - no repo_id specified")
        return
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        print(f"Uploading folder {folder_path} to {repo_id}...")
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                exist_ok=True
            )
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Repository might already exist: {e}")
        
        # Count files to upload
        file_count = len([f for f in os.listdir(folder_path) if f.endswith('.bin')])
        print(f"Found {file_count} files to upload")
        
        # Try to use multi_commits for large uploads (if supported)
        try:
            if file_count > 100:  # Use multi-commit for large uploads
                print("Using multi-commit upload for large number of files...")
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    multi_commits=True,
                    multi_commits_verbose=True
                )
            else:
                # Standard upload for smaller sets
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
        except TypeError as e:
            if "multi_commits" in str(e):
                print("multi_commits not supported in this version of huggingface_hub, using standard upload...")
                # Fall back to standard upload
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
            else:
                raise e
        
        print(f"Successfully uploaded folder {folder_path} to {repo_id}")
        
    except Exception as e:
        print(f"Error uploading folder {folder_path}: {e}")


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint8
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 1 byte as uint8)
    # construct the tokens numpy array, if not already
    print(f"\nwriting {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def tokenize(doc, tokenizer, max_length):
    # tokenizes a single document and returns a numpy array of uint8 tokens
    # uint8 can hold the 33 tokens
    return np.array(tokenizer.encode(doc["sequence"], add_special_tokens=True, truncation=True, padding=False, max_length=max_length), dtype=np.uint8)


def tokenize_fw(fw, split='train', data_name='omgprot50', max_length=1024, upload_repo=None, token=None):
    # tokenize all documents and write output shards, each of approximately shard_size tokens
    # ensures each shard contains complete sequences only
    
    # Check if .bin files already exist for this dataset/split
    existing_files = glob.glob(os.path.join(DATA_CACHE_DIR, f"{data_name}_{split}_*.bin"))
    
    if existing_files:
        print(f"Found {len(existing_files)} existing .bin files for {data_name}_{split}")
        print("Skipping tokenization and proceeding to upload...")
        
        # Upload existing files if upload_repo is specified
        if upload_repo:
            upload_folder_to_hf(DATA_CACHE_DIR, upload_repo, token=token)
        else:
            print("No upload repository specified, files are ready locally")
        return
    
    print(f"No existing .bin files found for {data_name}_{split}, proceeding with tokenization...")
    
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        current_shard = []
        current_size = 0
        progress_bar = None
        tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=max_length)

        for tokens in pool.imap(tokenize_fn, fw, chunksize=16):
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            
            # If adding this sequence would exceed shard size, write current shard and start new one
            if current_size + len(tokens) > args.shard_size and current_size > 0:
                # Convert accumulated tokens to numpy array and write
                all_tokens_np = np.concatenate(current_shard)
                filename = os.path.join(DATA_CACHE_DIR, f"{data_name}_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np)
                
                # Reset for next shard
                shard_index += 1
                current_shard = []
                current_size = 0
                progress_bar = None
            
            # Add sequence to current shard
            current_shard.append(tokens)
            current_size += len(tokens)
            if progress_bar:
                progress_bar.update(len(tokens))

        # Write final shard if there are remaining sequences
        if current_size > 0:
            all_tokens_np = np.concatenate(current_shard)
            filename = os.path.join(DATA_CACHE_DIR, f"{data_name}_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)
    
    # Upload all files at once after tokenization is complete
    if upload_repo:
        upload_folder_to_hf(DATA_CACHE_DIR, upload_repo, token=token)


parser = argparse.ArgumentParser(description="OMGprot50 dataset preprocessing")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("-d", "--data_name", type=str, default="omgprot50", help="Name of the dataset")
parser.add_argument("-r", "--upload_repo", type=str, default=None, help="Hugging Face repository ID to upload to (e.g., 'username/repo_name')")
parser.add_argument("-t", "--token", type=str, default=None, help="Hugging Face token for authentication (or set token environment variable)")


if __name__ == "__main__":
    args = parser.parse_args()
    data_name = args.data_name
    
    # Get HF token from args or environment
    token = args.token or os.environ.get("token")
    if args.upload_repo and not token:
        print("Warning: Upload repository specified but no HF token provided. Set --token or token environment variable.")

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), data_name)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    train_fw = load_dataset(f"Synthyra/{data_name}", split="train")
    valid_fw = load_dataset(f"Synthyra/{data_name}", split="valid")
    test_fw = load_dataset(f"Synthyra/{data_name}", split="test")
    tokenize_fw(valid_fw, split='valid', data_name=data_name, max_length=args.max_length, upload_repo=args.upload_repo, token=token)
    tokenize_fw(test_fw, split='test', data_name=data_name, max_length=args.max_length, upload_repo=args.upload_repo, token=token)
    tokenize_fw(train_fw, split='train', data_name=data_name, max_length=100000, upload_repo=args.upload_repo, token=token) # don't trim training data
