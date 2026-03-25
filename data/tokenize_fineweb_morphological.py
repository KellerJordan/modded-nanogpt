"""
Re-tokenize FineWeb with a trained morphological BPE tokenizer.

Produces .bin shards in the exact same format as fineweb.py:
  - 256 int32 header (magic=20240520, version=1, num_tokens)
  - uint16 token data

Usage:
    # Re-tokenize FineWeb-10B with trained tokenizer
    python data/tokenize_fineweb_morphological.py \
        --tokenizer_path=tokenizer/output/tokenizer.pkl \
        --version=10B

    # Smaller test run
    python data/tokenize_fineweb_morphological.py \
        --tokenizer_path=tokenizer/output/tokenizer.pkl \
        --version=10B --max_shards=5
"""

import os
import pickle
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def write_datafile(filename, toks):
    """
    Save token data as a .bin file matching modded-nanogpt format.
    Header: 256 int32s (magic, version, num_tokens, then zeros)
    Data: uint16 tokens
    """
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of tokens
    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


# Global tokenizer for multiprocessing (set in init_worker)
_enc = None
_eot = None

def init_worker(tokenizer_path):
    """Initialize tokenizer in each worker process."""
    global _enc, _eot
    with open(tokenizer_path, "rb") as f:
        _enc = pickle.load(f)
    _eot = _enc._special_tokens["<|endoftext|>"]


def tokenize_doc(doc):
    """Tokenize a single document, prepending EOT delimiter."""
    tokens = [_eot]
    tokens.extend(_enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def main():
    parser = argparse.ArgumentParser(description="Re-tokenize FineWeb with morphological BPE")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to trained tokenizer pickle (tokenizer.pkl)")
    parser.add_argument("--version", type=str, default="10B", choices=["10B", "100B"],
                        help="FineWeb version to tokenize (default: 10B)")
    parser.add_argument("--shard_size", type=int, default=10**8,
                        help="Tokens per shard (default: 100M)")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Maximum number of shards to write (for testing)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/fineweb{version}_morph/)")
    args = parser.parse_args()

    # Verify tokenizer exists
    assert os.path.exists(args.tokenizer_path), f"Tokenizer not found: {args.tokenizer_path}"

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), f"fineweb{args.version}_morph")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    remote_name = f"sample-{args.version}T"
    print(f"Loading FineWeb {args.version} (streaming)...")
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    # Tokenize with multiprocessing
    nprocs = max(1, os.cpu_count() - 2)
    print(f"Tokenizing with {nprocs} workers using {args.tokenizer_path}")

    with mp.Pool(nprocs, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        shard_index = 0
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize_doc, fw, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens",
                                       desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write current shard
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(args.output_dir,
                                       f"fineweb_{split}_{shard_index:06d}.bin")
                remainder = args.shard_size - token_count
                if progress_bar:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1

                if args.max_shards and shard_index >= args.max_shards:
                    print(f"Reached max_shards={args.max_shards}, stopping.")
                    break

                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write remaining tokens
        if token_count != 0 and not (args.max_shards and shard_index >= args.max_shards):
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(args.output_dir,
                                   f"fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

    print(f"Done! Shards written to {args.output_dir}/")
    print(f"To use: set train_files/val_files in train_gpt.py to point to {args.output_dir}/")


if __name__ == "__main__":
    main()
