"""
Train a morphological BPE tokenizer on FineWeb text data.

Usage:
    # Train with morpheme merges (recommended)
    python tokenizer/train.py --morphemes --vocab_size=50257

    # Train vanilla BPE (no morphemes, for comparison)
    python tokenizer/train.py --vocab_size=50257

    # Quick test with small data
    python tokenizer/train.py --morphemes --vocab_size=4096 --max_chars=10_000_000

    # Custom morpheme file (one per line)
    python tokenizer/train.py --morpheme_file=my_morphemes.txt --vocab_size=50257
"""

import os
import sys
import time
import json
import pickle
import argparse

from morphological_bpe import MorphologicalBPETrainer
from morphemes import DEFAULT_MORPHEMES

# Add parent dir so we can import data utils if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def text_iterator_from_fineweb(max_chars, doc_cap):
    """Stream text from FineWeb dataset via HuggingFace."""
    from datasets import load_dataset
    nchars = 0
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for doc in ds:
        text = doc["text"]
        if len(text) > doc_cap:
            text = text[:doc_cap]
        nchars += len(text)
        yield text
        if nchars >= max_chars:
            return


def text_iterator_from_file(path, max_chars, doc_cap):
    """Stream text from a local file (one document per line or full file)."""
    nchars = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if len(text) > doc_cap:
                text = text[:doc_cap]
            nchars += len(text)
            yield text
            if nchars >= max_chars:
                return


def main():
    parser = argparse.ArgumentParser(description="Train a morphological BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Target vocabulary size (default: 50257, matching GPT-2)")
    parser.add_argument("--max_chars", type=int, default=1_000_000_000,
                        help="Maximum characters to train on (default: 1B)")
    parser.add_argument("--doc_cap", type=int, default=10_000,
                        help="Maximum characters per document (default: 10,000)")
    parser.add_argument("--morphemes", action="store_true",
                        help="Enable forced morpheme merges (uses built-in English morpheme list)")
    parser.add_argument("--morpheme_file", type=str, default=None,
                        help="Path to custom morpheme file (one morpheme per line)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Train from local text file instead of FineWeb")
    parser.add_argument("--output_dir", type=str, default="tokenizer/output",
                        help="Directory to save trained tokenizer")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()

    # Resolve morphemes
    morpheme_list = None
    if args.morpheme_file:
        with open(args.morpheme_file, "r") as f:
            morpheme_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(morpheme_list)} morphemes from {args.morpheme_file}")
    elif args.morphemes:
        morpheme_list = DEFAULT_MORPHEMES
        print(f"Using {len(morpheme_list)} built-in English morphemes")
    else:
        print("No morphemes enabled (vanilla BPE)")

    # Build text iterator
    if args.input_file:
        print(f"Training from file: {args.input_file}")
        text_iter = text_iterator_from_file(args.input_file, args.max_chars, args.doc_cap)
    else:
        print(f"Training from FineWeb (max {args.max_chars:,} chars)")
        text_iter = text_iterator_from_fineweb(args.max_chars, args.doc_cap)

    # Train
    print(f"Target vocab size: {args.vocab_size:,}")
    trainer = MorphologicalBPETrainer()
    t0 = time.time()
    trainer.train_from_iterator(
        text_iter,
        vocab_size=args.vocab_size,
        forced_morphemes=morpheme_list,
        verbose=args.verbose or True,  # always verbose for training
    )
    t1 = time.time()
    print(f"Training time: {t1 - t0:.1f}s")
    print(f"Total merges: {len(trainer.merges)} "
          f"({trainer.forced_merge_count} forced + "
          f"{len(trainer.merges) - trainer.forced_merge_count} statistical)")

    # Export to tiktoken
    eot_id = args.vocab_size - 1  # reserve last ID for <|endoftext|>
    special_tokens = {"<|endoftext|>": eot_id}
    enc = trainer.to_tiktoken(name="morphological_bpe", special_tokens=special_tokens)

    # Sanity check
    test_text = "The unhappiness of the disconnected international community was unreasonable."
    ids = enc.encode(test_text)
    decoded = enc.decode(ids)
    assert decoded == test_text, f"Round-trip failed: {decoded!r} != {test_text!r}"
    print(f"Sanity check passed: {len(test_text)} chars -> {len(ids)} tokens")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save tiktoken encoding as pickle
    enc_path = os.path.join(args.output_dir, "tokenizer.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved tiktoken encoding to {enc_path}")

    # Save mergeable ranks for inspection
    ranks_path = os.path.join(args.output_dir, "mergeable_ranks.json")
    mergeable_ranks = trainer.get_mergeable_ranks()
    # JSON-serializable version (hex-encode the bytes keys)
    json_ranks = {k.hex(): v for k, v in mergeable_ranks.items()}
    with open(ranks_path, "w") as f:
        json.dump(json_ranks, f, indent=2)
    print(f"Saved mergeable ranks to {ranks_path}")

    # Save training metadata
    meta = {
        "vocab_size": args.vocab_size,
        "forced_morphemes": morpheme_list,
        "forced_merge_count": trainer.forced_merge_count,
        "total_merges": len(trainer.merges),
        "statistical_merges": len(trainer.merges) - trainer.forced_merge_count,
        "max_chars": args.max_chars,
        "training_time_seconds": t1 - t0,
        "eot_token_id": eot_id,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Print morpheme token mapping if morphemes were used
    if morpheme_list:
        print(f"\nForced morpheme tokens:")
        for morpheme in sorted(morpheme_list, key=lambda m: len(m.encode("utf-8"))):
            token_ids = enc.encode(morpheme)
            if len(token_ids) == 1:
                print(f"  {morpheme!r:>12} -> token {token_ids[0]}")


if __name__ == "__main__":
    main()
