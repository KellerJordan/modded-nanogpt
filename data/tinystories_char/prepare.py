import os
import argparse
import json
import numpy as np
from datasets import load_dataset

def main(args):
    # Load the TinyStories dataset splits using HuggingFace datasets.
    print("Loading TinyStories dataset splits...")
    train_dataset = load_dataset('roneneldan/TinyStories', split='train')
    val_dataset = load_dataset('roneneldan/TinyStories', split='validation')

    # Extract text from the dataset.
    # (Adjust the key if needed; here we assume the text is in the "text" column.
    #  Some versions might have "story", so you may change accordingly.)
    key = "text" if "text" in train_dataset.column_names else "story"
    train_texts = train_dataset[key]
    val_texts   = val_dataset[key]

    # Join all individual samples with a newline separator.
    train_text = "\n".join(train_texts)
    val_text   = "\n".join(val_texts)
    print(f"Train text length: {len(train_text)} characters")
    print(f"Validation text length: {len(val_text)} characters")

    # Build a character-level vocabulary from the training text.
    vocab = sorted(set(train_text))
    print(f"Initial vocabulary size: {len(vocab)}")

    # Special tokens
    # Even if you're just training a GPT, you might include:
    # - <|endoftext|>: Marks the end of a sequence. This helps the model know where a context ends.
    # - [PAD]: Useful for batching variable-length sequences, even if in our case you may use fixed-length chunks.
    # - [UNK]: Although a character-level vocabulary for a curated corpus should cover all characters,
    #          including an [UNK] token is common practice to safely handle any unexpected input.
    eos_token = "<|endoftext|>"
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    for token in [eos_token, pad_token, unk_token]:
        if token not in vocab:
            vocab.append(token)
    vocab = sorted(vocab)
    print(f"Vocabulary size after adding special tokens: {len(vocab)}")

    # Create mappings: character -> token id (stoi) and token id -> character (itos)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    # Save the tokenizer configuration as JSON.
    tokenizer_data = {
        "vocab": stoi,
        "itos": itos,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "unk_token": unk_token
    }
    tokenizer_path = os.path.join(args.out_dir, "tokenizer_ts_char.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Tokenize the texts on a character level.
    # Every character is converted to its corresponding token id.
    train_token_ids = np.array([stoi.get(ch, stoi[unk_token]) for ch in train_text], dtype=np.int16)
    val_token_ids   = np.array([stoi.get(ch, stoi[unk_token]) for ch in val_text], dtype=np.int16)
    print(f"Tokenized train text into {len(train_token_ids)} tokens.")
    print(f"Tokenized validation text into {len(val_token_ids)} tokens.")

    # Save token sequences to binary files.
    train_bin_path = os.path.join(args.out_dir, "train.bin")
    val_bin_path   = os.path.join(args.out_dir, "val.bin")
    train_token_ids.tofile(train_bin_path)
    val_token_ids.tofile(val_bin_path)
    print(f"Saved training tokens to {train_bin_path}")
    print(f"Saved validation tokens to {val_bin_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories character-level dataset for GPT training."
    )
    # Output directory where tokenizer JSON and .bin files will be saved.
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory to save tokenizer and binary files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
