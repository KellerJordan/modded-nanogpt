import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split

def main():
    # 1) Load and split TinyStories
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    split = split["test"].train_test_split(test_size=0.01, seed=42)
    train_texts = split["train"]["text"]
    val_texts   = split["test"]["text"]

    # 2) Build a *character-level* tokenizer using a WordLevel backbone
    #    (In principle, "WordLevel" can be used purely for collecting 'tokens',
    #     even if they're single characters, as long as we define the right pre_tokenizer/trainer.)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    # Split with each character as a separate token
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=1,  # If you want absolutely every character to appear
    )

    # 3) Train the tokenizer on the training set (concatenated text)
    train_texts_list = list(train_texts)  # HF dataset to normal Python list
    tokenizer.train_from_iterator(train_texts_list, trainer=trainer)

    # 4) Define a helper function to encode text as IDs:
    def tokenize_texts(tokenizer, texts):
        ids = []
        for text in texts:
            output = tokenizer.encode(text)
            ids.extend(output.ids)
        return np.array(ids, dtype=np.uint8)

    train_ids = tokenize_texts(tokenizer, train_texts)
    val_ids   = tokenize_texts(tokenizer, val_texts)
    
    def save_with_header(filename, ids):
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # Magic number
        header[1] = 1         # Version
        header[2] = len(ids)  # Number of tokens
        with open(filename, "wb") as f:
            f.write(header.tobytes())  # Write the header (256 * 4 bytes)
            f.write(ids.tobytes())     # Write the token IDs as uint16

    # 5) Save the tokenized data as .bin for efficient reading
    os.makedirs("data/tinystories_char", exist_ok=True)
    save_with_header("data/tinystories_char/train.bin", train_ids)
    save_with_header("data/tinystories_char/val.bin", val_ids)

    # 6) Save the tokenizer itself (JSON file) so you can load it later
    tokenizer.save("data/tinystories_char/char_tokenizer.json")

    print("Done! Created train.bin, val.bin, and char_tokenizer.json in data/tinystories_char/")

if __name__ == "__main__":
    main()
