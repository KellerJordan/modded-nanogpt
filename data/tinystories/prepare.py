import os
from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# Create data directory if it doesn't exist
data_dir = 'data/tinystories'
os.makedirs(data_dir, exist_ok=True)

# Load the TinyStories dataset
train_dataset = load_dataset('roneneldan/TinyStories', split='train')
val_dataset = load_dataset('roneneldan/TinyStories', split='validation')

# Define the special tokens
SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>"
}

# Initialize the tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()

# Set up the trainer
VOCAB_SIZE = 1000  # Custom vocabulary size
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[SPECIAL_TOKENS["eos_token"], "[UNK]"]
)

# Prepare the training data for the tokenizer
def get_training_corpus():
    for sample in train_dataset:
        yield sample['text']

# Train the tokenizer
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Configure the tokenizer post-processing
tokenizer.post_processor = TemplateProcessing(
    single="$A " + SPECIAL_TOKENS["eos_token"],
    special_tokens=[
        (SPECIAL_TOKENS["eos_token"], tokenizer.token_to_id(SPECIAL_TOKENS["eos_token"]))
    ],
)

# Save the tokenizer
tokenizer.save("tinystories_tokenizer.json")

# Load it with transformers for compatibility
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tinystories_tokenizer.json",
    eos_token=SPECIAL_TOKENS["eos_token"],
    unk_token="[UNK]",
    pad_token="[PAD]"  # Optional, but can be useful
)

# Function to tokenize and save dataset to binary file
def tokenize_and_save(dataset, filename):
    # Initialize a list to hold all tokens
    all_tokens = []

    # Process each text entry in the dataset
    for sample in dataset:
        tokens = tokenizer.encode(sample['text'])
        # Tokens already include the eos_token due to post-processing
        all_tokens.extend(tokens)

    # Convert tokens to NumPy array
    arr = np.array(all_tokens, dtype=np.uint16)

    # Prepare the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1         # Version
    header[2] = arr.size  # Number of tokens

    # Save to binary file
    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        arr.tofile(f)

train_bin_path = os.path.join(data_dir, 'train.bin')
tokenize_and_save(train_dataset, train_bin_path)

val_bin_path = os.path.join(data_dir, 'val.bin')
tokenize_and_save(val_dataset, val_bin_path)

# Verify the maximum token ID is within the vocabulary size
def verify_tokens(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    max_token_id = tokens.max()
    print(f"Maximum token id in {filename}: {max_token_id}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

verify_tokens(train_bin_path)
verify_tokens(val_bin_path)
print("Data preparation is complete.")