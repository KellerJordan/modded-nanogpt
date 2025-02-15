import os
import numpy as np
import glob
import torch
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast  # type: ignore

class DistributedDataLoader:
    """Handles distributed loading and batching of binary token data."""
    
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files found matching pattern: {filename_pattern}"

        # load and validate all data shards, count total tokens
        ntok_total = 0
        for fname in self.files:
            shard_ntok = self._peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1, \
                f"Shard {fname} too small for batch size and sequence length"
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # initialize state
        self.reset()

    def _peek_data_shard(self, filename):
        """Read only the header of a data shard to get token count."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            
        if header[0] != 20240520:
            raise ValueError(
                "Magic number mismatch in data file. Possible causes:\n"
                "- Incorrect --input_bin parameter\n"
                "- Dataset encoding changed, needs preprocessing\n"
                "- Run appropriate data preparation script first"
            )
        assert header[1] == 1, "Unsupported version"
        return header[2]  # number of tokens

    def _load_data_shard(self, filename):
        """Load full data shard including tokens."""
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520, "Magic number mismatch"
            assert header[1] == 1, "Unsupported version"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
            
        assert len(tokens) == ntok, "Token count mismatch"
        return tokens

    def reset(self):
        """Reset loader state to beginning of dataset."""
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = self._load_data_shard(self.files[self.current_shard])

    def advance(self):
        """Advance to next data shard."""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = self._load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        """Get next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        
        # advance position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
            
        return x.cuda(), y.cuda()

def setup_tokenizer(config):
    """Initialize the appropriate tokenizer based on the dataset."""
    if "tinystories_char" in config.data_path:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(config.data_path, "tokenizer_ts_char.json"),
            eos_token="<|endoftext|>",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
        config.vocab_size = tokenizer.vocab_size
        dataset_name = "TinyStoriesChar"
    
    elif "tinystories" in config.data_path:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(config.data_path, "tinystories_tokenizer.json"),
            eos_token="<|endoftext|>",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
        config.vocab_size = tokenizer.vocab_size
        dataset_name = "TinyStories"
    
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
        dataset_name = "FineWeb"
    
    return tokenizer, dataset_name

def setup_data(config, ddp_rank, ddp_world_size):
    """Initialize data loaders."""
    B, T = config.device_batch_size, config.sequence_length
    
    train_loader = DistributedDataLoader(
        config.input_bin, B, T, ddp_rank, ddp_world_size
    )
    
    val_loader = DistributedDataLoader(
        config.input_val_bin, B, T, ddp_rank, ddp_world_size
    )
    
    if ddp_rank == 0:
        print(f"Training DataLoader: {train_loader.ntok_total} tokens "
              f"across {len(train_loader.files)} files.")
        print(f"Validation DataLoader: {val_loader.ntok_total} tokens "
              f"across {len(val_loader.files)} files.")
    
    return train_loader, val_loader 