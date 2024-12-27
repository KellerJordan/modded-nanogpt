import torch
from pathlib import Path
from transformers import EsmTokenizer
from utils import ProteinMasker


def _peek_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2]) # number of tokens (claimed)


def _load_data_shard(path: Path, num_tokens):
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, batch_size, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size

        # glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        self.total_num_tokens = sum(self.files_num_tokens)

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size
        self.tokens = _load_data_shard(self.files[self.current_shard], self.files_num_tokens[self.current_shard])

    def next_batch(self):
        batch_size = self.batch_size * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.batch_size]
        # host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        input_ids = buf.to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return input_ids
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, batch_size):
        from huggingface_hub import hf_hub_download 
        from datasets import Dataset as HFDataset
        local_file = hf_hub_download(
            repo_id="Synthyra/omg_prot50",
            filename=f"data/test-00000-of-00001.parquet",
            repo_type="dataset"
        )
        data = HFDataset.from_parquet(local_file)
        print(data)
        sequences = data['sequence']
        self.sequences = sorted(sequences, key=len, reverse=True)
        self.num_tokens = sum(len(seq) for seq in self.sequences)
        self.num_seqs = len(self.sequences)
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer, 0.15) # 15% masking rate like ESM2 for inference
        self.batch_size = batch_size
        self.current_idx = 0

    def __len__(self):
        return self.num_tokens // self.batch_size

    def __getitem__(self, idx):
        batch_input_ids = []
        current_length = 0

        for _ in range(self.num_seqs):
            if self.current_idx >= self.num_seqs:
                self.current_idx = 0
            
            input_ids = self.tokenizer(
                self.sequences[self.current_idx][:1022],
                truncation=False,
                padding=False,
                add_special_tokens=True).input_ids
            current_length += len(input_ids)
            if current_length >= self.batch_size:
                break
            batch_input_ids.extend(input_ids)
            self.current_idx += 1
            
        input_ids = torch.tensor(batch_input_ids).view(1, -1)
        input_ids, labels = self.masker(input_ids)
        return input_ids, labels
