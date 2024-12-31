import torch
from pathlib import Path


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
    def __init__(self, filename_pattern: str, batch_size: int, rank: int, world_size: int):
        assert batch_size % world_size == 0
        self.world_size = world_size
        self.rank = rank
        self.files = sorted(Path.cwd().glob(filename_pattern))
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.next_shard = 0
        self.advance()

    def advance(self): # advance to next data shard
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.next_shard])
        self.next_shard = (self.next_shard + 1) % len(self.files)

    def next_batch(self):
        local_batch_size = self.batch_size // self.world_size
        buf = self.tokens[self.pos + self.rank * local_batch_size:][:local_batch_size + 1]
        # by @YouJiacheng: host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # targets
        # advance current position and load next shard if necessary
        self.pos += self.batch_size
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.advance()
        return inputs, targets


class DistributedPaddedDataLoader(DistributedDataLoader):
    def __init__(self, filename_pattern, seq_len, process_rank, num_processes, eos_id, pad_id):
        super().__init__(filename_pattern, seq_len, process_rank, num_processes)
        self.eos_id = eos_id
        self.pad_id = pad_id

    def reset(self):
        self.current_shard = self.process_rank - self.num_processes
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + self.num_processes) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard], self.files_num_tokens[self.current_shard])

    def next_batch(self):
        end_pos = self.current_position + self.batch_size
        buf = self.tokens[self.current_position:end_pos]
        input_ids = buf.to(device="cuda", dtype=torch.int32, non_blocking=True)
        keep = (input_ids == self.eos_id).cumsum(dim=0).argmax().item()
        keep = max(keep or 0, self.batch_size - 2048)
        input_ids[keep + 1:] = self.pad_id
        # advance current position and load next shard if necessary
        self.current_position += keep
        if self.current_position + self.batch_size >= len(self.tokens):
            self.advance()
        return input_ids
