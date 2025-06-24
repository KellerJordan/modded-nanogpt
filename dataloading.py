import torch
import random
from pathlib import Path
from transformers import EsmTokenizer
from typing import Tuple, Optional


def _load_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, 'number of tokens read does not match header?'
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern: str, batch_size: int, rank: int, world_size: int):
        assert batch_size % world_size == 0
        self.world_size = world_size
        self.rank = rank
        self.files = sorted(Path.cwd().glob(filename_pattern))
        self.batch_size = batch_size
        self.local_batch_size = self.batch_size // self.world_size
        self.reset()

    def reset(self):
        self.next_shard = 0
        self.advance()

    def advance(self): # advance to next data shard
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.next_shard])
        self.next_shard = (self.next_shard + 1) % len(self.files)

    def next_batch(self):
        buf = self.tokens[self.pos + self.rank * self.local_batch_size:][:self.local_batch_size]
        # by @YouJiacheng: host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        sequence = buf.to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        # advance current position and load next shard if necessary
        self.pos += self.batch_size
        if self.pos + self.batch_size >= len(self.tokens):
            self.advance()
        return sequence


class DistributedPaddedDataLoader(DistributedDataLoader):
    def __init__(self, filename_pattern, seq_len, process_rank, num_processes, cls_id, eos_id, pad_id, max_epochs=1):
        self.cls_id = cls_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self._leftover_tokens = torch.empty(0, dtype=torch.uint8)
        self.max_epochs = max_epochs
        super().__init__(filename_pattern, seq_len, process_rank, num_processes)

    def advance(self):
        self.pos = 0

        # handle epoch limit
        if self.next_shard // len(self.files) < self.max_epochs:
            raw_tokens = _load_data_shard(self.files[self.next_shard % len(self.files)])
            raw_tokens = torch.cat([self._leftover_tokens, raw_tokens], dim=0)
            self.next_shard += 1
        else:
            raw_tokens = self._leftover_tokens
        if not raw_tokens.numel():
            self._leftover_tokens = torch.empty(0, dtype=torch.uint8)
            self.tokens = torch.empty(0, dtype=torch.uint8)
            return
        
        # shuffle each epoch
        if self.next_shard % len(self.files) == 0:
            random.seed(self.next_shard)
            random.shuffle(self.files)

        processed_chunks = []
        curr_batch_len = 0
        eos_positions = (raw_tokens == self.eos_id).nonzero(as_tuple=True)[0]

        for i in range(len(eos_positions)):
            curr_eos = eos_positions[i]
            prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1  # EOS_idx + 1 = CLS_idx
            sample = raw_tokens[prev_eos_plus_one:curr_eos+1]  # One sample: "CLS ... EOS"

            if not sample[0] == self.cls_id and sample[-1] == self.eos_id:
                print(f"Warning: sample[0]=={sample[0]}, sample[-1]=={sample[-1]}, sample.numel()=={sample.numel()}")
                print(f"\ti={i}, eos_positions[:i]=={eos_positions[:i]}")
            assert curr_batch_len < self.local_batch_size, str((curr_batch_len, self.local_batch_size))

            # if adding sample exceeds the batch size resulting in truncation, pad to end of batch, starting a fresh batch
            if len(sample) + curr_batch_len >= self.local_batch_size:
                num_pad = self.local_batch_size - curr_batch_len
                processed_chunks.append(torch.full((num_pad,), self.pad_id, dtype=torch.uint8))
                curr_batch_len = 0

            # if len(sample) > local batch size, chunk evenly, making multiple padded batches, starting a fresh batch
            if len(sample) > self.local_batch_size:
                for split_sample in torch.chunk(sample, len(sample) // self.local_batch_size + 1):
                    processed_chunks.append(split_sample)
                    num_pad = self.local_batch_size - len(split_sample)
                    processed_chunks.append(torch.full((num_pad,), self.pad_id, dtype=torch.uint8))
                curr_batch_len = 0
                continue

            processed_chunks.append(sample)
            curr_batch_len += len(sample)
            curr_batch_len = 0 if curr_batch_len == self.local_batch_size else curr_batch_len

        self._leftover_tokens = raw_tokens[curr_eos+1:]
        self.tokens = torch.cat(processed_chunks, dim=0)


class MaskedDistributedPaddedDataLoader(DistributedPaddedDataLoader):
    """
    Enhanced dataloader that applies masking on the CPU side and supports async processing.
    """
    def __init__(
        self, 
        filename_pattern, 
        seq_len, 
        process_rank, 
        num_processes, 
        cls_id, 
        eos_id, 
        pad_id, 
        max_epochs=1,
        mask_token_id=None,
        special_token_ids=None,
        training=True,
        num_workers=0,
        prefetch_factor=2
    ):
        super().__init__(filename_pattern, seq_len, process_rank, num_processes, cls_id, eos_id, pad_id, max_epochs)
        
        # Masking parameters
        self.mask_token_id = mask_token_id if mask_token_id is not None else 32  # Default ESM2 mask token
        self.special_token_ids = special_token_ids if special_token_ids is not None else torch.tensor([cls_id, eos_id, pad_id])
        self.training = training
        
        # Async processing parameters (simplified)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Pre-computed batches cache for async processing
        self._batch_cache = []
        self._cache_size = max(1, num_workers * prefetch_factor)
        self._fill_cache()

    def _fill_cache(self):
        """Pre-fill the batch cache for async processing."""
        while len(self._batch_cache) < self._cache_size:
            try:
                raw_batch = super().next_batch()
                if raw_batch.numel() == 0:
                    break
                masked_batch, labels = self._apply_masking(raw_batch)
                self._batch_cache.append((masked_batch, labels))
            except Exception as e:
                print(f"Cache fill error: {e}")
                break

    def _apply_masking(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking logic to input sequence."""
        eps = 1e-3
        input_ids = input_ids.flatten()
        seq_len = len(input_ids)
        device = input_ids.device

        if self.training:  # sample uniform between 0 and 1
            t = torch.rand(1, device=device)
            t = (1 - eps) * t + eps
        else:  # evaluate at classic 15%
            t = torch.full((1,), 0.15, device=device)

        p_mask = t.repeat(seq_len)
        mask_indices = torch.rand(seq_len, device=device) < p_mask
        
        # prevent special tokens from being masked (cls, sep, eos, etc.)
        special_token_ids = self.special_token_ids.to(device)
        special_mask = torch.isin(input_ids, special_token_ids)
        mask_indices = mask_indices & ~special_mask

        # Create masked input
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        
        # Create labels
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices
        labels[non_mask_indices] = -100

        return noisy_batch, labels

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch with masking applied."""
        # Try to get from cache first
        if self._batch_cache:
            batch = self._batch_cache.pop(0)
            # Refill cache in background if needed
            if len(self._batch_cache) < self._cache_size // 2:
                self._fill_cache()
            return batch
        
        # If cache is empty, generate batch on-demand
        raw_batch = super().next_batch()
        if raw_batch.numel() == 0:
            return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)
        return self._apply_masking(raw_batch)

    def set_training(self, training: bool):
        """Set training mode for masking probability."""
        self.training = training
        # Clear cache when switching modes to ensure fresh batches
        self._batch_cache.clear()
        self._fill_cache()

    def reset(self):
        """Reset the dataloader and clear cache."""
        super().reset()
        self._batch_cache.clear()
        self._fill_cache()


def create_masked_dataloaders(
    train_pattern: str,
    valid_pattern: str,
    test_pattern: str,
    batch_size: int,
    rank: int,
    world_size: int,
    tokenizer: Optional[EsmTokenizer] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2
) -> Tuple[MaskedDistributedPaddedDataLoader, MaskedDistributedPaddedDataLoader, MaskedDistributedPaddedDataLoader]:
    """
    Factory function to create masked dataloaders for train/valid/test.
    """
    if tokenizer is None:
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    cls_id, eos_id, pad_id = tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    
    # Get special token IDs (excluding mask token)
    mask_token = tokenizer.mask_token
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(v) 
        for k, v in tokenizer.special_tokens_map.items() 
        if v != mask_token
    ]
    special_token_ids = torch.tensor(list(set(special_token_ids)))
    
    # Create dataloaders
    train_loader = MaskedDistributedPaddedDataLoader(
        train_pattern, batch_size, rank, world_size,
        cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=100,
        mask_token_id=mask_token_id, special_token_ids=special_token_ids,
        training=True, num_workers=num_workers, prefetch_factor=prefetch_factor
    )
    
    valid_loader = MaskedDistributedPaddedDataLoader(
        valid_pattern, batch_size // 8, rank, world_size,
        cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=1,
        mask_token_id=mask_token_id, special_token_ids=special_token_ids,
        training=False, num_workers=num_workers, prefetch_factor=prefetch_factor
    )
    
    test_loader = MaskedDistributedPaddedDataLoader(
        test_pattern, batch_size // 8, rank, world_size,
        cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=1,
        mask_token_id=mask_token_id, special_token_ids=special_token_ids,
        training=False, num_workers=num_workers, prefetch_factor=prefetch_factor
    )
    
    return train_loader, valid_loader, test_loader
