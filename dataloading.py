import torch
import random
from pathlib import Path
from transformers import EsmTokenizer
from typing import Tuple
import torch.utils.data as data
from torch.utils.data import DataLoader, IterableDataset


def _load_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8)
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


class DistributedDataLoaderWithMasking:
    def __init__(
            self,
            filename_pattern: str,
            batch_size: int,
            rank: int,
            world_size: int,
            training: bool,
            tokenizer: EsmTokenizer
        ):
        assert batch_size % world_size == 0
        self.world_size = world_size
        self.rank = rank
        self.files = sorted(Path.cwd().glob(filename_pattern))
        self.batch_size = batch_size
        self.local_batch_size = self.batch_size // self.world_size
        self.training = training
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        special_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = torch.tensor(special_tokens, device="cuda")
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
        sequence = sequence.flatten()

        # pick mask rate
        eps = 1e-3

        if self.training:
            mask_rate = torch.rand(1, device=sequence.device)
            mask_rate = (1 - eps) * mask_rate + eps
        else:
            mask_rate = torch.full((1,), 0.15, device=sequence.device)

        p_mask = mask_rate.repeat(len(sequence))
        mask_indices = torch.rand(len(sequence), device=sequence.device) < p_mask
        special_mask = torch.isin(sequence, self.special_tokens)
        mask_indices = mask_indices & ~special_mask

        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)
        labels = sequence.clone()
        non_mask_indices = ~mask_indices
        labels[non_mask_indices] = -100

        # advance current position and load next shard if necessary
        self.pos += self.batch_size
        if self.pos + self.batch_size >= len(self.tokens):
            self.advance()
        return noisy_batch, labels, mask_rate


class DistributedPaddedDataLoader(DistributedDataLoaderWithMasking):
    def __init__(
            self,
            filename_pattern,
            seq_len,
            process_rank,
            num_processes,
            max_epochs,
            training,
            tokenizer
        ):
        self._leftover_tokens = torch.empty(0, dtype=torch.uint8)
        self.max_epochs = max_epochs
        super().__init__(filename_pattern, seq_len, process_rank, num_processes, training, tokenizer)

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
        eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]

        for i in range(len(eos_positions)):
            curr_eos = eos_positions[i]
            prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1  # EOS_idx + 1 = CLS_idx
            sample = raw_tokens[prev_eos_plus_one:curr_eos+1]  # One sample: "CLS ... EOS"

            if not sample[0] == self.cls_token_id and sample[-1] == self.eos_token_id:
                print(f"Warning: sample[0]=={sample[0]}, sample[-1]=={sample[-1]}, sample.numel()=={sample.numel()}")
                print(f"\ti={i}, eos_positions[:i]=={eos_positions[:i]}")
            assert curr_batch_len < self.local_batch_size, str((curr_batch_len, self.local_batch_size))

            # if adding sample exceeds the batch size resulting in truncation, pad to end of batch, starting a fresh batch
            if len(sample) + curr_batch_len >= self.local_batch_size:
                num_pad = self.local_batch_size - curr_batch_len
                processed_chunks.append(torch.full((num_pad,), self.pad_token_id, dtype=torch.uint8))
                curr_batch_len = 0

            # if len(sample) > local batch size, chunk evenly, making multiple padded batches, starting a fresh batch
            if len(sample) > self.local_batch_size:
                for split_sample in torch.chunk(sample, len(sample) // self.local_batch_size + 1):
                    processed_chunks.append(split_sample)
                    num_pad = self.local_batch_size - len(split_sample)
                    processed_chunks.append(torch.full((num_pad,), self.pad_token_id, dtype=torch.uint8))
                curr_batch_len = 0
                continue

            processed_chunks.append(sample)
            curr_batch_len += len(sample)
            curr_batch_len = 0 if curr_batch_len == self.local_batch_size else curr_batch_len

        self._leftover_tokens = raw_tokens[curr_eos+1:]
        self.tokens = torch.cat(processed_chunks, dim=0)


class DistributedPaddedIterableDataset(IterableDataset):
    """An IterableDataset that handles distributed padded data loading with masking."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        training: bool,
        tokenizer: EsmTokenizer,
        num_workers: int = 1,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.max_epochs = max_epochs
        self.training = training
        self.num_workers = num_workers
        
        # Tokenizer IDs
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.special_tokens = [self.cls_token_id, self.eos_token_id, self.pad_token_id]
        
        # Get all files and distribute across processes (GPUs)
        all_files = sorted(Path.cwd().glob(filename_pattern))
        if not all_files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")
        
        # First distribute files across processes (GPUs)
        files_per_process = len(all_files) // self.num_processes
        extra_files = len(all_files) % self.num_processes
        
        start_idx = self.process_rank * files_per_process + min(self.process_rank, extra_files)
        end_idx = start_idx + files_per_process + (1 if self.process_rank < extra_files else 0)
        
        self.process_files = all_files[start_idx:end_idx]
    
    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            # Single worker mode
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Then distribute this process's files across workers
        files_per_worker = len(self.process_files) // num_workers
        extra_files = len(self.process_files) % num_workers
        
        start_idx = worker_id * files_per_worker + min(worker_id, extra_files)
        end_idx = start_idx + files_per_worker + (1 if worker_id < extra_files else 0)
        
        worker_files = self.process_files[start_idx:end_idx]
        
        # Process files cyclically for multiple epochs
        epoch = 0
        file_idx = 0
        leftover_tokens = torch.empty(0, dtype=torch.uint8)
        
        while epoch < self.max_epochs:
            # Shuffle files at the start of each epoch
            if file_idx == 0 and epoch > 0:
                random.seed(epoch + worker_id * 1000)
                random.shuffle(worker_files)
            
            # Load current file
            if file_idx < len(worker_files):
                raw_tokens = _load_data_shard(worker_files[file_idx])
                raw_tokens = torch.cat([leftover_tokens, raw_tokens], dim=0)
                file_idx += 1
            else:
                # End of epoch
                if leftover_tokens.numel() == 0:
                    epoch += 1
                    file_idx = 0
                    continue
                raw_tokens = leftover_tokens
                leftover_tokens = torch.empty(0, dtype=torch.uint8)
            
            # Process the tokens into batches
            eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(eos_positions) == 0:
                leftover_tokens = raw_tokens
                if file_idx >= len(worker_files):
                    epoch += 1
                    file_idx = 0
                continue
            
            # Process samples and create batches
            batch_tokens = []
            curr_batch_len = 0
            
            for i in range(len(eos_positions)):
                curr_eos = eos_positions[i]
                prev_eos_plus_one = 0 if i == 0 else eos_positions[i-1] + 1
                sample = raw_tokens[prev_eos_plus_one:curr_eos+1]
                
                # Handle samples that exceed batch size
                if len(sample) > self.seq_len:
                    # Split large samples into multiple batches
                    for j in range(0, len(sample), self.seq_len):
                        chunk = sample[j:j+self.seq_len]
                        if len(chunk) < self.seq_len:
                            # Pad the last chunk
                            padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)
                            chunk = torch.cat([chunk, padding])
                        
                        # Apply masking and yield batch
                        input_ids, labels, mask_rate = self._apply_masking(chunk)
                        yield input_ids, labels, mask_rate
                    continue
                
                # Check if adding this sample would exceed batch size
                if len(sample) + curr_batch_len > self.seq_len:
                    # Pad current batch and yield
                    if curr_batch_len > 0:
                        padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                        batch_tokens.append(padding)
                        batch = torch.cat(batch_tokens)
                        
                        # Apply masking and yield
                        input_ids, labels, mask_rate = self._apply_masking(batch)
                        yield input_ids, labels, mask_rate
                    
                    # Start new batch
                    batch_tokens = [sample]
                    curr_batch_len = len(sample)
                else:
                    # Add to current batch
                    batch_tokens.append(sample)
                    curr_batch_len += len(sample)
                
                # Yield complete batch
                if curr_batch_len == self.seq_len:
                    batch = torch.cat(batch_tokens)
                    input_ids, labels, mask_rate = self._apply_masking(batch)
                    yield input_ids, labels, mask_rate
                    batch_tokens = []
                    curr_batch_len = 0
            
            # Save leftover tokens for next file
            if len(eos_positions) > 0:
                leftover_tokens = raw_tokens[eos_positions[-1]+1:]
            
            # Yield final incomplete batch if at end of epoch
            if file_idx >= len(worker_files) and curr_batch_len > 0:
                padding = torch.full((self.seq_len - curr_batch_len,), self.pad_token_id, dtype=torch.uint8)
                batch_tokens.append(padding)
                batch = torch.cat(batch_tokens)
                input_ids, labels, mask_rate = self._apply_masking(batch)
                yield input_ids, labels, mask_rate
                
                epoch += 1
                file_idx = 0
    
    def _apply_masking(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to a sequence (on CPU)."""
        # Convert to int32
        sequence = sequence.to(dtype=torch.int32)
        
        # Pick mask rate
        eps = 1e-3
        if self.training:
            mask_rate = torch.rand(1)
            mask_rate = (1 - eps) * mask_rate + eps
        else:
            mask_rate = torch.full((1,), 0.15)
        
        # Create mask
        p_mask = mask_rate.repeat(len(sequence))
        mask_indices = torch.rand(len(sequence)) < p_mask
        
        # Don't mask special tokens
        special_mask = torch.isin(sequence, torch.tensor(self.special_tokens, dtype=torch.int32))
        mask_indices = mask_indices & ~special_mask
        
        # Create noisy batch and labels
        noisy_batch = torch.where(mask_indices, self.mask_token_id, sequence)
        labels = sequence.clone()
        labels[~mask_indices] = -100
        
        return noisy_batch, labels, mask_rate


class OptimizedDistributedPaddedDataLoader:
    """Drop-in replacement for DistributedPaddedDataLoader using multi-worker optimization."""
    
    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        max_epochs: int,
        training: bool,
        tokenizer: EsmTokenizer,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        self.filename_pattern = filename_pattern
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.training = training
        
        # Create the dataset to get file count
        self._dataset = DistributedPaddedIterableDataset(
            filename_pattern=filename_pattern,
            seq_len=seq_len,
            process_rank=process_rank,
            num_processes=num_processes,
            max_epochs=max_epochs,
            training=training,
            tokenizer=tokenizer,
            num_workers=num_workers,
        )
        
        # Store file list for compatibility - only this process's files
        self.files = self._dataset.process_files
        
        # Create the optimized dataloader
        self.dataloader = DataLoader(
            self._dataset,
            batch_size=None,  # Dataset returns complete batches
            num_workers=num_workers,
            pin_memory=True,  # Pin memory for faster GPU transfer
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        )
        
        # Create iterator
        self._iterator = None
        self._exhausted = False
    
    def reset(self):
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)
        self._exhausted = False
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch, ensuring GPU transfer happens here."""
        if self._iterator is None:
            self.reset()
        
        try:
            input_ids, labels, mask_rate = next(self._iterator)
            # Transfer to GPU with non-blocking
            input_ids = input_ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            mask_rate = mask_rate.cuda(non_blocking=True)
            return input_ids, labels, mask_rate
        except StopIteration:
            self._exhausted = True
            # Return empty tensors to signal end of data
            return torch.empty(0, device='cuda'), torch.empty(0, device='cuda'), torch.empty(0, device='cuda')
