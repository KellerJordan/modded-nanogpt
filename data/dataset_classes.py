import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset
from typing import List, Dict


class SequenceDatasetFromList(TorchDataset):
    def __init__(self, sequences, **kwargs):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class IterableDatasetFromHF(IterableDataset):
    def __init__(self, dataset, col_name='seqs', **kwargs):
        """
        Wrap a streaming Hugging Face dataset (IterableDataset) into a PyTorch IterableDataset.
        
        Args:
            dataset (IterableDataset): Streaming Hugging Face dataset.
            col_name (str): The column name containing the sequences.
        """
        self.dataset = dataset
        self.col_name = col_name

    def __iter__(self):
        for example in self.dataset:
            yield example[self.col_name]


class SequenceCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.eos_token = tokenizer.eos_token

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        seq = ''.join([self.cls_token + s + self.eos_token for s in batch])
        input_ids = self.tokenizer.encode(seq, add_special_tokens=False, return_tensors='pt')
        return {'input_ids':input_ids}


class TokenBasedIterableDataset(IterableDataset):
    def __init__(self, dataset, target_token_count=8192, col_name='seqs', **kwargs):
        """
        Wrap a streaming dataset to yield batches based on token count rather than sequence count.
        
        Args:
            dataset (IterableDataset): Streaming Hugging Face dataset
            tokenizer: Tokenizer to use for counting tokens
            target_token_count (int): Target number of tokens per batch
            col_name (str): Column name containing sequences
        """
        self.dataset = dataset
        self.target_token_count = target_token_count
        self.col_name = col_name

    def __iter__(self):
        accumulated_sequences = []
        current_token_count = 0
        
        for example in self.dataset:
            seq = example[self.col_name]
            seq_token_count = len(seq) + 2 # +2 for cls and eos tokens
            
            # If adding this sequence would exceed target and we have accumulated sequences, yield batch
            if current_token_count + seq_token_count > self.target_token_count and accumulated_sequences:
                yield accumulated_sequences
                accumulated_sequences = []
                current_token_count = 0
            
            accumulated_sequences.append(seq)
            current_token_count += seq_token_count
            
            # If we've reached the target, yield batch
            if current_token_count >= self.target_token_count:
                yield accumulated_sequences
                accumulated_sequences = []
                current_token_count = 0
        
        # Yield any remaining sequences
        if accumulated_sequences:
            yield accumulated_sequences


class TokenBasedSequenceCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.eos_token = tokenizer.eos_token

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        seq = ''.join([self.cls_token + s + self.eos_token for s in batch])
        input_ids = self.tokenizer.encode(seq, add_special_tokens=False, return_tensors='pt')
        return {'input_ids':input_ids}