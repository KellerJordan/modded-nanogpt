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
