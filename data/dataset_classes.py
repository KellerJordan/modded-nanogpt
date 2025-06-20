import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset
from typing import List, Tuple, Dict


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
    def __init__(self, tokenizer, max_length=512, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(batch,
                          padding='longest',
                          truncation=True,
                          max_length=self.max_length,
                          return_tensors='pt',
                          add_special_tokens=True)
        return batch
