#!/usr/bin/env python3
"""
Example script demonstrating token-based batching approaches for protein sequence training.
"""

import torch
from datasets import load_dataset
from transformers import EsmTokenizer
from data.dataset_classes import (
    TokenBasedCollator, 
    TokenBasedIterableDataset, 
    IterableDatasetFromHF,
    SequenceCollator
)

def main():
    # Load tokenizer
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    # Load dataset
    dataset = load_dataset("Synthyra/omg_prot50", split="train", streaming=True).shuffle(seed=42)
    
    print("=" * 60)
    print("APPROACH 1: TokenBasedCollator")
    print("=" * 60)
    print("Uses PyTorch DataLoader with variable batch sizes based on token count")
    print()
    
    # Approach 1: Use TokenBasedCollator with regular DataLoader
    target_token_count = 1024
    regular_dataset = IterableDatasetFromHF(dataset, col_name='sequence')
    token_collator = TokenBasedCollator(tokenizer, target_token_count=target_token_count)
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        regular_dataset, 
        batch_size=10,  # This will be dynamic based on token count
        collate_fn=token_collator
    )
    
    print(f"Target token count: {target_token_count}")
    print("First 3 batches from TokenBasedCollator:")
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        input_ids = batch['input_ids']
        actual_tokens = input_ids.shape[1]
        print(f"Batch {i+1}: {actual_tokens} tokens")
    
    print()
    print("=" * 60)
    print("APPROACH 2: TokenBasedIterableDataset")
    print("=" * 60)
    print("Pre-processes sequences to yield consistent token counts")
    print()
    
    # Approach 2: Use TokenBasedIterableDataset
    dataset_reset = load_dataset("Synthyra/omg_prot50", split="train", streaming=True).shuffle(seed=42)
    token_dataset = TokenBasedIterableDataset(
        dataset_reset, 
        tokenizer, 
        target_token_count=target_token_count, 
        col_name='sequence'
    )
    
    # Simple collator for pre-processed sequences
    def simple_collator(batch):
        # Each item in batch is already a concatenated sequence string
        input_ids = tokenizer.encode(batch[0], add_special_tokens=False, return_tensors='pt')
        return {'input_ids': input_ids}
    
    dataloader2 = DataLoader(
        token_dataset,
        batch_size=1,  # Since dataset yields complete token-counted batches
        collate_fn=simple_collator
    )
    
    print(f"Target token count: {target_token_count}")
    print("First 3 batches from TokenBasedIterableDataset:")
    
    for i, batch in enumerate(dataloader2):
        if i >= 3:
            break
        input_ids = batch['input_ids']
        actual_tokens = input_ids.shape[1]
        print(f"Batch {i+1}: {actual_tokens} tokens")
    
    print()
    print("=" * 60)
    print("COMPARISON: Traditional Sequence-Based Batching")
    print("=" * 60)
    
    # Traditional approach for comparison
    dataset_reset2 = load_dataset("Synthyra/omg_prot50", split="train", streaming=True).shuffle(seed=42)
    regular_dataset2 = IterableDatasetFromHF(dataset_reset2, col_name='sequence')
    sequence_collator = SequenceCollator(tokenizer)
    
    dataloader3 = DataLoader(
        regular_dataset2,
        batch_size=5,  # Fixed number of sequences
        collate_fn=sequence_collator
    )
    
    print("Fixed batch size: 5 sequences")
    print("First 3 batches from traditional SequenceCollator:")
    
    for i, batch in enumerate(dataloader3):
        if i >= 3:
            break
        input_ids = batch['input_ids']
        actual_tokens = input_ids.shape[1]
        print(f"Batch {i+1}: {actual_tokens} tokens")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("• TokenBasedCollator: More consistent token counts per batch")
    print("• TokenBasedIterableDataset: Pre-processes for exact token targets")
    print("• Traditional: Variable token counts based on sequence lengths")
    print()
    print("For training with consistent computational load, use token-based approaches!")

if __name__ == "__main__":
    main() 