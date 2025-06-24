#!/usr/bin/env python3

import torch
import time
from transformers import EsmTokenizer
from dataloading import DistributedPaddedDataLoader, OptimizedDistributedPaddedDataLoader
import argparse


def test_dataloader(dataloader_class, name, args):
    """Test a dataloader and measure its performance."""
    print(f"\nTesting {name}...")
    
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    if dataloader_class == OptimizedDistributedPaddedDataLoader:
        loader = dataloader_class(
            filename_pattern=args.input_bin,
            seq_len=args.batch_size,
            process_rank=0,
            num_processes=1,
            max_epochs=1,
            training=True,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        loader = dataloader_class(
            filename_pattern=args.input_bin,
            seq_len=args.batch_size,
            process_rank=0,
            num_processes=1,
            max_epochs=1,
            training=True,
            tokenizer=tokenizer,
        )
    
    print(f"Number of files: {len(loader.files)}")
    
    # Warm up
    loader.reset()
    for _ in range(5):
        input_ids, labels, mask_rate = loader.next_batch()
        if input_ids.numel() == 0:
            break
    
    # Measure loading time
    loader.reset()
    start_time = time.time()
    num_batches = 0
    total_tokens = 0
    
    while num_batches < args.num_batches:
        input_ids, labels, mask_rate = loader.next_batch()
        if input_ids.numel() == 0:
            print(f"Reached end of data after {num_batches} batches")
            break
        
        num_batches += 1
        total_tokens += input_ids.numel()
        
        if num_batches % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {num_batches} batches, {total_tokens:,} tokens in {elapsed:.2f}s "
                  f"({total_tokens/elapsed:,.0f} tokens/s)")
    
    elapsed = time.time() - start_time
    print(f"\nFinal results for {name}:")
    print(f"  Total batches: {num_batches}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens/elapsed:,.0f} tokens/s")
    print(f"  Time per batch: {elapsed/num_batches*1000:.2f}ms")
    
    return elapsed, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Test optimized dataloader")
    parser.add_argument("--input_bin", type=str, default='data/omgprot50/omgprot50_train_*.bin',
                        help="Input bin files pattern")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size in tokens")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to test")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor")
    parser.add_argument("--compare", action="store_true", help="Compare with original dataloader")
    
    args = parser.parse_args()
    
    print(f"Testing dataloaders with batch_size={args.batch_size}, num_batches={args.num_batches}")
    
    # Test optimized dataloader
    opt_time, opt_tokens = test_dataloader(
        OptimizedDistributedPaddedDataLoader, 
        f"OptimizedDistributedPaddedDataLoader (workers={args.num_workers})",
        args
    )
    
    # Optionally compare with original
    if args.compare:
        orig_time, orig_tokens = test_dataloader(
            DistributedPaddedDataLoader,
            "DistributedPaddedDataLoader (original)",
            args
        )
        
        print(f"\nSpeedup: {orig_time/opt_time:.2f}x")
        print(f"Throughput improvement: {(opt_tokens/opt_time)/(orig_tokens/orig_time):.2f}x")


if __name__ == "__main__":
    main() 