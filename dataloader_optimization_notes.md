# Optimized DataLoader Implementation

## Overview
The optimized dataloader implementation moves data loading and masking operations to separate CPU workers, allowing them to run in parallel with GPU computation. This prevents the GPU from waiting on data preparation.

## Key Changes

1. **Multi-worker Support**: The new `OptimizedDistributedPaddedDataLoader` uses PyTorch's DataLoader with multiple workers to parallelize data loading.

2. **CPU-based Preprocessing**: All data loading, padding, and masking operations now happen on CPU workers in parallel.

3. **Memory Pinning**: Uses pinned memory for faster CPU-to-GPU transfers.

4. **Persistent Workers**: Workers stay alive between epochs to avoid startup overhead.

## Usage

The optimized dataloader is a drop-in replacement for the original `DistributedPaddedDataLoader`. It's already integrated into `train.py` and controlled by two command-line arguments:

```bash
# Run with default settings (4 workers, prefetch factor 2)
python train.py

# Customize worker settings
python train.py --num_workers 8 --prefetch_factor 4

# Disable multi-worker loading (for debugging)
python train.py --num_workers 0
```

## Performance Tuning

- **num_workers**: Set to the number of CPU cores available. Start with 4 and increase if data loading is still a bottleneck.
- **prefetch_factor**: Number of batches to prefetch per worker. Default is 2, which is usually sufficient.

## Implementation Details

The optimization consists of:

1. `DistributedPaddedIterableDataset`: An IterableDataset that handles distributed file processing and yields individual batches.

2. `OptimizedDistributedPaddedDataLoader`: A wrapper that provides the same interface as the original dataloader but uses PyTorch's DataLoader internally.

3. Proper distributed support: Files are first distributed across GPUs, then across workers within each GPU process.

## Testing

Use the provided test script to compare performance:

```bash
# Test optimized dataloader
python test_optimized_dataloader.py

# Compare with original implementation  
python test_optimized_dataloader.py --compare
``` 