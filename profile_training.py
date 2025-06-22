#!/usr/bin/env python3
"""
Profiling script for train_plm.py
Runs training for 25 steps with default configuration and profiles execution time.
"""

import cProfile
import pstats
import io
import argparse
from train_plm import main, parse_args
import sys


def create_profiling_args():
    """Create arguments for profiling run with 25 steps."""
    # Create a mock argument parser to get default values
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="Synthyra/speedrun_profile_test")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=25)  # Override for profiling
    parser.add_argument("--wandb_project", type=str, default="SpeedrunPLM")
    parser.add_argument("--save_every", type=int, default=25)  # Save at end for profiling
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bugfix", action="store_true", default=False)
    parser.add_argument("--p_attention", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--num_att_tokens", type=int, default=512)
    parser.add_argument("--expansion_ratio", type=float, default=8/3)
    parser.add_argument("--soft_logit_cap", type=float, default=16.0)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--sliding_window_size", type=int, default=512)
    parser.add_argument("--target_token_count", type=int, default=16*1024)
    parser.add_argument("--disable_muon", action="store_true", default=False)
    parser.add_argument("--unet", action="store_true", default=False)
    
    # Parse empty args to get defaults
    args = parser.parse_args([])
    return args


def run_profiled_training():
    """Run the training with profiling enabled."""
    print("Setting up profiling for train_plm.py...")
    print("Configuration: 25 steps, default parameters")
    print("=" * 60)
    
    # Get arguments for profiling
    args = create_profiling_args()
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    print("Starting profiled training run...")
    profiler.enable()
    
    try:
        # Run the main training function
        main(args)
    except Exception as e:
        print(f"Error during training: {e}")
        profiler.disable()
        raise
    else:
        profiler.disable()
        print("Training completed successfully!")
    
    # Generate profiling report
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    
    # Create string buffer for stats
    stats_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_buffer)
    
    # Sort by cumulative time and show top functions
    stats.sort_stats('cumulative')
    stats.print_stats(50)  # Show top 50 functions
    
    # Get the profiling output
    profile_output = stats_buffer.getvalue()
    
    # Print to console
    print(profile_output)
    
    # Save detailed profiling results to file
    with open('training_profile_results.txt', 'w') as f:
        f.write("PROFILING RESULTS FOR TRAIN_PLM.PY\n")
        f.write("Configuration: 25 steps, default parameters\n")
        f.write("=" * 60 + "\n\n")
        f.write("TOP FUNCTIONS BY CUMULATIVE TIME:\n")
        f.write("-" * 40 + "\n")
        f.write(profile_output)
        
        # Also add stats sorted by total time
        f.write("\n\nTOP FUNCTIONS BY TOTAL TIME:\n")
        f.write("-" * 40 + "\n")
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('tottime')
        stats.print_stats(50)
        f.write(stats_buffer.getvalue())
        
        # Add stats filtered for specific modules
        f.write("\n\nMODEL-RELATED FUNCTIONS:\n")
        f.write("-" * 40 + "\n")
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats('model.*')
        f.write(stats_buffer.getvalue())
        
        f.write("\n\nTRAINER-RELATED FUNCTIONS:\n")
        f.write("-" * 40 + "\n")
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats('trainer.*|Trainer.*')
        f.write(stats_buffer.getvalue())
    
    print(f"\nDetailed profiling results saved to: training_profile_results.txt")
    
    # Print summary of top time consumers
    print("\n" + "=" * 60)
    print("SUMMARY OF TOP TIME CONSUMERS:")
    print("=" * 60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Get top 10 functions by cumulative time
    stats_items = list(stats.stats.items())[:10]
    
    print(f"{'Function':<50} {'Cumulative Time (s)':<20} {'Calls':<10}")
    print("-" * 80)
    
    for (filename, line, func_name), (cc, nc, tt, ct, callers) in stats_items:
        func_display = f"{filename.split('/')[-1]}:{func_name}"
        if len(func_display) > 49:
            func_display = func_display[:46] + "..."
        print(f"{func_display:<50} {ct:<20.4f} {cc:<10}")


if __name__ == "__main__":
    print("PyTorch Training Profiler")
    print("This will run train_plm.py for 25 steps with profiling enabled.")
    print("Make sure you have the required dependencies installed.")
    print()
    
    try:
        run_profiled_training()
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to run profiling: {e}")
        sys.exit(1) 