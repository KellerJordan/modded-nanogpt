import argparse
from dataclasses import dataclass

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train GPT model with customizable parameters.")
    
    # Configurable arguments
    parser.add_argument("--data_path", type=str, default="data/fineweb10B")
    parser.add_argument("--num_iterations", type=int, default=4578)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head_mode", type=str, default="euc", help="Set the mode for LM Head")
    parser.add_argument("--attn_mode", type=str, default="euc", help="Set the mode for attention layers")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--k_lr", type=float, default=0.0)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=6,
                       help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=12,
                       help="Number of transformer layers")
    
    return parser.parse_args()

@dataclass
class FullConfig:
    data_path: str
    num_iterations: int
    seed: int
    head_mode: str
    attn_mode: str
    curvature: float
    k_lr: float
    head_dim: int
    n_heads: int
    n_layers: int
    
    def __post_init__(self):
        # Add derived configuration parameters
        self.device_batch_size = 4
        self.sequence_length = 1024
        self.batch_size = 32
        self.val_tokens = 378880
        self.n_layer = 12
        self.n_embd = self.head_dim * self.n_heads
        self.vocab_size = None  # Set later based on tokenizer
        self.cooldown_frac = 0.1
        self.val_loss_every = 200
        self.train_loss_every = 1
        self.generate_every = 1000
        self.save_every = 1000
        
        # Derived paths
        self.input_bin = f"{self.data_path}/train.bin"
        self.input_val_bin = f"{self.data_path}/val.bin" 