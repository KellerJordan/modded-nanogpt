from dataclasses import dataclass

@dataclass
class FullConfig:
    # Data hyperparams
    data_path: str = "data/fineweb10B"
    input_bin: str = ""
    input_val_bin: str = ""
    num_vocab: int = 50304
    sequence_length: int = 1024
    # Optimization hyperparams
    batch_size: int = 512     # global batch size (across devices)
    device_batch_size: int = 32  # per-device batch size
    num_iterations: int = 1000
    cooldown_frac: float = 0.4
    weight_decay: float = 0
    # Evaluation/logging
    generate_every: int = 0
    train_loss_every: int = 10
    val_loss_every: int = 10
    val_tokens: int = 10_485_760
    save_every: int = 0
    # Model architecture
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 6
    # Rather than specify n_embd directly,
    # you could also define a `head_dim`, if you like;
    # default is zero, bc unless it is set, it is n_embd // n_heads)
    head_dim: int = 0
    n_embd: int = 768
    head_mode: str = "euc"
    attn_mode: str = "euc"
    curvature: float = 1.0
    k_lr: float = 0.0
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """
        Dynamically set up paths and possibly recalculate n_embd from n_heads, head_dim.
        You can also unify any validation logic here.
        """
        # If you want n_embd to be set from n_heads * head_dim:
        if self.head_dim:
            self.n_embd = self.n_heads * self.head_dim

        # Decide how to set the input bins.
        if "tinystories" in self.data_path:
            self.input_bin = f"{self.data_path}/train.bin"
            self.input_val_bin = f"{self.data_path}/val.bin"
        if "shakespeare" in self.data_path:
            self.input_bin = f"{self.data_path}/train.bin"
            self.input_val_bin = f"{self.data_path}/val.bin"
        elif "finewebedu" in self.data_path:
            self.input_bin = f"{self.data_path}/finewebedu_train_*.bin"
            self.input_val_bin = f"{self.data_path}/finewebedu_val_*.bin"
        elif "fineweb" in self.data_path:
            self.input_bin = f"{self.data_path}/fineweb_train_*.bin"
            self.input_val_bin = f"{self.data_path}/fineweb_val_*.bin"
        else:
            raise ValueError("Specify a proper data path (contains 'tinystories' or 'fineweb')")
