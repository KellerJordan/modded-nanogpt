import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Linear, correction_fn



class MHMoE(nn.Module):
    """Multi-Head Mixture of Experts - fully batched implementation"""
    
    def __init__(self, config):
        super().__init__()
        
        # MH-MoE specific parameters
        self.num_heads = getattr(config, 'moe_num_heads', 4)
        self.num_experts = getattr(config, 'moe_num_experts', 8)
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Calculate intermediate size for experts
        corrected_dim = correction_fn(config.expansion_ratio, self.head_dim)
        
        # Multi-head projection layers
        self.head_projection = Linear(self.hidden_size, self.hidden_size)
        self.output_projection = Linear(self.hidden_size, self.hidden_size)
        
        # Unified gating for all heads - single parameter tensor
        init_scaler = 1.0 / (self.head_dim ** 0.5)
        self.gate_weights = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.num_experts) * init_scaler
        )
        
        # Large parameter tensors for all experts across all heads
        self.expert_up_weights = nn.Parameter(
            torch.randn(self.num_heads, self.num_experts, self.head_dim, corrected_dim) * init_scaler
        )
        self.expert_down_weights = nn.Parameter(
            torch.randn(self.num_heads, self.num_experts, corrected_dim, self.head_dim) * init_scaler
        )
        
    def forward(self, x):
        bs, d = x.shape # (bs, d)
        
        x = self.head_projection(x) # (bs, d)
        
        x_heads = x.view(bs, self.num_heads, self.head_dim) # (bs, n, h)
        
        gate_logits = torch.matmul(x_heads, self.gate_weights) # (bs, n, e)
        
        gates = F.softmax(gate_logits, dim=-1) # (bs, n, e)
        
        all_top2_vals, all_top2_indices = torch.topk(gates, k=2, dim=-1)
        
        # Normalize top-2 values to sum to 1
        all_top2_vals = all_top2_vals / all_top2_vals.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss - vectorized across all heads
        expert_usage = gates.mean(dim=0)  # (n, e)
        ideal_usage = 1.0 / self.num_experts
        balance_loss = ((expert_usage - ideal_usage) ** 2).sum()
        total_balance_loss = balance_loss
        
        # Now compute all expert outputs for all heads at once
        x_expanded = x_heads.unsqueeze(2) # (bs, n, 1, h)
        
        up_output = torch.matmul(x_expanded, self.expert_up_weights) # (bs, n, e, c)
        
        # Apply activation
        activated = F.relu(up_output).square()
        
        all_expert_outputs = torch.matmul(activated, self.expert_down_weights) # (bs, n, e, h)
        
        # Now we need to select the top-2 experts for each head
        # Create indices for gathering
        batch_seq_indices = torch.arange(bs, device=x.device)[:, None, None] # (bs, 1, 1)
        head_indices = torch.arange(self.num_heads, device=x.device)[None, :, None] # (1, n, 1)
        
        # Gather top-2 expert outputs for each head
        # all_expert_outputs: (bs, n, e, c)
        # all_top2_indices: (bs, n, 2)
        selected_expert_outputs = all_expert_outputs[batch_seq_indices, head_indices, all_top2_indices] # (bs, n, 2, h)
        
        # Weight by gate values and sum over the 2 experts
        # all_top2_vals: (bs, n, 2)
        all_top2_vals = all_top2_vals.unsqueeze(-1) # (bs, n, 2, 1)
        weighted_outputs = selected_expert_outputs * all_top2_vals # (bs, n, 2, h)
        head_outputs = weighted_outputs.sum(dim=2)  # (bs, n, h)

        # Reshape back to original dimensions
        output = head_outputs.reshape(bs, self.hidden_size) # (bs, d)
        
        # Final output projection
        output = self.output_projection(output) # (bs, d)
        
        # Store balance loss for potential regularization
        self.balance_loss = total_balance_loss
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        corrected_dim = correction_fn(config.expansion_ratio, config.hidden_size)
        self.up = Linear(config.hidden_size, corrected_dim)
        self.down = Linear(corrected_dim, config.hidden_size)
        self.down.weight.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.relu(self.up(x)).square())


if __name__ == "__main__":
    # py -m model.mlp
    class Config:
        hidden_size = 32
        expansion_ratio = 2
        moe_num_heads = 4
        moe_num_experts = 8

    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MHMoE(config).to(device)
    print("MHMoE instantiated successfully.")

    # Test forward pass with random input   
    x = torch.randn(5, config.hidden_size).to(device)  # batch size 5
    output = model(x)
    print("Forward pass output shape:", output.shape)
    assert output.shape == (5, config.hidden_size), "Output shape mismatch!"

    # Test balance loss is computed
    balance_loss = model.balance_loss
    print("Balance loss:", balance_loss.item())
    assert isinstance(balance_loss, torch.Tensor), "Balance loss is not a tensor!"
    print("All MHMoE tests passed.")

    # Additional checks: gating and selection
    with torch.no_grad():
        # Forward pass to get intermediate tensors
        bs = 3
        x = torch.randn(bs, config.hidden_size)
        x_proj = model.head_projection(x)
        x_heads = x_proj.view(bs, model.num_heads, model.head_dim)
        gate_logits = torch.matmul(x_heads, model.gate_weights)
        gates = torch.softmax(gate_logits, dim=-1)
        all_top2_vals, all_top2_indices = torch.topk(gates, k=2, dim=-1)
        # Check that top-2 gate values sum to 1 for each head
        top2_sum = all_top2_vals.sum(dim=-1)
        assert torch.allclose(top2_sum, torch.ones_like(top2_sum), atol=1e-5), "Top-2 gate values do not sum to 1!"
        print("Top-2 gate values sum to 1 for each head.")
        # Check that selected expert indices are within range
        assert torch.all((all_top2_indices >= 0) & (all_top2_indices < model.num_experts)), "Expert indices out of range!"
        print("Selected expert indices are within valid range.")
    print("Gating and selection checks passed.")