import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Linear, correction_fn


class VectorizedRouter(nn.Module):
    """
    Router module computes routing weights for tokens over experts.

    Input:
        x: Tensor of shape [bs, seq_len, hidden_size] - input embeddings

    Output:
        routing_weights: Tensor of shape [bs, seq_len, num_experts] - expert routing weights
    """
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Expert embeddings: one embedding vector per expert
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size)
        )
        torch.nn.init.kaiming_uniform_(self.expert_embeddings, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing weights for each token and expert.

        Args:
            x: Tensor of shape (bs, n, h)

        Returns:
            routing_weights: Tensor of shape (bs, n, e)
        """
        dot_product = torch.einsum("bsh,eh->bse", x, self.expert_embeddings)
        routing_weights = F.softmax(dot_product, dim=-1)
        return routing_weights



class MHMoE(nn.Module):
    """Multi-Head Mixture of Experts - fully batched implementation"""
    
    def __init__(self, config):
        super().__init__()
        
        # MH-MoE specific parameters
        self.num_heads = getattr(config, 'moe_num_heads', 4)
        self.num_experts = getattr(config, 'moe_num_experts', 8)
        self.topk = getattr(config, 'topk', 2)
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Calculate intermediate size for experts
        intermediate_dim = correction_fn(config.expansion_ratio, self.head_dim)
        
        # Multi-head projection layers
        self.head_projection = Linear(self.hidden_size, self.hidden_size)
        self.output_projection = Linear(self.hidden_size, self.hidden_size)
        
        init_scaler = 1.0 / (self.head_dim ** 0.5)
        self.router = VectorizedRouter(self.head_dim, self.num_experts)
        
        # Large parameter tensors for all experts across all heads
        self.expert_up_weights = nn.Parameter(
            torch.randn(self.num_experts, self.head_dim, intermediate_dim) * init_scaler
        )
        self.expert_down_weights = nn.Parameter(
            torch.randn(self.num_experts, intermediate_dim, self.head_dim) * init_scaler
        )
        
    def forward(self, x):
        bs, d = x.shape # (bs, d)
        
        x = self.head_projection(x) # (bs, d)
        
        x = x.view(bs, self.num_heads, self.head_dim) # (bs, n, h)
        
        routing_weights = self.router(x) # (bs, n, e)
        
        x = torch.matmul(x.unsqueeze(1), self.expert_up_weights)
        # (bs, 1, n, h) @ (e, h, c) -> (bs, e, n, c)

        x = F.relu(x).square()
        
        x = torch.matmul(x, self.expert_down_weights) # (bs, e, n, h)

        topk = torch.topk(routing_weights, k=self.topk, dim=-1)
        topk_values = topk.values # (bs, n, k)
        topk_indices = topk.indices # (bs, n, k)

        weights = torch.zeros_like(routing_weights)
        weights.scatter_(dim=-1, index=topk_indices, src=topk_values) # (bs, n, e)
        weights_flat = weights.view(-1, self.num_experts)

        load_balance_loss = self.num_experts / bs
        experts_used = (weights_flat > 0).float()
        expert_freq = experts_used.sum(dim=0, keepdim=True) / bs
        expert_freq = expert_freq.unsqueeze(0) # (1, 1, e)
        load_balance_loss *= (expert_freq * routing_weights).sum()
        self.load_balancing_loss = load_balance_loss

        # weights: (bs, n, e), x: (bs, e, n, h)
        # We want to weight x by weights along the expert dimension
        # First, permute x to (bs, n, e, h)
        x = x.permute(0, 2, 1, 3)  # (bs, n, e, h)
        weights = weights.unsqueeze(-1)  # (bs, n, e, 1)
        weighted_output = (weights * x).sum(dim=2)
        # (bs, n, e, 1) * (bs, n, e, h) -> (bs, n, e, h) -> (bs, n, h)

        # Merge heads back to (bs, d)
        y = weighted_output.reshape(bs, -1)
        y = self.output_projection(y)
        return y


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
    # Simple test for MHMoE routing and loss
    import torch
    from types import SimpleNamespace
    
    # Create a simple config
    config = SimpleNamespace()
    config.hidden_size = 64
    config.expansion_ratio = 4
    config.moe_num_heads = 4
    config.moe_num_experts = 8
    config.topk = 2
    
    # Create model and test data
    model = MHMoE(config)
    batch_size = 3
    x = torch.randn(batch_size, config.hidden_size)
    
    print("=== Testing MHMoE ===")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Check routing weights
    with torch.no_grad():
        x_proj = model.head_projection(x)
        x_heads = x_proj.view(batch_size, model.num_heads, model.head_dim)
        routing_weights = model.router(x_heads)
        
        print(f"Routing weights shape: {routing_weights.shape}")
        print(f"Routing weights sum per head: {routing_weights.sum(dim=-1)}")
        
        # Check if routing weights sum to 1 (should be close to 1 due to softmax)
        weights_sum = routing_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), \
            f"Routing weights should sum to 1, got {weights_sum}"
        
        # Check topk selection
        topk_values, topk_indices = torch.topk(routing_weights, k=model.topk, dim=-1)
        print(f"Top-{model.topk} expert indices shape: {topk_indices.shape}")
        print(f"Top-{model.topk} expert values shape: {topk_values.shape}")
        
        # Verify topk values are in descending order
        for i in range(model.topk - 1):
            assert torch.all(topk_values[..., i] >= topk_values[..., i + 1]), \
                "Top-k values should be in descending order"
    
    # Check load balancing loss
    print(f"Load balancing loss: {model.load_balancing_loss.item():.6f}")
    assert hasattr(model, 'load_balancing_loss'), "Model should have load_balancing_loss attribute"
    assert model.load_balancing_loss.item() >= 0, "Load balancing loss should be non-negative"
    
    # Test with different batch sizes
    for bs in [1, 2, 5]:
        x_test = torch.randn(bs, config.hidden_size)
        output_test = model(x_test)
        assert output_test.shape == (bs, config.hidden_size), \
            f"Output shape mismatch for batch size {bs}"
    
    print("✓ All tests passed!")
    print("✓ Routing weights sum to 1")
    print("✓ Top-k selection works correctly")
    print("✓ Load balancing loss computed")
    print("✓ Output shapes are correct")
