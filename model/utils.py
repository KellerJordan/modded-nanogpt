import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))
    

def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


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
