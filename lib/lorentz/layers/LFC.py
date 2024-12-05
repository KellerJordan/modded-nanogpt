import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz


class LorentzFullyConnected(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
