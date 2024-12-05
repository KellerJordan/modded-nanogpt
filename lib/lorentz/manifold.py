import torch

from lib.geoopt import Lorentz
from lib.geoopt.manifolds.lorentz import math


class CustomLorentz(Lorentz):
    def _init__(self, k=1.0, learnable=False):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)

    def sqdist(self, x, y, dim=-1):
        """ Squared Lorentzian distance, as defined in the paper 'Lorentzian Distance Learning for Hyperbolic Representation'"""
        return -2*self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8):
        """ Centroid implementation. Adapted the code from Chen et al. (2022) """
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=-2)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def switch_man(self, x, manifold_in: Lorentz):
        """ Projection between Lorentz manifolds (e.g. change curvature) """
        x = manifold_in.logmap0(x)
        return self.expmap0(x)
    
    def pt_addition(self, x, y):
        """ Parallel transport addition proposed by Chami et al. (2019) """
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)

    #################################################
    #       Reshaping operations
    #################################################
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements flattening operation directly on the manifold. Based on Lorentz Direct Concatenation (Qu et al., 2022) """
        bs,h,w,c = x.shape
        # bs x H x W x C
        time = x.narrow(-1, 0, 1).view(-1, h*w)
        space = x.narrow(-1, 1, x.shape[-1] - 1).flatten(start_dim=1) # concatenate all x_s

        time_rescaled = torch.sqrt(torch.sum(time**2, dim=-1, keepdim=True)+(((h*w)-1)/-self.k))
        x = torch.cat([time_rescaled, space], dim=-1)

        return x

    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        """ Implements reshaping a flat tensor to an image directly on the manifold. Based on Lorentz Direct Split (Qu et al., 2022) """
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2]-1))
        img = self.add_time(space)

        return img


    #################################################
    #       Activation functions
    #################################################
    def lorentz_relu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        """ Implements ReLU activation directly on the manifold. """
        return self.lorentz_activation(x, torch.relu, add_time)

    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        """ Implements activation directly on the manifold. """
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x
    
    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements ReLU activation in tangent space. """
        return self.expmap0(torch.relu(self.logmap0(x)))
