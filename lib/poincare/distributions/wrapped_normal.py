import torch
from torch.distributions import MultivariateNormal

import math

from lib.geoopt.manifolds.stereographic import PoincareBall
         

class PoincareWrappedNormal(torch.nn.Module):
    """ Implementation of a Hyperbolic Wrapped Normal distribution with diagonal covariance matrix defined by Mathieu et al. (2019).

    - Implemented for use in VAE training.
    - Original source: https://github.com/emilemathieu/pvae
    
    Args:
        mean_H: Mean in hyperbolic space (can be batched)
        var: Diagonal of covariance matrix (can be batched)
    """
    def __init__(self, manifold: PoincareBall):
        super(PoincareWrappedNormal, self).__init__()

        # Save variables
        self.manifold = manifold
        
    def make_covar(self, var: torch.tensor):
        """ Creates covariance matrix and rescales values
        
        Args:
            var: Diagonal of covariance matrix (bs x n) or full covariance matrix (bs x n x n) in Euclidean space
        """
        assert (len(var.shape)==2 or len(var.shape)==3), "Wrong input shapes."

        if len(var.shape)==2:
            var = torch.diag_embed(var)
        covar = var

        return covar


    def rsample(self, mean_H, covar, num_samples=1, keepdim=False, ret_uv=False):
        """ Implements sampling from Wrapped normal distribution using reparametrization trick.

        Some intermediate results are saved to object for efficient log_prob calculation.

        Returns:
            Returns num_samples points for each gaussian (or batch instance)
            -> If num_samples==1: Returns shape (bs x num_features)
            -> If num_samples>1: Returns shape (num_samples x bs x num_features)
        """

        # "1. Sample a vector v_t from the Gaussian distribution N(0,Sigma) defined over R^n"
        v = MultivariateNormal(
                        torch.zeros_like(covar)[..., 0], 
                        covar
                    ).rsample((num_samples,))

        v = v / self.manifold.lambda_x(self.manifold.origin(v.shape[0], v.shape[1], device=v.device), keepdim=True)
        u = self.manifold.transp0(mean_H, v)
        z = self.manifold.expmap(mean_H, u)

        if (num_samples==1) and (not keepdim):
            z = z.squeeze(0)

        if ret_uv:
            return z, u, v
        else:
            return z

    def log_prob(self, z, mean_H, covar, u=None, v=None):
        """ Implements computation of probability densitiy, log likelihood of wrapped normal distribution by Mathieu et al. (2019)

        Args:
            z: Latent embedding in hyperbolic space 
                -> Shape = (num_samples x bs x d+1) or (bs x d+1)
            mean_H: mean in hyperbolic space
            covar: covaricance matrix in Euclidean space
            u,v: Intermediate results from sampling for efficient calculation

        Returns:
            Computation of log_prob.
        """
        n = mean_H.shape[-1] # Dimensionality

        no_mult_samples = len(z.shape)==2

        if no_mult_samples:
            z = z.unsqueeze(0)

        v = self.manifold.logmap(mean_H, z)
        v = self.manifold.transp0back(mean_H, v)
        u = v * self.manifold.lambda_x(self.manifold.origin(v.shape[0], v.shape[1], device=v.device))

        norm_pdf = MultivariateNormal(
                        torch.zeros_like(covar)[..., 0], 
                        covar
                    ).log_prob(u)
        
        # Compute log likelihood
        d = self.manifold.dist(mean_H, z)
        logdetexp = (n - 1) * torch.log(torch.sinh(self.manifold.c.sqrt()*d) / self.manifold.c.sqrt() / d)
    
        logp_z = norm_pdf - logdetexp

        if no_mult_samples:
            logp_z = logp_z.squeeze(0)

        return logp_z