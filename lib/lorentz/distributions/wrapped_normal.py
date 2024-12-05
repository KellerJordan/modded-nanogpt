import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import math

from lib.lorentz.manifold import CustomLorentz

class LorentzWrappedNormal(torch.nn.Module):
    """ Implementation of a Hyperbolic Wrapped Normal distribution with diagonal covariance matrix defined by Nagano et al. (2019).

    - Implemented for use in VAE training.
    - Original source: https://github.com/pfnet-research/hyperbolic_wrapped_distribution
    
    Args:
        mean_H: Mean in hyperbolic space (can be batched)
        var: Diagonal of covariance matrix (can be batched)
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzWrappedNormal, self).__init__()

        # Save variables
        self.manifold = manifold
        
    def make_covar(self, var: torch.tensor, rescale=True):
        """ Creates covariance matrix and rescales values
        
        Args:
            var: Diagonal of covariance matrix (bs x n) or full covariance matrix (bs x n x n) in Euclidean space
        """
        assert (len(var.shape)==2 or len(var.shape)==3), "Wrong input shapes."

        if len(var.shape)==2:
            var = torch.diag_embed(var)
        covar = var
        # Scale down Euclidean variance (otherwise x_t very large, when dimensionality is high)
        if rescale and covar.shape[-1]>6: # Scale when dim>6
            covar = covar*(2.5/math.sqrt(covar.shape[-1]))

        return covar


    def rsample(self, mean_H, covar, num_samples=1, keepdim=False, ret_uv=False):
        """ Implements sampling from Wrapped normal distribution using reparametrization trick.

        Some intermediate results are saved to object for efficient log_prob calculation.

        Returns:
            Returns num_samples points for each gaussian (or batch instance)
            -> If num_samples==1: Returns shape (bs x num_features)
            -> If num_samples>1: Returns shape (num_samples x bs x num_features)
        """

        # "1. Sample a vector v_t from the Gaussian distribution N(0,Sigma) defined over R^n" (Nagano et al., 2019)
        vT = MultivariateNormal(
                        torch.zeros((mean_H.shape[0], mean_H.shape[1]-1), device=covar.device), 
                        covar
                    ).rsample((num_samples,))

        # 2. Interpret vT as an element of tangent space T_(mu_0)H^n subspace of R^(n+1) by rewriting v_t as v=[0,v_t] (Nagano et al., 2019)
        v = F.pad(vT, pad=(1, 0))

        # 3. Parallel transport the vector v from origin to mu in T_(mu)H^n subspace of R^(n+1) along the geodesic from origin to mu
        u = self.manifold.transp0(mean_H, v)
        
        #  4. Map u to z by exponential map
        z = self.manifold.expmap(mean_H, u)

        if (num_samples==1) and (not keepdim):
            z = z.squeeze(0)

        if ret_uv:
            return z, u, v
        else:
            return z

    def log_prob(self, z, mean_H, covar, u=None, v=None):
        """ Implements computation of probability densitiy, log likelihood of wrapped normal distribution by Nagano et al. (2019)

        Args:
            z: Latent embedding in hyperbolic space 
                -> Shape = (num_samples x bs x d+1) or (bs x d+1)
            mean_H: Mean in hyperbolic space (can be batched)
            var: Full covariance matrix (can be batched)
            u,v: Intermediate results from sampling for efficient calculation

        Returns:
            Computation of log_prob.
        """
        nT = mean_H.shape[-1]-1 # Dimensionality of Tangent space

        no_mult_samples = len(z.shape)==2

        if no_mult_samples:
            z = z.unsqueeze(0)

        if (v is None) or (u is None):
            # 1. Map z to tangent space of mean
            u = self.manifold.logmap(mean_H, z)
            # 2. Inverse parallel transport of origin to mean -> is the same as parallel transport from mean to origin
            v = self.manifold.transp0back(mean_H, u)
        vT = v[..., 1:]

        # 3. Calculate log likelihood logp(z)
        # -> Calculate log p(v)
        logp_vT = (MultivariateNormal(
                        torch.zeros_like(covar)[..., 0], 
                        covar
                    ).log_prob(vT))
        # -> Calculate log p(z)
        r = self.manifold.norm(u)
        log_det_proj_mu = (nT-1) * (torch.log(torch.sinh(r))-torch.log(r)) # Logarithm rules for easier computation
        
        # Compute log likelihood
        logp_z = logp_vT - log_det_proj_mu

        if no_mult_samples:
            logp_z = logp_z.squeeze(0)

        return logp_z
