"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import abstractmethod
from dataclasses import dataclass

import torch
from distributed_shampoo.utils.abstract_dataclass import AbstractDataclass
from numpy.polynomial.polynomial import polyval


class CostModel(AbstractDataclass):
    """Abstract base class for computing tensor cost metrics.

    This class defines the interface for cost models that calculate
    resource usage metrics for tensors.
    """

    @abstractmethod
    def cost(self, tensor: torch.Tensor) -> float:
        """Compute cost for a tensor.

        Args:
            tensor (Tensor): The tensor to compute the cost for.

        Returns:
            estimated_cost (float): The computed cost value.
        """


@dataclass
class PolynomialComputationalCostModel(CostModel):
    """Polynomial cost model for computational complexity estimation.

    This model uses a polynomial function to estimate computational costs
    based on tensor dimensions.


    Attributes:
        coefficients (tuple[float, ...]): The coefficients of the polynomial in increasing order of degree.
            For example, (a, b, c) represents a*x^0 + b*x^1 + c*x^2.
        min_cost (float): The minimum cost threshold. Any estimated cost below this will be raised to this value. (Default: 0)

    """

    coefficients: tuple[float, ...]
    min_cost: float = 0

    def cost(self, tensor: torch.Tensor) -> float:
        return sum(
            max(self.min_cost, polyval(x=dim_size, c=self.coefficients))  # type: ignore[misc,call-overload]
            for dim_size in tensor.shape
        )


@dataclass
class AlignedMemoryCostModel(CostModel):
    """Memory cost model with alignment padding.

    Attributes:
        alignment_bytes (int): The number of bytes to align memory to. (Default: 64)

    """

    alignment_bytes: int = 64

    def cost(self, tensor: torch.Tensor) -> float:
        # Calculate raw buffer size (number of elements * bytes per element)
        buffer_size = tensor.numel() * tensor.element_size()

        # Round up to the nearest multiple of alignment_bytes
        aligned_buffer_size = (
            (buffer_size + self.alignment_bytes - 1)
            // self.alignment_bytes
            * self.alignment_bytes
        )
        return aligned_buffer_size


# Default cost is AlignedMemoryCostModel for backward compatibility.
DefaultCostModel = AlignedMemoryCostModel()
