"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from distributed_shampoo.utils.shampoo_utils import (
    prepare_update_param_buffers,
    redistribute_and_update_params,
)
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def generate_param_shapes(num_params: int) -> list[tuple[int, ...]]:
    """Generate parameter shapes for testing.

    For N parameters, we generate the following shapes:
        [(1, 2), (2, 3), (3, 4), ..., (N, N + 1)].
    """
    return [(i, i + 1) for i in range(1, num_params + 1)]


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class RedistributeAndUpdateParamsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_redistribute_and_update_params(self, num_params: int) -> None:
        device_mesh = init_device_mesh("cuda", (4,))
        shapes = generate_param_shapes(num_params)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        update_buffers = prepare_update_param_buffers(dtensor_params, self.world_size)
        self.assertEqual(
            len(update_buffers),
            int(np.ceil(num_params / self.world_size) * self.world_size),
        )
        for i, buffer in enumerate(update_buffers):
            if i < num_params:
                self.assertEqual(buffer.numel(), dtensor_params[i].to_local().numel())
            else:
                self.assertEqual(buffer.numel(), 0)

        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()
        # Fill the locally assigned parameters with the rank as value.
        local_full_params = [
            torch.zeros(s, device="cuda").fill_(rank)
            for i, s in enumerate(shapes)
            if i % self.world_size == rank
        ]
        redistribute_and_update_params(
            dtensor_params, local_full_params, update_buffers, dist_group
        )
        for i, param in enumerate(dtensor_params):
            np.testing.assert_allclose(
                param.to_local().cpu().numpy(), i % self.world_size
            )
