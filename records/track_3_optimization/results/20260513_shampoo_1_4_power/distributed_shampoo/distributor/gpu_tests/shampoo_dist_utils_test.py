"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

from functools import partial
from operator import attrgetter
from unittest import mock

import torch
from distributed_shampoo.distributor import shampoo_dist_utils
from distributed_shampoo.distributor.shampoo_dist_utils import get_device_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class ShampooDistUtilsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _verify_deivce_mesh(self, device_mesh: DeviceMesh) -> None:
        replicate_mesh = device_mesh["replicate"]
        shard_mesh = device_mesh["shard"]

        self.assertEqual(device_mesh.get_group(0), device_mesh.get_group("replicate"))
        self.assertEqual(device_mesh.get_group(1), device_mesh.get_group("shard"))

        self.assertEqual(device_mesh.get_group("shard"), shard_mesh.get_group())
        self.assertEqual(device_mesh.get_group("replicate"), replicate_mesh.get_group())

        self.assertCountEqual(
            device_mesh.get_all_groups(),
            (shard_mesh.get_group(), replicate_mesh.get_group()),
        )

    @with_comms
    def test_get_device_mesh(self) -> None:
        mesh = tuple(
            map(
                # Some type-checkers are not able to recognize the `tuple` below as a function. Use `partial` here to explicitly make a Callable for those type-checkers.
                partial(tuple),
                torch.tensor(range(self.world_size))
                .view(-1, self.world_size // 2)
                .tolist(),
            )
        )

        self._verify_deivce_mesh(
            device_mesh=get_device_mesh(
                device_type=attrgetter("device_type")(self),
                mesh=mesh,
                mesh_dim_names=("replicate", "shard"),
            )
        )

        # Test the caching property of get_device_mesh() by mocking DeviceMesh.__init__().
        # DeviceMesh.__init__() should not be called due to caching, and the output of
        # get_device_mesh() should be the same as the previous one.
        with mock.patch.object(
            shampoo_dist_utils.DeviceMesh,
            "__init__",
        ) as mock_device_mesh_init:
            device_mesh = get_device_mesh(
                device_type=attrgetter("device_type")(self),
                mesh=mesh,
                mesh_dim_names=("replicate", "shard"),
            )

            mock_device_mesh_init.assert_not_called()

        self._verify_deivce_mesh(device_mesh=device_mesh)
