"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from contextlib import contextmanager
from functools import cache
from typing import Generator

from torch.autograd import profiler
from torch.distributed.device_mesh import DeviceMesh


@contextmanager
def shampoo_comm_profiler(name: str) -> Generator[None, None, None]:
    """Context manager that profiles communication operations in Shampoo distributors.

    Args:
        name (str): The name to use for profiling (e.g., "ClassName::method_name").

    Example:
        with shampoo_comm_profiler("HybridShardShampooDistributor::all_gather_into_tensor"):
            dist.all_gather_into_tensor(...)

    """
    with profiler.record_function(name):
        yield


@cache
def get_device_mesh(
    device_type: str,
    mesh: tuple[tuple[int, ...], ...] | tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMesh:
    """Returns device mesh from provided device type, mesh, and mesh dim names.
    This function will cache previous meshes according to the input.

    Args:
        device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
        mesh (tuple[tuple[int, ...], ...] | tuple[int, ...]):  A multi-dimensional array describing the layout
                of devices, where the IDs are global IDs of the default process group.
        mesh_dim_names (tuple[str, ...] | None): Names of mesh dimensions. (Default: None)

    Returns:
        device_mesh (DeviceMesh): Device mesh.


    """
    return DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)
