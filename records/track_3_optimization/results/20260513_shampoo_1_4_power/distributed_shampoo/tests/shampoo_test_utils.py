"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Callable
from functools import partial
from itertools import pairwise, repeat
from typing import overload

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, fully_shard, FullyShardedDataParallel
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT


class _ModelWithScalarAndLinearAndDeadLayers(nn.Module):
    def __init__(
        self,
        model_linear_layers_dims: tuple[int, ...],
        model_dead_layers_dims: tuple[int, ...] | None,
        enable_learnable_scalar: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Choose torch.tensor over nn.Parameter(requires_grad=False) to avoid model.parameters() including this scalar when enable_learnable_scalar is False.
        self.scalar: torch.Tensor = (
            nn.Parameter(torch.zeros(()))
            if enable_learnable_scalar
            else torch.zeros(())
        )

        # fully_shard doesn't support containers so we fall back to use nn.Sequential
        self.linear_layers: nn.Sequential = nn.Sequential(
            *(nn.Linear(a, b, bias=bias) for a, b in pairwise(model_linear_layers_dims))
        )
        if model_dead_layers_dims is not None:
            self.dead_layers: nn.Sequential = nn.Sequential(
                *(
                    nn.Linear(a, b, bias=False)
                    for a, b in pairwise(model_dead_layers_dims)
                )
            )
            # Initialize dead layers with zeros for the ease of testing if needed
            for m in self.dead_layers:
                assert isinstance(m, nn.Linear)
                nn.init.zeros_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers(x) + self.scalar


@overload
def construct_training_problem(
    model_linear_layers_dims: tuple[int, ...],
    model_dead_layers_dims: tuple[int, ...] | None = (10, 10),
    enable_learnable_scalar: bool = True,
    device: torch.device | None = None,
    bias: bool = False,
    fill: float | tuple[float, ...] = 0.0,
    post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]: ...


@overload
def construct_training_problem(
    model_linear_layers_dims: tuple[int, ...],
    model_dead_layers_dims: tuple[int, ...] | None = (10, 10),
    enable_learnable_scalar: bool = True,
    device: torch.device | None = None,
    bias: bool = False,
    fill: float | tuple[float, ...] = 0.0,
    post_model_decoration: Callable[[nn.Module], FSDPModule] = lambda x: fully_shard(x),
) -> tuple[FSDPModule, nn.Module, torch.Tensor, torch.Tensor]: ...


def construct_training_problem(
    model_linear_layers_dims: tuple[int, ...],
    model_dead_layers_dims: tuple[int, ...] | None = (10, 10),
    enable_learnable_scalar: bool = True,
    device: torch.device | None = None,
    bias: bool = False,
    fill: float | tuple[float, ...] = 0.0,
    post_model_decoration: Callable[[nn.Module], nn.Module | FSDPModule] = lambda x: x,
) -> tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]:
    """
    Constructs a training problem (model, loss, data, target) with the given model dimensions and attributes.

    Args:
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the model linear layers.
        model_dead_layers_dims (tuple[int, ...] | None): The dimensions of the model dead linear layers. (Default: (10, 10))
        enable_learnable_scalar (bool): Whether to enable a learnable scalar multiplier for the input tensor. (Default: True)
        device (torch.device | None): The device to use. (Default: None)
        bias (bool): Whether to use bias in the linear (non-dead) layers. (Default: False)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 0.0)
        post_model_decoration (Callable[[nn.Module], nn.Module | FSDPModule]): A function to apply additional modifications to the model, useful for FullyShardedDataParallel and FSDPModule. (Default: identity function)


    Returns:
        model (nn.Module | FSDPModule): The model as specified from the input arguments.
        loss (nn.Module): The loss function (currently always set to MSE).
        data (torch.Tensor): A randomly generated input tensor corresponding to the input dimension.
        target (torch.Tensor): A target tensor of zeros corresponding to the output dimension.
    """
    seed_value = 42
    torch.manual_seed(seed_value)

    # Note: Generate a random tensor first then moving it to device to guarantee the same tensor value on different devices.
    data = torch.randn(model_linear_layers_dims[0], dtype=torch.float)
    data /= torch.linalg.norm(data)
    data = data.to(device=device)

    model = _ModelWithScalarAndLinearAndDeadLayers(
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layers_dims=model_dead_layers_dims,
        enable_learnable_scalar=enable_learnable_scalar,
        bias=bias,
    ).to(device=device)

    for m, f in zip(
        model.linear_layers,
        repeat(fill, len(model.linear_layers)) if isinstance(fill, float) else fill,
        strict=True,
    ):
        # Directly fills the weight tensor with the value 'f' without tracking in autograd.
        assert isinstance(m, nn.Linear)
        nn.init.constant_(m.weight, f)
        if bias:
            # If bias is used, directly fills the bias tensor with the value 'f' without tracking in autograd.
            nn.init.constant_(m.bias, f)

    loss = nn.MSELoss()

    target = torch.tensor([[0.0] * model_linear_layers_dims[-1]]).to(device=device)

    return post_model_decoration(model), loss, data, target


@overload
def train_model(
    optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    model_factory: Callable[
        [], tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]
    ],
    num_steps: int = 5,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, torch.optim.Optimizer]: ...


@overload
def train_model(
    optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    model_factory: Callable[
        [], tuple[FSDPModule, nn.Module, torch.Tensor, torch.Tensor]
    ],
    num_steps: int = 5,
) -> tuple[
    FSDPModule, nn.Module, torch.Tensor, torch.Tensor, torch.optim.Optimizer
]: ...


def train_model(
    optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    model_factory: Callable[
        [], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]
    ],
    num_steps: int = 5,
) -> tuple[
    nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor, torch.optim.Optimizer
]:
    """
    Trains a model using the specified optimizer and model factories for a given number of steps.

    This function is useful for pre-training a model and optimizer for a specified number of steps.
    Users can use the outputs to continue training or testing the model further.
    This is particularly beneficial when users want to initialize a model with pre-trained weights and optimizer state before fine-tuning or evaluating on a different dataset or task.

    Args:
        optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an optimizer instance.
        model_factory (Callable[[], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]]): A factory function that returns a tuple containing the model, loss function, validation data, and target data.
        num_steps (int): The number of training steps to perform. (Default: 5)

    Returns:
        model (nn.Module | FSDPModule): The trained model.
        loss (nn.Module): The loss function used during training.
        validation_data (torch.Tensor): The input data used for validation.
        target (torch.Tensor): The target data used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """
    # Get model, loss, validation_data, and target from the factory.
    model, loss, validation_data, target = model_factory()
    assert isinstance(model, nn.Module)
    optimizer = optim_factory(model.parameters())

    # Pregenerate num_steps of tensor and put into a PyTorch dataset with seed
    seed_value = 42
    torch.manual_seed(seed_value)

    train_data = torch.randn(num_steps, *validation_data.shape, dtype=torch.float)
    train_data /= torch.linalg.norm(train_data, dim=1, keepdim=True)
    train_data = train_data.to(device=validation_data.device)
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )

    # Set optimizer to train mode if it supports train/eval modes (e.g., GPA-AdamW).
    train_mode = getattr(optimizer, "train", None)
    if callable(train_mode):
        train_mode()

    # Train for the specified number of steps
    for _, (step_data,) in enumerate(train_loader):
        optimizer.zero_grad()
        objective = loss(model(step_data), target)
        objective.backward()
        optimizer.step()

    return model, loss, validation_data.unsqueeze(0), target, optimizer


def compare_two_optimizers_models_devices_on_weight_and_loss(
    control_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    control_model_factory: Callable[
        [], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]
    ],
    experimental_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    experimental_model_factory: Callable[
        [], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]
    ],
    control_and_experimental_same_device: bool = True,
    total_steps: int = 5,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """
    Compare the performance of two optimizers on models across different devices by evaluating their weights and loss.

    Args:
        control_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): Factory function for the control optimizer.
        control_model_factory (Callable[[], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]]): Factory function for the control model.
        experimental_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): Factory function for the experimental optimizer.
        experimental_model_factory (Callable[[], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]]): Factory function for the experimental model.
        control_and_experimental_same_device (bool): Whether both models are on the same device. (Default: True)
        total_steps (int): Number of training steps. (Default: 5)
        rtol (float | None): Relative tolerance for comparing weights and losses. (Default: None)
        atol (float | None): Absolute tolerance for comparing weights and losses. (Default: None)

    Returns:
        None
    """

    def generated_comparable_trained_weights_and_final_loss(
        optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
        model_factory: Callable[
            [], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]
        ],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Trains a model and returns the trained weights and final loss.

        Args:
            optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function to create an optimizer.
            model_factory (Callable[[], tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]]): A factory function to create a model, loss, data, and target.

        Returns:
            trained_weights (list[torch.Tensor]): A list of trained model parameters.
            final_loss (torch.Tensor): The final loss value after training.
        """
        # Using partial here to prevent Pyre complaints on incompatible parameter type.
        model, loss, validation_data, target, _ = partial(
            train_model, model_factory=model_factory
        )(optim_factory=optim_factory, num_steps=total_steps)

        # We only care about model_linear_layers_dim params, not model_dead_layer params.
        assert isinstance(model, nn.Module)
        linear_layers = model.get_submodule("linear_layers")
        match model:
            case FSDPModule():
                # When FullyShard or Hybrid Shard is used, model parameters are DTensors. We obtain the full value of
                # parameters from DTensors.
                trained_weights = []
                for param in linear_layers.parameters():
                    # Need this assertion to pass the type-checking test.
                    assert isinstance(param, DTensor)
                    trained_weights.append(param.full_tensor().view(-1).detach().cpu())
            case FullyShardedDataParallel():
                with FullyShardedDataParallel.summon_full_params(model):
                    trained_weights = [
                        param.view(-1).detach().cpu()
                        for param in linear_layers.parameters()
                    ]
            case _:
                trained_weights = [
                    param.view(-1).detach().cpu()
                    for param in linear_layers.parameters()
                ]

        return trained_weights, loss(model(validation_data), target).detach()

    control_params, control_loss = generated_comparable_trained_weights_and_final_loss(
        optim_factory=control_optim_factory,
        model_factory=control_model_factory,
    )
    experimental_params, experimental_loss = (
        generated_comparable_trained_weights_and_final_loss(
            optim_factory=experimental_optim_factory,
            model_factory=experimental_model_factory,
        )
    )
    torch.testing.assert_close(
        actual=experimental_loss,
        expected=control_loss,
        rtol=rtol,
        atol=atol,
        check_device=control_and_experimental_same_device,
    )
    torch.testing.assert_close(
        actual=experimental_params,
        expected=control_params,
        rtol=rtol,
        atol=atol,
        check_device=control_and_experimental_same_device,
    )


def compare_two_optimizers_devices_on_weight_and_loss(
    control_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    control_device: torch.device | None,
    experimental_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    experimental_device: torch.device | None,
    model_linear_layers_dims: tuple[int, ...] = (10, 1, 1),
    model_dead_layers_dims: tuple[int, ...] | None = None,
    enable_learnable_scalar: bool = True,
    fill: float | tuple[float, ...] = 1.0,
    total_steps: int = 5,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """
    Compare the performance of two optimizers on a simple neural network across different devices.

    Args:
        control_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the control optimizer.
        control_device (torch.device | None): The device to use for the control optimizer.
        experimental_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the experimental optimizer.
        experimental_device (torch.device | None): The device to use for the experimental optimizer.
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the linear layers in the neural network. (Default: (10, 1, 1))
        model_dead_layers_dims (tuple[int, ...] | None): The dimensions of the dead layers in the neural network. (Default: None)
        enable_learnable_scalar (bool): Whether to enable a learnable scalar multiplier for the input tensor. (Default: True)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 1.0)
        total_steps (int): The number of training steps. (Default: 5)
        rtol (float | None): The relative tolerance for comparing weights and losses. (Default: None)
        atol (float | None): The absolute tolerance for comparing weights and losses. (Default: None)

    Returns:
        None
    """

    device_aware_training_problem_factory = partial(
        construct_training_problem,
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layers_dims=model_dead_layers_dims,
        enable_learnable_scalar=enable_learnable_scalar,
        fill=fill,
    )
    compare_two_optimizers_models_devices_on_weight_and_loss(
        control_optim_factory=control_optim_factory,
        control_model_factory=partial(
            device_aware_training_problem_factory, device=control_device
        ),
        experimental_optim_factory=experimental_optim_factory,
        experimental_model_factory=partial(
            device_aware_training_problem_factory, device=experimental_device
        ),
        control_and_experimental_same_device=control_device == experimental_device,
        total_steps=total_steps,
        rtol=rtol,
        atol=atol,
    )


def compare_two_optimizers_on_weight_and_loss(
    control_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    experimental_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    model_linear_layers_dims: tuple[int, ...] = (10, 1, 1),
    model_dead_layers_dims: tuple[int, ...] | None = None,
    enable_learnable_scalar: bool = True,
    device: torch.device | None = None,
    fill: float | tuple[float, ...] = 1.0,
    total_steps: int = 5,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """
    Compare the performance of two optimizers on a simple neural network using the same device.

    Args:
        control_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the control optimizer.
        experimental_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the experimental optimizer.
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the linear layers in the neural network. (Default: (10, 1, 1))
        model_dead_layers_dims (tuple[int, ...] | None): The dimensions of the dead layers in the neural network. (Default: None)
        enable_learnable_scalar (bool): Whether to enable a learnable scalar multiplier for the input tensor. (Default: True)
        device (torch.device | None): The device to use for training. (Default: None)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 1.0)
        total_steps (int): The number of training steps. (Default: 5)
        rtol (float | None): The relative tolerance for comparing weights and losses. (Default: None)
        atol (float | None): The absolute tolerance for comparing weights and losses. (Default: None)

    Returns:
        None
    """
    compare_two_optimizers_devices_on_weight_and_loss(
        control_optim_factory=control_optim_factory,
        control_device=device,
        experimental_optim_factory=experimental_optim_factory,
        experimental_device=device,
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layers_dims=model_dead_layers_dims,
        enable_learnable_scalar=enable_learnable_scalar,
        fill=fill,
        total_steps=total_steps,
        rtol=rtol,
        atol=atol,
    )


def compare_optimizer_on_cpu_and_device(
    optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    device: torch.device,
    model_linear_layers_dims: tuple[int, ...] = (10, 1, 1),
    model_dead_layers_dims: tuple[int, ...] | None = None,
    enable_learnable_scalar: bool = True,
    fill: float | tuple[float, ...] = 1.0,
    total_steps: int = 5,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """
    Compare the performance of the same optimizer on a simple neural network across CPU and another device.

    Args:
        optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the optimizer.
        device (torch.device): The other experimental device to use for the training.
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the linear layers in the neural network. (Default: (10, 1, 1))
        model_dead_layers_dims (tuple[int, ...] | None): The dimensions of the dead layers in the neural network. (Default: None)
        enable_learnable_scalar (bool): Whether to enable a learnable scalar multiplier for the input tensor. (Default: True)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 1.0)
        total_steps (int): The number of training steps. (Default: 5)
        rtol (float | None): The relative tolerance for comparing weights and losses. (Default: None)
        atol (float | None): The absolute tolerance for comparing weights and losses. (Default: None)

    Returns:
        None
    """
    compare_two_optimizers_devices_on_weight_and_loss(
        control_optim_factory=optim_factory,
        control_device=torch.device("cpu"),
        experimental_optim_factory=optim_factory,
        experimental_device=device,
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layers_dims=model_dead_layers_dims,
        enable_learnable_scalar=enable_learnable_scalar,
        fill=fill,
        total_steps=total_steps,
        rtol=rtol,
        atol=atol,
    )
