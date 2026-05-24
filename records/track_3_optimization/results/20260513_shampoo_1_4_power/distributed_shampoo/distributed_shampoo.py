"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
import operator
from collections.abc import Callable
from dataclasses import asdict, fields, is_dataclass
from types import LambdaType
from typing import Any, overload

import torch
from distributed_shampoo.distributor._shampoo_fully_shard_lossless_distributor import (
    FullyShardLosslessDistributor,
)
from distributed_shampoo.distributor._shampoo_hybrid_shard_lossless_distributor import (
    HybridShardLosslessDistributor,
)
from distributed_shampoo.distributor.shampoo_ddp_distributor import DDPDistributor
from distributed_shampoo.distributor.shampoo_distributor import (
    Distributor,
    DistributorInterface,
)
from distributed_shampoo.distributor.shampoo_fsdp_distributor import FSDPDistributor
from distributed_shampoo.distributor.shampoo_fully_shard_distributor import (
    FullyShardDistributor,
)
from distributed_shampoo.distributor.shampoo_hsdp_distributor import HSDPDistributor
from distributed_shampoo.distributor.shampoo_hybrid_shard_distributor import (
    HybridShardDistributor,
)
from distributed_shampoo.preconditioner.adagrad_preconditioner_list import (
    AdagradPreconditionerList,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigendecompositionConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from distributed_shampoo.preconditioner.sgd_preconditioner_list import (
    SGDPreconditionerList,
)
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import (
    EigendecomposedKLShampooPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    RootInvKLShampooPreconditionerList,
    RootInvShampooPreconditionerList,
)
from distributed_shampoo.preconditioner.sign_descent_preconditioner_list import (
    SignDescentPreconditionerList,
)
from distributed_shampoo.preconditioner.spectral_descent_preconditioner_list import (
    SpectralDescentPreconditionerList,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    BaseShampooPreconditionerConfig,
    BETA3,
    BETAS,
    DDPDistributedConfig,
    DefaultShampooConfig,
    DefaultShampooRuntimeConfig,
    DefaultSingleDeviceDistributedConfig,
    DISTRIBUTED_CONFIG,
    DistributedConfig,
    DISTRIBUTOR,
    EigendecomposedKLShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    EPSILON,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPDistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    GeneralizedPrimalAveragingConfig,
    GRAFTING_CONFIG,
    GRAFTING_PRECONDITIONER_LIST,
    HSDPDistributedConfig,
    HybridShardDistributedConfig,
    ITERATE_AVERAGING_CONFIG,
    IterateAveragingConfig,
    LR,
    LR_CPU_PINNED,
    LR_SUM,
    LR_TENSOR,
    MASKED_BLOCKED_GRADS,
    MASKED_BLOCKED_PARAMS,
    MASKED_FILTERED_GRAD_LIST,
    MASKED_WEIGHT_BUFFER_LIST,
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    PEAK_LR,
    PRECONDITION_FREQUENCY,
    PRECONDITIONER_CONFIG,
    PreconditionerConfig,
    PREVIOUS_GRAD_SELECTOR,
    RMSpropPreconditionerConfig,
    RootInvKLShampooPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    ScheduleFreeConfig,
    SGDPreconditionerConfig,
    SHAMPOO_PRECONDITIONER_LIST,
    ShampooPT2CompileConfig,
    ShampooRuntimeConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    START_PRECONDITIONING_STEP,
    STEP,
    TRAIN_MODE,
    USE_BIAS_CORRECTION,
    USE_PIN_MEMORY,
    WEIGHT_BUFFER,
    WEIGHT_BUFFER_LIST,
    WEIGHT_DECAY,
    WEIGHT_DECAY_TYPE,
    WeightDecayType,
)
from distributed_shampoo.utils.shampoo_state_dict_utils import (
    extract_state_dict_content,
    update_param_state_dict_object,
)
from distributed_shampoo.utils.shampoo_utils import compress_list
from torch.optim.optimizer import Optimizer, ParamsT, StateDict

logger: logging.Logger = logging.getLogger(__name__)


class DistributedShampoo(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm.

    --------
    Features
    --------

    1. Layerwise Grafting: In order to tune Shampoo, we can "graft" a layer-wise learning rate schedule from a previous method
        and apply it to Shampoo. This is performed by taking the norm of the layer-wise step of the grafted method, normalizing
        the Shampoo step, and re-scaling the normalized Shampoo step by the product of the norm of the grafted step + learning rate.

        This may be interpreted as an additional block re-scaling of the entire Shampoo preconditioner.
        This is the key ingredient to making Shampoo work in practice.

        We support the following methods:
            - GraftingType.NONE: Performs no grafting.
            - GraftingType.SGD: Grafts the stochastic gradient method.
            - GraftingType.ADAGRAD: Grafts the Adagrad method.
            - GraftingType.RMSPROP: Grafts the RMSprop method.
            - GraftingType.ADAM: Grafts the Adam method.

        NOTE: These methods do not graft the first-moment component - it is entirely based upon grafting using the
        diagonal preconditioner. If using an exponential moving average of the gradient (or gradient filtering), we
        can set beta1 as the same value from before, and both Shampoo and the grafted method will use the filtered
        gradient.

    2. Blocking for Large-Dimensional Tensors: In order to scale Shampoo to large-dimensional tensors, we block the tensor
        and apply Shampoo to each block. For simplicity, suppose we have a linear layer/matrix parameter, W is a m x n matrix:

                [[w_11 w_12 ... w_1n]
                 [w_21 w_22 ... w_2n]
            W =           :
                 [w_m1 w_m2 ... w_mn]]

        Given a max_preconditioner_dim b > 0, blocks W and applies Shampoo to each block, i.e., if b divides both m, n, then:

                [[W_11 W_12 ... W_1k]
                 [W_21 W_22 ... W_2k]
            W =           :
                 [W_l1 W_l2 ... W_lk]]

        where l = m / b, k = n / b, and apply Shampoo to W_ij which is a b x b matrix. This can be viewed as further blocking
        each block of the Shampoo block-diagonal preconditioner.

        Computational cost = O(b^3)
        Memory cost = 4mn (including root inverse preconditioners)

    3. Distributed Memory and Computation: We support different distributed training setups through the distributed_config option,
        which specifies a configuration specific to that setting.

        - None: Performs serial single-GPU training. Replicates all computation and optimizer states across all
            devices.

        - DDPDistributedConfig: Supports multi-GPU distributed data-parallel training via torch.distributed. Assigns optimizer states
            and computation for each block in a greedy fashion to different workers. Leverages DTensor in order to distribute the
            per-block optimizer states from Shampoo. An AllGather communication is performed in order to synchronize the parameter
            updates applied to all parameter blocks.

            Distributed Training Specific Fields:
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - Only supports homogeneous hardware architectures.

        - FSDPDistributedConfig: Supports multi-GPU fully-sharded data-parallel training via torch.distributed. This option uses
            additional metadata in order to reconstruct valid tensor blocks of the original parameter from the flattened parameter
            representation.

            Distributed Training Specific Fields:
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in FSDP.

        - HSDPDistributedConfig: Supports hierarchical parallelism approach that combines DDP and FSDP to scale up training on large models.
            It works by dividing the model into smaller sub-models, each of which is trained in parallel using data parallelism.
            The gradients from each sub-model are then aggregated and used to update the full model.

            Distributed Training Specific Fields:
                - device_mesh: A 2D device mesh that specifies the layout of the model parallelism and data parallelism.
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in HSDP.
                - Only works with the option sharding_strategy=ShardingStrategy.HYBRID_SHARD.
                - Within data parallelism process groups, only supports homogeneous hardware architectures.

        - FullyShardDistributedConfig: Supports per-parameter FSDP training, a.k.a. FSDP2, or "fully_shard" api in torch.distributed. Please see
            README for more detailed introduction on Shampoo FSDP2.

            Requirements:
                - torch.distributed must be initialized in advance.

        - HybridShardDistributedConfig: Supports hierarchical parallelism approach that combines DDP and FSDP to scale up training on large models
            for FSDP2. Please see README for more detailed introduction.

            Distributed Training Specific Fields:
                - device_mesh: A 2D device mesh that specifies the layout of the model parallelism and data parallelism.
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - Within data parallelism process groups, only supports homogeneous hardware architectures.

    4. PyTorch 2.0 Compile Support: Shampoo supports PyTorch 2.0's compilation feature to speed up model training. This is enabled by
        setting up the shampoo_pt2_compile_config arg for Shampoo PyTorch 2.0 compilation.

        - If shampoo_pt2_compile_config = None, ignores compilation, and Shampoo will run in eager mode.
            Shampoo PT2 eager mode means the optimizer runs on plain python code, there is no graphs and lower level kernels used
            to speed up the optimizer stage; and typically you would expect lower QPS for model training as a result.
            For more details regarding PT2 compilation: https://pytorch.org/get-started/pytorch-2.0/

        - If shampoo_pt2_compile_config is set to ShampooPT2CompileConfig class, Shampoo will run in PT2 mode. Shampoo PT2 mode typically gives
            on par numerics and model quality, plus higher QPS. But due to differences in lower level kernel implementation, model quality on par
            is not always guaranteed. If you see model quality gap, please switch back to Shampoo PT2 eager mode by setting
            shampoo_pt2_compile_config = None.

        Shampoo PT2 compilation can also be customized for the backend and options via ShampooPT2CompileConfig.

    5. Eigenvalue correction (SOAP): We can (approximately) correct the eigenvalues of Shampoo's preconditioner by accumulating a running
        average of the squared gradient in the eigenbasis of Shampoo's preconditioner. This running average (with hyperparameter `betas[1]`) is
        updated every iteration while the eigenbasis of Shampoo's preconditioner is only computed every `precondition_frequency` steps.
        Alternatively, this can be seen as running Adam in the eigenbasis of Shampoo's preconditioner, also known as SOAP.

        When setting `preconditioner_config` as an instance of `EigenvalueCorrectedShampooPreconditionerConfig`, there is typically no need to use learning
        rate grafting from Adam (`grafting_config=None`) and, when they are available, Adam's optimal `lr`, `betas`, and `weight_decay` should be
        a good starting point for further tuning. However, the case of `beta2=1.0`, i.e. an AdaGrad-like accumulation, has not been explored yet.
        Also, in settings where Shampoo would usually graft its learning rate from SGD, grafting might still be beneficial.

    6. Generalized Primal Averaging / Schedule-Free: We incorporate generalized primal averaging or Schedule-Free into the codebase which additionally
        maintains a model evaluation sequence that averages the iterates as well as computes the gradient at the interpolation between the averaged and
        current iterates. This can be viewed as a generalization of the primal averaging formulation of Nesterov momentum.

        To use this, set `iterate_averaging_config` as `GeneralizedPrimalAveragingConfig / ScheduleFreeConfig`.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate. (Default: 1e-2)
        betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            (Default: (0.9, 1.0))
        beta3 (float): Coefficient used for computing running average of gradient only for the current iteration.
            This can be used to replicate a version of NAdam if set appropriately. For example, if beta1 = 0.9, then applying
            beta1 interpolation a second time is equivalent to setting beta3 = 0.9 * 0.9 = 0.81.
            If set to -1.0, will set equal to beta1. (Default: -1.0)
        epsilon (float): Term added to the denominator to improve numerical stability, also known as the damping term. (Default: 1e-12)
        weight_decay (float): Weight decay (L2 penalty). (Default: 0.)
        weight_decay_type (WeightDecayType): Enum for selecting type of weight decay to apply. (Default: WeightDecayType.DECOUPLED)
            Options:
                - L2: Applies weight decay by adding a multiple of the weights to the gradient before preconditioning.
                - DECOUPLED: Applies weight decay by adding a multiple of the weights independent of the preconditioned gradient.
                - CORRECTED: Applies weight decay by adding a multiple of the weights scaled by the learning rate divided by the maximum learning rate,
                    independent of the preconditioned gradient. This was shown to yield nice stability properties for Adam, i.e., a steady-state
                    ||g|| / ||w|| ratio for normalized layers in Defazio (2025). This needs to be further studied for AdaGrad-style methods.
                - INDEPENDENT: Applies weight decay by adding a multiple of the weights divided by the maximum learning rate,
                    independent of the preconditioned gradient. This is a variant of decoupled weight decay that scales by 1 / peak_lr.
        max_preconditioner_dim (int | float): Maximum preconditioner dimension. (Default: 1024)
        precondition_frequency (int): Frequency of updating all components of the preconditioner.
            If this field is an instance of ClassicShampooPreconditionerConfig, this is the update frequency of the root inverse of the preconditioner.
            If this field is an instance of EigenvalueCorrectedShampooPreconditionerConfig, this is the update frequency of the eigenbasis of preconditioner.
            (Default: 1)
        start_preconditioning_step (int): Iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        iterate_averaging_config (IterateAveragingConfig | None): Configuration for iterate averaging. If None, ignores iterate averaging. (Default: None)
            If this field is GeneralizedPrimalAveragingConfig, Shampoo is enhanced with the generalized primal averaging method (see https://arxiv.org/pdf/2512.17131).
            If this field is ScheduleFreeConfig, Shampoo is enhanced with Schedule-Free (see https://arxiv.org/abs/2405.15682).
        grafting_config (PreconditionerConfig | None): Configuration for grafting method. If None, ignores grafting.
            (Default: None)
        use_pin_memory (bool): Whether to use pin memory to remove sync point in memory copy. (Default: False)
        shampoo_pt2_compile_config (ShampooPT2CompileConfig | None): Configuration for Shampoo PT2 compilation. If None,
            ignores compilation, and Shampoo will run in eager mode. (Default: None)
        distributed_config (DistributedConfig): Configuration for applying Shampoo to different distributed training frameworks, such as distributed-data parallel (DDP) training.
            (Default: DefaultSingleDeviceDistributedConfig)
        preconditioner_config (PreconditionerConfig): Configuration for preconditioner computation.
            If this field is an instance of ClassicShampooPreconditionerConfig, Shampoo uses the root inverse of the preconditioner.
            If this field is an instance of EigenvalueCorrectedShampooPreconditionerConfig, Shampoo uses the corrected eigenvalues/running Adam in the eigenbasis of preconditioner.
            (Default: DefaultShampooConfig)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 1.0),
        beta3: float = -1.0,
        epsilon: float = 1e-12,
        weight_decay: float = 0.0,
        weight_decay_type: WeightDecayType = WeightDecayType.DECOUPLED,
        max_preconditioner_dim: int | float = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        use_bias_correction: bool = True,
        iterate_averaging_config: IterateAveragingConfig | None = None,
        grafting_config: PreconditionerConfig | None = None,
        use_pin_memory: bool = False,
        shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = None,
        distributed_config: DistributedConfig = DefaultSingleDeviceDistributedConfig,
        preconditioner_config: PreconditionerConfig = DefaultShampooConfig,
        shampoo_runtime_config: ShampooRuntimeConfig = DefaultShampooRuntimeConfig,
    ) -> None:
        super().__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                BETA3: beta3,
                EPSILON: epsilon,
                WEIGHT_DECAY: weight_decay,
                PEAK_LR: lr,
                WEIGHT_DECAY_TYPE: weight_decay_type,
                MAX_PRECONDITIONER_DIM: max_preconditioner_dim,
                PRECONDITION_FREQUENCY: precondition_frequency,
                START_PRECONDITIONING_STEP: start_preconditioning_step,
                USE_BIAS_CORRECTION: use_bias_correction,
                ITERATE_AVERAGING_CONFIG: iterate_averaging_config,
                GRAFTING_CONFIG: grafting_config,
                USE_PIN_MEMORY: use_pin_memory,
                DISTRIBUTED_CONFIG: distributed_config,
                PRECONDITIONER_CONFIG: preconditioner_config,
            },
        )
        self.register_state_dict_post_hook(self._post_state_dict_hook)
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)
        self.register_load_state_dict_post_hook(self._post_load_state_dict_hook)

        def param_group_hyperparameter_check(param_group: dict[str, Any]) -> None:
            if not param_group[LR] >= 0.0:
                raise ValueError(f"Invalid {param_group[LR]=}. Must be >= 0.0.")
            if not 0.0 <= param_group[BETAS][0] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETAS][0]=}. Must be in [0.0, 1.0)."
                )
            if not 0.0 <= param_group[BETAS][1] <= 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETAS][1]=}. Must be in [0.0, 1.0]."
                )
            if param_group[BETA3] == -1.0:
                param_group[BETA3] = param_group[BETAS][0]
            elif not 0.0 <= param_group[BETA3] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETA3]=}. Must be in [0.0, 1.0)."
                )
            if (
                isinstance(
                    param_group[PRECONDITIONER_CONFIG],
                    BaseShampooPreconditionerConfig,
                )
                and isinstance(
                    param_group[PRECONDITIONER_CONFIG].amortized_computation_config,
                    EigendecompositionConfig,
                )
                and isinstance(
                    param_group[
                        PRECONDITIONER_CONFIG
                    ].amortized_computation_config.rank_deficient_stability_config,
                    PseudoInverseConfig,
                )
            ):
                if param_group[EPSILON] != 0.0:
                    raise ValueError(
                        f"Invalid {param_group[EPSILON]=}. Must be == 0.0 when PseudoInverseConfig is used."
                    )
            elif not param_group[EPSILON] > 0.0:
                raise ValueError(f"Invalid {param_group[EPSILON]=}. Must be > 0.0.")
            if not param_group[WEIGHT_DECAY] >= 0.0:
                raise ValueError(
                    f"Invalid {param_group[WEIGHT_DECAY]=}. Must be >= 0.0."
                )
            if (
                param_group[WEIGHT_DECAY_TYPE]
                in (
                    WeightDecayType.CORRECTED,
                    WeightDecayType.INDEPENDENT,
                )
                and not param_group[PEAK_LR] > 0.0
            ):
                raise ValueError(
                    f"Invalid {param_group[PEAK_LR]=}. Must be > 0.0 when using WeightDecayType.CORRECTED or WeightDecayType.INDEPENDENT."
                )
            if (
                isinstance(param_group[MAX_PRECONDITIONER_DIM], float)
                and param_group[MAX_PRECONDITIONER_DIM] != math.inf
            ):
                raise ValueError(
                    f"Invalid {param_group[MAX_PRECONDITIONER_DIM]=}. Must be an integer or math.inf."
                )
            if not param_group[MAX_PRECONDITIONER_DIM] >= 1:
                raise ValueError(
                    f"Invalid {param_group[MAX_PRECONDITIONER_DIM]=}. Must be >= 1."
                )
            if not param_group[PRECONDITION_FREQUENCY] >= 1:
                raise ValueError(
                    f"Invalid {param_group[PRECONDITION_FREQUENCY]=}. Must be >= 1."
                )
            if not param_group[START_PRECONDITIONING_STEP] >= -1:
                raise ValueError(
                    f"Invalid {param_group[START_PRECONDITIONING_STEP]=}. Must be >= -1."
                )

            if isinstance(
                param_group[PRECONDITIONER_CONFIG],
                (SignDescentPreconditionerConfig, SpectralDescentPreconditionerConfig),
            ):
                preconditioner_config_name = param_group[
                    PRECONDITIONER_CONFIG
                ].__class__.__name__
                # Warn about hyperparameters that won't have any effect.
                logger.warning(
                    f"{param_group[BETAS][1]=} does not have any effect when {preconditioner_config_name} is used."
                )
                logger.warning(
                    f"{param_group[EPSILON]=} does not have any effect when {preconditioner_config_name} is used."
                )
                logger.warning(
                    f"{param_group[PRECONDITION_FREQUENCY]=} does not have any effect when {preconditioner_config_name} is used. Setting precondition_frequency to 1..."
                )
                param_group[PRECONDITION_FREQUENCY] = 1

            if (
                isinstance(
                    param_group[PRECONDITIONER_CONFIG],
                    SpectralDescentPreconditionerConfig,
                )
                and param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality != 2
            ):
                logger.warning(
                    f"{param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=} is not equal to 2. Setting target_parameter_dimensionality to 2..."
                )
                param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality = 2

            # Provide warning/error for start_preconditioning_step.
            if param_group[START_PRECONDITIONING_STEP] == -1:
                param_group[START_PRECONDITIONING_STEP] = param_group[
                    PRECONDITION_FREQUENCY
                ]
                logger.warning(
                    f"start_preconditioning_step set to -1. Setting start_preconditioning_step equal to {param_group[PRECONDITION_FREQUENCY]=} by default."
                )
            if (
                param_group[START_PRECONDITIONING_STEP]
                < param_group[PRECONDITION_FREQUENCY]
            ):
                raise ValueError(
                    f"Invalid {param_group[START_PRECONDITIONING_STEP]=}. Must be >= {param_group[PRECONDITION_FREQUENCY]=}."
                )

        # Perform per param_group hyperparameter checks.
        for i, param_group in enumerate(self.param_groups):
            logger.info(f"Checking param_group {i} hyperparameters...")
            param_group_hyperparameter_check(param_group=param_group)

        # Initialize non-group-related fields.
        self._shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = (
            shampoo_pt2_compile_config
        )
        self._runtime_config: ShampooRuntimeConfig = shampoo_runtime_config

        # Initialize list containing group state dictionaries.
        self._per_group_state_lists: list[dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        # NOTE: _instantiate_distributor() has to be called first and _initialize_blocked_parameters_state() second.
        self._instantiate_distributor()
        self._initialize_blocked_parameters_state()
        self._instantiate_shampoo_preconditioner_list()
        self._instantiate_grafting()
        self._instantiate_steps()
        self._instantiate_lr_tensors()
        self._instantiate_filtered_grads()
        self._instantiate_iterate_averaging()
        self._instantiate_per_group_step(
            shampoo_pt2_compile_config=shampoo_pt2_compile_config
        )

    @torch.no_grad()
    def _instantiate_distributor(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if not group[PARAMS]:
                raise ValueError(f"Shampoo got an empty parameter {group=}")

            match group[DISTRIBUTED_CONFIG]:
                case SingleDeviceDistributedConfig():
                    distributor_cls: type[DistributorInterface] = Distributor
                case HSDPDistributedConfig():
                    distributor_cls = HSDPDistributor
                case HybridShardDistributedConfig(
                    param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
                ):
                    distributor_cls = HybridShardDistributor
                case (
                    HybridShardDistributedConfig(
                        param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
                    )
                    | HybridShardDistributedConfig(
                        param_assignment_strategy=FSDPParamAssignmentStrategy.ROUND_ROBIN
                    )
                ):
                    distributor_cls = HybridShardLosslessDistributor
                case DDPDistributedConfig():
                    distributor_cls = DDPDistributor
                case FSDPDistributedConfig():
                    distributor_cls = FSDPDistributor
                case FullyShardDistributedConfig(
                    param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
                ):
                    distributor_cls = FullyShardDistributor
                case (
                    FullyShardDistributedConfig(
                        param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
                    )
                    | FullyShardDistributedConfig(
                        param_assignment_strategy=FSDPParamAssignmentStrategy.ROUND_ROBIN
                    )
                ):
                    distributor_cls = FullyShardLosslessDistributor
                case _:
                    raise NotImplementedError(
                        f"{group[DISTRIBUTED_CONFIG]=} not supported!"
                    )

            # Instantiate distributors for each group.
            state_lists[DISTRIBUTOR] = distributor_cls(group, self._runtime_config)

            if not state_lists[DISTRIBUTOR].local_blocked_params:
                # If the number of trainers is more than the number of blocks,
                # some workers may not receive any parameter blocks which can waste resources.
                logger.warning(
                    "There is no params assigned to the current rank. "
                    f"In DDP Shampoo, this may happen when the value of num_trainers_per_group field in {group[DISTRIBUTED_CONFIG]=} "
                    "is more than the number of global blocked params so some trainers get assigned no blocked params. "
                    "This will cause idle ranks. Please check the num_trainers_per_group setting and consider reducing it. "
                    "Similarly, in FSDP lossless Shampoo, this can happen under ROUND_ROBIN assignment strategy and "
                    "when the optimizer shard group size is larger than the number of params."
                )

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _initialize_blocked_parameters_state(self) -> None:
        for state_lists in self._per_group_state_lists:
            # NOTE: We need to initialize the optimizer states within the optimizer's state dictionary.
            for block_info in state_lists[DISTRIBUTOR].local_block_info_list:
                param_state = self.state[block_info.param]
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) not in param_state, (
                    "There should not exist any optimizer state yet. Maybe verify that _instantiate_distributor was called before all other instantiation functions."
                )
                param_state[block_index] = {}

    @torch.no_grad()
    def _preconditioner_config_to_list_cls(
        self,
        state_lists: dict[str, Any],
        group: dict[str, Any],
        preconditioner_config: PreconditionerConfig,
    ) -> PreconditionerList | None:
        match preconditioner_config:
            case None:
                return None
            case SGDPreconditionerConfig():
                return SGDPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                )
            case (
                AdaGradPreconditionerConfig()
                | RMSpropPreconditionerConfig()
                | AdamPreconditionerConfig()
            ):
                # Order of cases matters: check more specific (derived) classes first,
                # base classes last. Otherwise, a base class pattern will match derived
                # class instances due to structural pattern matching behavior.
                match preconditioner_config:
                    case RMSpropPreconditionerConfig(
                        beta2=_beta2,
                        drop_weighting_factor_on_gsquare=True,
                    ):
                        # Also matches AdamPreconditionerConfig with flag=True due to inheritance
                        beta2 = _beta2
                        weighting_factor: float = 1.0
                        use_bias_correction: bool = False
                    case AdamPreconditionerConfig(beta2=_beta2):
                        beta2 = _beta2
                        weighting_factor = 1 - beta2
                        use_bias_correction = True
                    case RMSpropPreconditionerConfig(beta2=_beta2):
                        beta2 = _beta2
                        weighting_factor = 1 - beta2
                        use_bias_correction = False
                    case AdaGradPreconditionerConfig():
                        beta2 = 1.0
                        weighting_factor = 1.0
                        use_bias_correction = False
                    case _:
                        raise AssertionError(
                            f"Unexpected preconditioner config: {preconditioner_config}"
                        )
                return AdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].local_block_info_list,
                    beta2=beta2,
                    weighting_factor=weighting_factor,
                    epsilon=preconditioner_config.epsilon,
                    use_bias_correction=use_bias_correction,
                )
            case (
                RootInvShampooPreconditionerConfig()
                | EigendecomposedShampooPreconditionerConfig()
                | EigenvalueCorrectedShampooPreconditionerConfig()
                | RootInvKLShampooPreconditionerConfig()
                | EigendecomposedKLShampooPreconditionerConfig()
            ):
                preconditioner_config_to_list_cls: dict[
                    type[PreconditionerConfig], Callable[..., PreconditionerList]
                ] = {
                    RootInvShampooPreconditionerConfig: RootInvShampooPreconditionerList,
                    EigendecomposedShampooPreconditionerConfig: EigendecomposedShampooPreconditionerList,
                    EigenvalueCorrectedShampooPreconditionerConfig: EigenvalueCorrectedShampooPreconditionerList,
                    RootInvKLShampooPreconditionerConfig: RootInvKLShampooPreconditionerList,
                    EigendecomposedKLShampooPreconditionerConfig: EigendecomposedKLShampooPreconditionerList,
                }
                beta2 = group[BETAS][1]
                return preconditioner_config_to_list_cls[type(preconditioner_config)](
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=preconditioner_config,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].local_block_info_list,
                    beta2=beta2,
                    weighting_factor=1.0
                    if preconditioner_config.drop_weighting_factor_on_gsquare
                    or beta2 == 1.0
                    else 1 - beta2,
                    epsilon=group[EPSILON],
                    use_bias_correction=group[USE_BIAS_CORRECTION],
                )
            case SignDescentPreconditionerConfig():
                return SignDescentPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=preconditioner_config,
                )
            case SpectralDescentPreconditionerConfig():
                assert group[DISTRIBUTED_CONFIG].target_parameter_dimensionality == 2, (
                    f"{group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=} must be 2 when using SpectralDescentPreconditionerConfig."
                )
                return SpectralDescentPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=preconditioner_config,
                )
            case _:
                raise NotImplementedError(f"{preconditioner_config=} not supported!")

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            assert group[PRECONDITIONER_CONFIG] is not None, (
                f"{group[PRECONDITIONER_CONFIG]=} is None. Please check the instantiation of DistributedShampoo."
            )
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = (
                self._preconditioner_config_to_list_cls(
                    state_lists=state_lists,
                    group=group,
                    preconditioner_config=group[PRECONDITIONER_CONFIG],
                )
            )

    @torch.no_grad()
    def _instantiate_grafting(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            state_lists[GRAFTING_PRECONDITIONER_LIST] = (
                self._preconditioner_config_to_list_cls(
                    state_lists=state_lists,
                    group=group,
                    preconditioner_config=group[GRAFTING_CONFIG],
                )
            )

    @torch.no_grad()
    def _instantiate_steps(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # NOTE: We instantiate a single step tensor on CPU for each group in order
            #       to track the number of steps taken by all parameters within the group.
            #       Instantiating on CPU avoids GPU synchronization.
            state_lists[STEP] = torch.tensor(0, dtype=torch.int64, device="cpu")

            # In order to ensure that the step counter is checkpointed correctly, we store it
            # as a tensor (which is replicated across all devices) under the first parameter's state.
            self.state[group[PARAMS][0]][STEP] = state_lists[STEP]

    @torch.no_grad()
    def _instantiate_lr_tensors(self) -> None:
        """Pre-allocate persistent lr tensors to avoid per-step pinned memory allocation."""
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            device = group[PARAMS][0].device
            # Pre-allocate a pinned CPU staging tensor and a GPU lr tensor.
            # This avoids cudaHostAlloc + H2D copy every step.
            state_lists[LR_TENSOR] = torch.empty((), dtype=torch.float, device=device)
            state_lists[LR_CPU_PINNED] = torch.empty(
                (), dtype=torch.float, pin_memory=group[USE_PIN_MEMORY]
            )

    @torch.no_grad()
    def _instantiate_filtered_grads(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[BETAS][0] == 0.0:
                continue

            # Construct local filtered gradient list.
            local_filtered_grad_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].local_blocked_params,
                state_lists[DISTRIBUTOR].local_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[block_info.param], (
                    f"{block_index=} not found in {self.state[block_info.param]=}. "
                    "Please check the initialization of self.state[block_info.param][block_index] "
                    "within _initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                    "within Distributor for the correctness of block_index."
                )
                block_state = self.state[block_info.param][block_index]

                block_state[FILTERED_GRAD] = block_info.allocate_zeros_tensor(
                    size=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                local_filtered_grad_list.append(
                    block_info.get_tensor(block_state[FILTERED_GRAD])
                )

            state_lists[FILTERED_GRAD_LIST] = local_filtered_grad_list
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_iterate_averaging(self) -> None:
        # NOTE: Since we are using the memory-efficient implementation of GPA and Schedule-Free, when iterate
        # averaging is enabled, we instantiate a weight buffer (block_state[WEIGHT_BUFFER]) that stores each
        # parameter's "z" sequence. The current parameters are instead treated as the "y" sequence where the
        # gradient is computed.
        #
        # To use the "x" sequence, one must enable the eval mode for the optimizer, which switches the "y"
        # sequence to the "x" sequence. Train mode switches from the "x" sequence to the "y" sequence.
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[ITERATE_AVERAGING_CONFIG] is None:
                continue

            # Construct local weight buffer list.
            local_weight_buffer_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].local_blocked_params,
                state_lists[DISTRIBUTOR].local_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[block_info.param], (
                    f"{block_index=} not found in {self.state[block_info.param]=}. "
                    "Please check the initialization of self.state[block_info.param][block_index] "
                    "within _initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                    "within Distributor for the correctness of block_index."
                )
                block_state = self.state[block_info.param][block_index]

                block_state[WEIGHT_BUFFER] = block_info.allocate_zeros_tensor(
                    size=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                # Get the local tensor from the DTensor and copy into it.
                local_weight_buffer = block_info.get_tensor(block_state[WEIGHT_BUFFER])
                local_weight_buffer.copy_(block)
                local_weight_buffer_list.append(local_weight_buffer)

            state_lists[WEIGHT_BUFFER_LIST] = local_weight_buffer_list
            # Here, we set masked weight buffer list to weight buffer list because we assume
            # all parameters are active.
            state_lists[MASKED_WEIGHT_BUFFER_LIST] = state_lists[WEIGHT_BUFFER_LIST]

            # Instantiate summed learning rate tensor used by iterate averaging
            # to track the cumulative learning rate across steps.
            state_lists[LR_SUM] = torch.tensor(
                0.0,
                dtype=torch.float,
                device=group[PARAMS][0].device,
            )

            # Instantiate a single boolean tensor on CPU for each group in order
            # to track the train and eval mode of each parameter group in the optimizer.
            # This is used by iterate averaging to determine whether to apply train or eval updates.
            state_lists[TRAIN_MODE] = torch.tensor(True, dtype=torch.bool, device="cpu")

            # In order to ensure that the train mode and learning rate sum are checkpointed correctly,
            # we store it as a tensor (which is replicated across all devices) under the first parameter's state.
            self.state[group[PARAMS][0]][TRAIN_MODE] = state_lists[TRAIN_MODE]
            self.state[group[PARAMS][0]][LR_SUM] = state_lists[LR_SUM]

    @torch.no_grad()
    def _instantiate_per_group_step(
        self, shampoo_pt2_compile_config: ShampooPT2CompileConfig | None
    ) -> None:
        # Use PT2 to compile the step function for each parameter group.
        self._per_group_step: Callable[..., None] = (
            torch.compile(
                self._per_group_step_impl, **asdict(shampoo_pt2_compile_config)
            )
            if shampoo_pt2_compile_config is not None
            else self._per_group_step_impl
        )
        if shampoo_pt2_compile_config is not None:
            logger.info(
                f"DistributedShampoo optimizer initialization is using {shampoo_pt2_compile_config=}"
            )

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(
        state_lists: dict[str, Any],
        group: dict[str, Any],
        shampoo_pt2_enabled: bool = False,
    ) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

        # Warning for potential PT2 recompile due to gradient selector change.
        # This warning is expected in either training from scratch or reloading from a checkpoint, as state_lists[PREVIOUS_GRAD_SELECTOR] is initialized to `None`, triggering this warning.
        if state_lists[PREVIOUS_GRAD_SELECTOR] is not None and shampoo_pt2_enabled:
            grad_selector_different = [
                a ^ b
                for a, b in zip(
                    state_lists[DISTRIBUTOR].local_grad_selector,
                    state_lists[PREVIOUS_GRAD_SELECTOR],
                    strict=True,
                )
            ]
            mismatch_grad_selector_indices = [
                i
                for i, is_grad_selector_different in enumerate(grad_selector_different)
                if is_grad_selector_different
            ]
            logger.warning(
                f"""PT2 will recompile because the gradient selection of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model.
                Details:
                - Current step: {state_lists[STEP].item()}
                - Changed gradient selector indices: {mismatch_grad_selector_indices}"""
            )

        # Updates masked state lists if previous block selector disagrees with current selector.
        # State list compression is necessary in order to avoid handling gradients with None.
        state_lists[PREVIOUS_GRAD_SELECTOR] = state_lists[
            DISTRIBUTOR
        ].local_grad_selector
        state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
            DISTRIBUTOR
        ].local_masked_blocked_params
        state_lists[SHAMPOO_PRECONDITIONER_LIST].compress_preconditioner_list(
            local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
        )
        if group[GRAFTING_CONFIG] is not None:
            state_lists[GRAFTING_PRECONDITIONER_LIST].compress_preconditioner_list(
                local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[BETAS][0] != 0.0:
            state_lists[MASKED_FILTERED_GRAD_LIST] = compress_list(
                state_lists[FILTERED_GRAD_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[ITERATE_AVERAGING_CONFIG] is not None:
            state_lists[MASKED_WEIGHT_BUFFER_LIST] = compress_list(
                state_lists[WEIGHT_BUFFER_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )

    @torch.no_grad()
    @torch.compiler.disable
    def _precondition_and_grafting(
        self,
        state_lists: dict[str, Any],
        masked_filtered_grad_list: tuple[torch.Tensor, ...],
        use_grafting_method: bool,
        grafting_config_not_none: bool,
    ) -> tuple[torch.Tensor, ...]:
        # Precondition gradients.
        # If the step count is less than start_preconditioning_step, then we use the grafting method.
        # Assumes that the step state is consistent across all parameters.
        if use_grafting_method:
            masked_blocked_search_directions = state_lists[
                GRAFTING_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

        # Otherwise, we use Shampoo.
        else:
            masked_blocked_search_directions = state_lists[
                SHAMPOO_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

            # Apply grafting.
            if grafting_config_not_none:
                grafting_norm_list = torch._foreach_norm(
                    state_lists[GRAFTING_PRECONDITIONER_LIST].precondition(
                        masked_grad_list=masked_filtered_grad_list,
                    )
                )
                shampoo_norm_list = torch._foreach_norm(
                    masked_blocked_search_directions
                )
                torch._foreach_add_(shampoo_norm_list, 1e-16)
                torch._foreach_div_(grafting_norm_list, shampoo_norm_list)
                torch._foreach_mul_(
                    masked_blocked_search_directions, grafting_norm_list
                )

        return masked_blocked_search_directions

    @torch.no_grad()
    def _add_l2_regularization(
        self,
        state_lists: dict[str, Any],
        weight_decay: float,
        weight_decay_type: WeightDecayType,
    ) -> None:
        # Add L2 regularization / weight decay to the gradient if L2 weight decay is applied.
        if weight_decay != 0.0 and weight_decay_type == WeightDecayType.L2:
            torch._foreach_add_(
                state_lists[MASKED_BLOCKED_GRADS],
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

    @torch.no_grad()
    def _update_preconditioners(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        perform_amortized_computation: bool,
        grafting_config_not_none: bool,
    ) -> None:
        # Update Shampoo and grafting preconditioners.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
            perform_amortized_computation=perform_amortized_computation,
        )
        if grafting_config_not_none:
            state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )

    @torch.no_grad()
    def _compute_filtered_grad_list(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        beta1: float,
        beta3: float,
        use_bias_correction: bool,
    ) -> tuple[torch.Tensor, ...]:
        if beta1 != 0.0:
            # Computes filtered gradient or EMA of the gradients with respect to beta3 if beta3 != beta1.
            masked_filtered_grad_list = (
                torch._foreach_lerp(
                    state_lists[MASKED_FILTERED_GRAD_LIST],
                    state_lists[MASKED_BLOCKED_GRADS],
                    weight=1 - beta3,
                )
                if beta3 != beta1
                else state_lists[MASKED_FILTERED_GRAD_LIST]
            )

            # Update EMA of the gradients (with respect to beta1).
            torch._foreach_lerp_(
                state_lists[MASKED_FILTERED_GRAD_LIST],
                state_lists[MASKED_BLOCKED_GRADS],
                weight=1 - beta1,
            )

            # Apply bias correction if necessary.
            if use_bias_correction:
                bias_correction1 = 1.0 - beta3 * beta1 ** (step - 1)
                masked_filtered_grad_list = torch._foreach_div(
                    masked_filtered_grad_list,
                    bias_correction1,
                )
        else:
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]

        return masked_filtered_grad_list

    @torch.no_grad()
    def _apply_decoupled_or_corrected_weight_decay(
        self,
        state_lists: dict[str, Any],
        masked_blocked_search_directions: tuple[torch.Tensor, ...],
        lr: torch.Tensor,
        weight_decay: float,
        peak_lr: float,
        weight_decay_type: WeightDecayType,
    ) -> None:
        if weight_decay != 0.0 and weight_decay_type in (
            WeightDecayType.DECOUPLED,
            WeightDecayType.CORRECTED,
            WeightDecayType.INDEPENDENT,
        ):
            match weight_decay_type:
                case WeightDecayType.DECOUPLED:
                    alpha = weight_decay
                case WeightDecayType.CORRECTED:
                    alpha = weight_decay * lr.item() / peak_lr
                case WeightDecayType.INDEPENDENT:
                    alpha = weight_decay / peak_lr
                case _:
                    raise ValueError(
                        f"Invalid weight decay type: {weight_decay_type=}!"
                    )
            torch._foreach_add_(
                masked_blocked_search_directions,
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=alpha,
            )

    @staticmethod
    @torch.no_grad()
    def _get_train_and_eval_interp_coeffs(
        iterate_averaging_config: IterateAveragingConfig | None,
        lr: torch.Tensor | None,
        state_lists: dict[str, Any],
    ) -> tuple[float, float]:
        if iterate_averaging_config is None:
            return 0.0, 0.0

        match iterate_averaging_config:
            case GeneralizedPrimalAveragingConfig():
                train_interp_coeff = iterate_averaging_config.train_interp_coeff
                eval_interp_coeff = iterate_averaging_config.eval_interp_coeff
            case ScheduleFreeConfig():
                train_interp_coeff = iterate_averaging_config.train_interp_coeff
                if lr is not None:
                    # Based on Equation (23) in The Road Less Scheduled (https://arxiv.org/pdf/2405.15682).
                    # In the simplest case, computes eval_interp_coeff = 1 - gamma_t^2 / \sum_{i=1}^t gamma_i^2.
                    #
                    # Note that the discrepancy in notation from the paper stems the Schedule-Free update:
                    #   x_{t + 1} = (t / t + 1) * x_t + (1 / t + 1) * z_{t + 1},
                    # whereas GPA uses the update:
                    #   x_{t + 1} = mu_x * x_t + (1 - mu_x) * z_{t + 1}.
                    lr_sum = state_lists[LR_SUM]
                    lr_power = torch.pow(
                        lr, iterate_averaging_config.eval_coeff_lr_power
                    )
                    lr_sum += lr_power
                    eval_interp_coeff = 1.0 - (lr_power / lr_sum).item()
                else:
                    logger.warning(
                        "lr or lr_sum not provided to _get_train_and_eval_interp_coeffs; returning eval_interp_coeff = 0.0!"
                    )
                    eval_interp_coeff = 0.0
            case _:
                raise ValueError(
                    f"Unsupported iterate averaging config {iterate_averaging_config=}!"
                )

        return (train_interp_coeff, eval_interp_coeff)

    @torch.no_grad()
    def _apply_in_place_primal_averaging(
        self,
        state_lists: dict[str, Any],
        masked_blocked_search_directions: tuple[torch.Tensor, ...],
        train_interp_coeff: float,
        eval_interp_coeff: float,
    ) -> None:
        # If train_interp_coeff = 0, then GPA is equivalent to the base optimizer, so we
        # return immediately.
        if train_interp_coeff == 0.0:
            return

        # Raise user error when attempting to perform primal averaging but not in train mode.
        if not state_lists[TRAIN_MODE]:
            raise RuntimeError(
                "Called _apply_in_place_primal_averaging when not in train mode. Call optimizer.train() before calling optimizer.step()!"
            )

        # Compute (1 - mu_x) * (Z - Y) using the OLD Z value before updating Z.
        # This must be done before updating Z to ensure correct coefficients.
        z_minus_y_term = torch._foreach_sub(
            state_lists[MASKED_WEIGHT_BUFFER_LIST],  # Z (old)
            state_lists[MASKED_BLOCKED_PARAMS],  # Y = W
        )

        # Update Z with gradient descent step: Z <- Z - lr * P.
        # At this point, masked_blocked_search_directions contains -lr * P.
        torch._foreach_add_(
            state_lists[MASKED_WEIGHT_BUFFER_LIST],
            masked_blocked_search_directions,
        )

        # This computes: - (1 - mu_x * mu_y) * lr * P.
        torch._foreach_mul_(
            masked_blocked_search_directions,
            (1 - train_interp_coeff * eval_interp_coeff),
        )
        # This computes: (1 - mu_x) * (Z_old - Y) - (1 - mu_x * mu_y) * lr * P.
        torch._foreach_add_(
            masked_blocked_search_directions,
            z_minus_y_term,
            alpha=1 - eval_interp_coeff,
        )

    @torch.no_grad()
    def _compute_search_directions(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        weight_decay: float,
        peak_lr: float,
        weight_decay_type: WeightDecayType,
        grafting_config_not_none: bool,
        perform_amortized_computation: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        train_interp_coeff: float,
        eval_interp_coeff: float,
    ) -> tuple[torch.Tensor, ...]:
        # Incorporate L2-regularization or (coupled) weight decay if enabled.
        #   G <- G + weight_decay * W
        self._add_l2_regularization(
            state_lists,
            weight_decay,
            weight_decay_type,
        )

        # Update Shampoo and grafting preconditioners.
        # Example for AdaGrad accumulation:
        # 1. Update factor matrices/grafting preconditioners.
        #   L <- L + G * G^T
        #   R <- R + G^T * G
        #   V <- V + G^2    (element-wise)
        #   (and similar)
        # 2. Compute root inverse if necessary.
        #   L_inv <- L ** (-1/4)
        #   R_inv <- R ** (-1/4)
        #   (and similar);
        self._update_preconditioners(
            state_lists=state_lists,
            step=step,
            perform_amortized_computation=perform_amortized_computation,
            grafting_config_not_none=grafting_config_not_none,
        )

        # Compute filtered gradient or EMA of the gradients if beta1 > 0 and beta3 > 0.
        # Note that we use two beta factors here akin to Lion.
        #   G_bar <- beta3 * G_tilde + (1 - beta3) * G
        #   G_tilde <- beta1 * G_tilde + (1 - beta1) * G
        masked_filtered_grad_list = self._compute_filtered_grad_list(
            state_lists,
            step,
            beta1,
            beta3,
            use_bias_correction,
        )

        # Precondition and graft filtered gradients.
        # PT2 compile is currently disabled for preconditioning and grafting.
        # NOTE: Preconditioning and grafting is not compatible with PT2 compile.
        #
        #   P_shampoo <- L_inv * G_bar * R_inv (and similar)
        #   P_grafting <- G_bar / (sqrt(V) + epsilon)
        #   P <- P_grafting                                     if step < start_preconditioning_step
        #   P <- ||P_grafting|| / ||P_shampoo|| * P_shampoo     otherwise
        masked_blocked_search_directions = self._precondition_and_grafting(
            state_lists,
            masked_filtered_grad_list,
            use_grafting_method,
            grafting_config_not_none,
        )

        # Incorporate decoupled or corrected weight decay into search direction if enabled.
        #   P <- P + weight_decay * W                       if decoupled
        #   P <- P + weight_decay * (lr / peak_lr) * W      if corrected
        #   P <- P + weight_decay * (1 / peak_lr) * W       if independent
        self._apply_decoupled_or_corrected_weight_decay(
            state_lists,
            masked_blocked_search_directions,
            lr,
            weight_decay,
            peak_lr,
            weight_decay_type,
        )

        # Multiplies the learning rate to the search direction / update.
        torch._foreach_mul_(masked_blocked_search_directions, -lr)

        # Incorporates primal averaging into the search direction if enabled.
        # NOTE: When primal averaging is enabled, we set Y = W in train mode.
        #   P <- (1 - mu_x * mu_y) * P + (1 - mu_x) * (Z - W)
        #
        # This is equivalent to the expanded update:
        #   P <- mu_x * Y + (1 - mu_x) * Z - (1 - mu_x * mu_y) * lr * P
        self._apply_in_place_primal_averaging(
            state_lists,
            masked_blocked_search_directions,
            train_interp_coeff,
            eval_interp_coeff,
        )

        return masked_blocked_search_directions

    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        weight_decay: float,
        peak_lr: float,
        weight_decay_type: WeightDecayType,
        grafting_config_not_none: bool,
        perform_amortized_computation: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        train_interp_coeff: float,
        eval_interp_coeff: float,
    ) -> None:
        # This method computes search directions and updates parameters in one step
        # It's designed to be compiled with PyTorch 2.0 for performance optimization

        # Call update_params on the distributor with the computed search directions
        # The distributor is responsible for applying updates to the actual parameters
        state_lists[DISTRIBUTOR].update_params(
            # Compute search directions based on current state and optimization parameters
            # This returns the directions in which parameters should be updated
            blocked_search_directions=self._compute_search_directions(
                state_lists=state_lists,
                step=step,
                lr=lr,
                beta1=beta1,
                beta3=beta3,
                weight_decay=weight_decay,
                peak_lr=peak_lr,
                weight_decay_type=weight_decay_type,
                grafting_config_not_none=grafting_config_not_none,
                perform_amortized_computation=perform_amortized_computation,
                use_bias_correction=use_bias_correction,
                use_grafting_method=use_grafting_method,
                train_interp_coeff=train_interp_coeff,
                eval_interp_coeff=eval_interp_coeff,
            )
            # Only update parameters if there are gradients to use
            # Otherwise, return an empty tuple to avoid unnecessary computation
            if state_lists[MASKED_BLOCKED_GRADS]
            else ()
        )

    @overload
    @torch.no_grad()
    def step(self, closure: None = None) -> None: ...

    @overload
    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure (Callable[[], float] | None): A closure that reevaluates the model and returns the loss. (Default: None)

        Returns:
            loss (float | None): The loss value returned by the closure if provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Construct blocked gradient list.
            state_lists[MASKED_BLOCKED_GRADS] = state_lists[
                DISTRIBUTOR
            ].merge_and_block_gradients()

            # Based on the current block selector, mask lists of parameters and optimizer states.
            DistributedShampoo._mask_state_lists(
                state_lists=state_lists,
                group=group,
                shampoo_pt2_enabled=self._shampoo_pt2_compile_config is not None,
            )

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            step_val = step.item()
            # NOTE: Reuse pre-allocated lr tensors to avoid per-step pinned
            # memory allocation (cudaHostAlloc). Fill the pinned CPU tensor and
            # copy to the persistent GPU tensor with non_blocking to overlap H2D.
            # Using a persistent tensor also avoids PT2 recompilation: since lr
            # is the same tensor object every step, PT2 treats it as a dynamic
            # input rather than specializing on each new tensor/value.
            state_lists[LR_CPU_PINNED].fill_(group[LR])
            lr = state_lists[LR_TENSOR]
            lr.copy_(state_lists[LR_CPU_PINNED], non_blocking=True)
            beta1 = group[BETAS][0]
            beta3 = group[BETA3]
            weight_decay = group[WEIGHT_DECAY]
            peak_lr = group[PEAK_LR]
            weight_decay_type = group[WEIGHT_DECAY_TYPE]
            grafting_config_not_none = group[GRAFTING_CONFIG] is not None
            perform_amortized_computation = (
                step_val % group[PRECONDITION_FREQUENCY] == 0
                and step_val > group[START_PRECONDITIONING_STEP]
            ) or step_val == group[START_PRECONDITIONING_STEP]
            use_bias_correction = group[USE_BIAS_CORRECTION]
            # Check if we apply the grafting method or not.
            use_grafting_method = (
                step_val < group[START_PRECONDITIONING_STEP]
                and grafting_config_not_none
            )
            # Set train and eval interpolation coefficients if enabled.
            train_interp_coeff, eval_interp_coeff = (
                self._get_train_and_eval_interp_coeffs(
                    iterate_averaging_config=group[ITERATE_AVERAGING_CONFIG],
                    lr=lr,
                    state_lists=state_lists,
                )
            )

            self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                beta3,
                weight_decay,
                peak_lr,
                weight_decay_type,
                grafting_config_not_none,
                perform_amortized_computation,
                use_bias_correction,
                use_grafting_method,
                train_interp_coeff,
                eval_interp_coeff,
            )

            # Explicitly set masked blocked gradients to None to save memory so the original param.grad has no pointer to it.
            state_lists[MASKED_BLOCKED_GRADS] = None

        return loss

    # ============================================================
    # TRAIN/EVAL MODE SWITCHING
    # ============================================================

    @torch.no_grad()
    def train(self) -> None:
        logger.info("Enabling train mode in Distributed Shampoo!")
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Skip groups without iterate averaging or already in train mode.
            if group[ITERATE_AVERAGING_CONFIG] is None or state_lists[TRAIN_MODE]:
                continue

            train_interp_coeff, _ = self._get_train_and_eval_interp_coeffs(
                iterate_averaging_config=group[ITERATE_AVERAGING_CONFIG],
                lr=None,
                state_lists=state_lists,
            )
            parameter_updates = torch._foreach_sub(
                state_lists[WEIGHT_BUFFER_LIST],
                state_lists[DISTRIBUTOR].local_blocked_params,
            )
            torch._foreach_mul_(parameter_updates, 1 - train_interp_coeff)
            state_lists[DISTRIBUTOR].update_params(
                blocked_search_directions=tuple(parameter_updates),
                use_masked_tensors=False,
            )

            # Set train mode to True.
            state_lists[TRAIN_MODE].fill_(True)

    @torch.no_grad()
    def eval(self) -> None:
        logger.info("Enabling eval mode in Distributed Shampoo!")
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Skip groups without iterate averaging or already in eval mode.
            if group[ITERATE_AVERAGING_CONFIG] is None or not state_lists[TRAIN_MODE]:
                continue

            train_interp_coeff, _ = self._get_train_and_eval_interp_coeffs(
                iterate_averaging_config=group[ITERATE_AVERAGING_CONFIG],
                lr=None,
                state_lists=state_lists,
            )
            parameter_updates = torch._foreach_sub(
                state_lists[WEIGHT_BUFFER_LIST],
                state_lists[DISTRIBUTOR].local_blocked_params,
            )
            torch._foreach_mul_(parameter_updates, 1 - 1 / train_interp_coeff)
            state_lists[DISTRIBUTOR].update_params(
                blocked_search_directions=tuple(parameter_updates),
                use_masked_tensors=False,
            )

            # Set train mode to False.
            state_lists[TRAIN_MODE].fill_(False)

    # ============================================================
    # CHECKPOINTING / STATE DICT METHODS
    # ============================================================

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Override __setstate__ to handle OptimizerModule conversion during state dict loading.

        This method is called by the parent's load_state_dict() method after tensor casting.
        We need to properly convert nested dictionaries back to OptimizerModule objects
        using the same logic as update_param_state_dict_object.

        Args:
            state (dict[str, Any]): The state dictionary containing optimizer state and param_groups.
        """

        # We handle "state" separately from the rest of the state dict because it contains
        # OptimizerModule.

        if "state" not in state:
            super().__setstate__(state)
            return

        # The current `self.state` should already have OptimizerModule objects from initialization.
        # `state["state"]` is a nested dictionary of tensors, so we need to use the data from the
        # nested dictionary to update the OptimizerModule objects.
        for param_id, param_state_to_load in state["state"].items():
            if param_id in self.state:
                update_param_state_dict_object(
                    current_param_state_dict=self.state[param_id],
                    param_state_dict_to_load=param_state_to_load,
                    enable_missing_key_check=False,
                )
            else:
                # If parameter doesn't exist in current state, just assign it directly.
                logger.warning(
                    f"Found parameter {param_id} in state dict to load that doesn't "
                    "exist in current state. Directly assigning it."
                )
                self.state[param_id] = param_state_to_load

        del state["state"]
        super().__setstate__(state)

    @staticmethod
    def _post_state_dict_hook(optimizer: Optimizer, state_dict: StateDict) -> None:
        """Process the state dictionary after it's created by state_dict().

        This hook extracts the actual content from each parameter state using the
        extract_state_dict_content utility function. This ensures that custom dataclass
        objects in the state are converted to nested dictionaries so that their tensor fields
        are recognized as tensors during serialization.

        Args:
            optimizer (Optimizer): The optimizer instance (unused but required by hook signature).
            state_dict (StateDict): The state dictionary created by state_dict() method
                containing optimizer state and parameter groups.

        Returns:
            None: The state_dict is modified in-place.
        """

        def _has_lambda_recursively(obj: Any) -> bool:
            """Recursively check if an object contains lambda functions."""
            if isinstance(obj, LambdaType):
                return True
            if is_dataclass(obj):
                return any(
                    _has_lambda_recursively(getattr(obj, f.name)) for f in fields(obj)
                )
            return False

        # for state exist on the ranks
        state_dict["state"] = {
            k: extract_state_dict_content(v) for k, v in state_dict["state"].items()
        }

        # for state that doesn't exist on the rank, we assign empty dict
        # to make sure all ranks have the same keys in state_dict['state']
        param_ids = []
        for group in state_dict["param_groups"]:
            param_ids.extend(group["params"])
            for v in group.values():
                if _has_lambda_recursively(v):
                    logger.warning(
                        f"Found {v=}. Note that lambda function cannot be pickled. "
                        "torch.save() cannot serialize lambda functions, because it "
                        "relies on Python's pickle module for serialization, and pickle "
                        "does not support lambda functions."
                    )

        state_dict["state"].update(
            {
                param_id: {}
                for param_id in param_ids
                if param_id not in state_dict["state"]
            }
        )

    @staticmethod
    def _pre_load_state_dict_hook(optimizer: Optimizer, state_dict: StateDict) -> None:
        """Save the current train mode for each parameter group before loading state dict.

        This allows the post-load hook to restore the original mode after loading,
        ensuring that the optimizer remains in the same mode the user had set.
        """
        saved_train_modes: list[bool] = [
            bool(state_lists[TRAIN_MODE].item())
            for state_lists, group in zip(
                operator.attrgetter("_per_group_state_lists")(optimizer),
                optimizer.param_groups,
                strict=True,
            )
            if group[ITERATE_AVERAGING_CONFIG] is not None
        ]
        optimizer._pre_load_train_modes = saved_train_modes  # type: ignore[attr-defined]

    @staticmethod
    def _post_load_state_dict_hook(optimizer: Optimizer) -> None:
        """Perform post-load operations after checkpoint loading.

        This hook performs two operations:
        1. Refreshes assigned params in lossless distributors after checkpoint load.
        2. Restores the train/eval mode that was active before loading the checkpoint.
        """

        for state_lists in operator.attrgetter("_per_group_state_lists")(optimizer):
            distributor = state_lists[DISTRIBUTOR]
            if isinstance(
                distributor,
                (FullyShardLosslessDistributor, HybridShardLosslessDistributor),
            ):
                distributor.refresh_assigned_full_params()
                state_lists[MASKED_BLOCKED_PARAMS] = distributor.local_blocked_params

        # Restore the original train/eval mode after loading the checkpoint.
        saved_train_modes: list[bool] = getattr(optimizer, "_pre_load_train_modes", [])
        if saved_train_modes:
            # Mixed train/eval modes across parameter groups is not supported
            # since train() and eval() always operate on all groups uniformly.
            assert all(m == saved_train_modes[0] for m in saved_train_modes), (
                "Mixed train/eval modes across parameter groups is not supported."
            )
            operator.attrgetter("train" if saved_train_modes[0] else "eval")(
                optimizer
            )()

            # Clean up temporary attribute.
            del optimizer._pre_load_train_modes  # type: ignore
