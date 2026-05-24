"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    DefaultNewtonSchulzOrthogonalizationConfig,
    DefaultPerturbationConfig,
    EigenConfig,
    EigendecompositionConfig,
    EighEigendecompositionConfig,
    MatrixFunctionConfig,
    NewtonSchulzOrthogonalizationConfig,
    OrthogonalizationConfig,
    PerturbationConfig,
    PseudoInverseConfig,
    QREigendecompositionConfig,
    RankDeficientStabilityConfig,
    RootInvConfig,
    SVDOrthogonalizationConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    BaseShampooPreconditionerConfig,
    ClassicShampooPreconditionerConfig,
    DDPDistributedConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSOAPConfig,
    DefaultSpectralDescentPreconditionerConfig,
    DistributedConfig,
    EigendecomposedKLShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPDistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    GeneralizedPrimalAveragingConfig,
    HSDPDistributedConfig,
    HybridShardDistributedConfig,
    IterateAveragingConfig,
    PreconditionerConfig,
    RMSpropPreconditionerConfig,
    RootInvKLShampooPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    ScheduleFreeConfig,
    SGDPreconditionerConfig,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    WeightDecayType,
)


__all__ = [
    "DistributedShampoo",
    # PT2 compile.
    "ShampooPT2CompileConfig",
    # `distributed_config` options.
    "DistributedConfig",  # Abstract base class.
    "SingleDeviceDistributedConfig",
    "DefaultSingleDeviceDistributedConfig",  # Default `SingleDeviceDistributedConfig`.
    "DDPDistributedConfig",
    "FSDPParamAssignmentStrategy",
    "FSDPDistributedConfig",
    "FullyShardDistributedConfig",
    "HSDPDistributedConfig",
    "HybridShardDistributedConfig",
    # `precision_config`.
    # `preconditioner_config` options.
    "PreconditionerConfig",  # Abstract base class.
    "BaseShampooPreconditionerConfig",  # Abstract base class (based on `PreconditionerConfig`).
    "ClassicShampooPreconditionerConfig",  # Abstract base class (based on `BaseShampooPreconditionerConfig`).
    "RootInvShampooPreconditionerConfig",  # Based on `ClassicShampooPreconditionerConfig`.
    "DefaultShampooConfig",  # Default `RootInvShampooPreconditionerConfig` using `EigenConfig`.
    "RootInvKLShampooPreconditionerConfig",  # Based on `RootInvShampooPreconditionerConfig`.
    "EigendecomposedShampooPreconditionerConfig",  # Based on `ClassicShampooPreconditionerConfig`.
    "EigendecomposedKLShampooPreconditionerConfig",  # Based on `EigendecomposedShampooPreconditionerConfig`.
    "EigenvalueCorrectedShampooPreconditionerConfig",  # Based on `BaseShampooPreconditionerConfig`.
    "DefaultEigenvalueCorrectedShampooConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `EighEigendecompositionConfig`.
    "DefaultSOAPConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `QREigendecompositionConfig`.
    "SpectralDescentPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "SignDescentPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "DefaultSpectralDescentPreconditionerConfig",  # Default `SpectralDescentPreconditionerConfig` using `NewtonSchulzOrthogonalizationConfig`.
    "DefaultSignDescentPreconditionerConfig",  # Default `SignDescentPreconditionerConfig`.
    "SGDPreconditionerConfig",
    "AdaGradPreconditionerConfig",
    "RMSpropPreconditionerConfig",
    "AdamPreconditionerConfig",
    # matrix functions configs.
    "RankDeficientStabilityConfig",  # Abstract base class.
    "PerturbationConfig",  # Based on `RankDeficientStabilityConfig`.
    "DefaultPerturbationConfig",  # Default `PerturbationConfig`.
    "PseudoInverseConfig",  # Based on `RankDeficientStabilityConfig`.
    "MatrixFunctionConfig",  # Abstract base class.
    "EigendecompositionConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EighEigendecompositionConfig",  # Based on `EigendecompositionConfig`.
    "DefaultEigendecompositionConfig",  # Default `EigendecompositionConfig` using `EighEigendecompositionConfig`.
    "QREigendecompositionConfig",  # Based on `EigendecompositionConfig`.
    "RootInvConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EigenConfig",  # Based on `RootInvConfig` and `EigendecompositionConfig`.
    "DefaultEigenConfig",  # Default `RootInvConfig` using `EigenConfig`.
    "CoupledNewtonConfig",  # Based on `RootInvConfig`.
    "CoupledHigherOrderConfig",  # Based on `RootInvConfig`.
    "OrthogonalizationConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "SVDOrthogonalizationConfig",  # Based on `OrthogonalizationConfig`.
    "NewtonSchulzOrthogonalizationConfig",  # Based on `OrthogonalizationConfig`.
    "DefaultNewtonSchulzOrthogonalizationConfig",  # Default `OrthogonalizationConfig` using `NewtonSchulzOrthogonalizationConfig`.
    # Iterate averaging configs.
    "IterateAveragingConfig",  # Abstract base class.
    "GeneralizedPrimalAveragingConfig",  # Based on `IterateAveragingConfig`.
    "ScheduleFreeConfig",  # Based on `IterateAveragingConfig`.
    # Weight decay type.
    "WeightDecayType",
    # Other utilities.
    "compile_fsdp_parameter_metadata",  # For `FSDPDistributedConfig` and `HSDPDistributedConfig`.
]
