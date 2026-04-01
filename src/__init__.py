"""
Logistics ETA-PDE: Physics-Informed ETA Prediction
==================================================

A physics-informed framework for global logistics ETA prediction using
non-homogeneous Navier-Stokes equations with jump discontinuities.
"""

from .logistics_ns_solver import (
    LogisticsNSEquation,
    LogisticsFlowSimulator,
    VaRCalculator
)

from .pino_model import (
    HybridPINO,
    PhysicsInformedLoss,
    LogisticsFNO,
    DistributionalHead
)

from .visualization import LogisticsVisualizer

__version__ = "0.1.0"
__author__ = "Logistics ETA-PDE Research Team"

__all__ = [
    'LogisticsNSEquation',
    'LogisticsFlowSimulator',
    'VaRCalculator',
    'HybridPINO',
    'PhysicsInformedLoss',
    'LogisticsFNO',
    'DistributionalHead',
    'LogisticsVisualizer'
]
