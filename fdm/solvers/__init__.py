"""Solver modules for various PDEs using Finite Difference Method"""

from .base_solver import BaseSolver
from .helmholtz import HelmholtzSolver2D
from .heat_equation import HeatSolver1D, HeatSolver2D
from .burgers_equation import BurgersSolver1D

__all__ = [
    "BaseSolver",
    "HelmholtzSolver2D",
    "HeatSolver1D",
    "HeatSolver2D",
    "BurgersSolver1D",
]
