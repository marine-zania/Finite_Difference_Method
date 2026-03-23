"""
FDM (Finite Difference Method) Package for solving PDEs
========================================================

This package provides modular solvers for various partial differential equations
using the Finite Difference Method, including:
- 2D Helmholtz equation in waveguides
- Heat equations (1D and 2D)
- Burgers equation (1D)

"""

__version__ = "1.0.0"
__author__ = "Zania"

from .solvers import HelmholtzSolver2D, HeatSolver1D, HeatSolver2D, BurgersSolver1D

__all__ = [
    "HelmholtzSolver2D",
    "HeatSolver1D",
    "HeatSolver2D",
    "BurgersSolver1D",
]
