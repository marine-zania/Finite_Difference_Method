"""Base class for Finite Difference Method solvers"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSolver(ABC):
    """
    Abstract base class for FDM solvers.
    
    Provides common interface for solving PDEs using the Finite Difference Method.
    """
    
    def __init__(self, name: str = "FDM Solver"):
        """
        Initialize the solver.
        
        Parameters
        ----------
        name : str
            Name identifier for the solver
        """
        self.name = name
        self.solution = None
        self.grid = None
        
    @abstractmethod
    def setup_grid(self, *args, **kwargs):
        """Set up the computational grid"""
        pass
    
    @abstractmethod
    def construct_matrix(self, *args, **kwargs):
        """Construct the system matrix for FDM"""
        pass
    
    @abstractmethod
    def solve(self, *args, **kwargs):
        """Solve the PDE"""
        pass
    
    def get_solution(self):
        """Return the computed solution"""
        return self.solution
    
    def __repr__(self):
        return f"{self.name} Solver"
