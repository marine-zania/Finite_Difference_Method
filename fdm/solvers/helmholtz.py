"""
2D Helmholtz Equation Solver using Finite Difference Method

Solves the 2D Helmholtz equation: ∇²u(x,y) + k²u(x,y) = 0
for various waveguide geometries.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from .base_solver import BaseSolver


class HelmholtzSolver2D(BaseSolver):
    """
    Solver for 2D Helmholtz equation in rectangular waveguides.
    
    The Helmholtz equation is: ∇²u(x,y) + k²u(x,y) = 0
    where u(x,y) is the field (e.g., Hz component of EM waves).
    """
    
    def __init__(self, length_a: float, length_b: float, k_squared: float = None):
        """
        Initialize the Helmholtz solver.
        
        Parameters
        ----------
        length_a : float
            Waveguide dimension in x-direction (mm)
        length_b : float
            Waveguide dimension in y-direction (mm)
        k_squared : float, optional
            Wave number squared. If None, solving eigenvalue problem.
        """
        super().__init__("2D Helmholtz")
        self.a = length_a
        self.b = length_b
        self.k_squared = k_squared
        self.N = None  # Grid resolution parameter
        self.hx = None
        self.hy = None
        self.eigvals = None
        self.eigvecs = None
        self.modes_list = None
        
    def setup_grid(self, N: int):
        """
        Set up computational grid.
        
        Parameters
        ----------
        N : int
            Number of interior grid points in each direction
        """
        self.N = N
        self.hx = self.a / (N + 1)  # grid spacing in x
        self.hy = self.b / (N + 1)  # grid spacing in y
        self.grid = {
            'N': N,
            'hx': self.hx,
            'hy': self.hy,
            'x': np.linspace(0, self.a, N + 2),
            'y': np.linspace(0, self.b, N + 2),
        }
        
    def construct_matrix(self):
        """
        Construct the finite-difference Laplacian matrix.
        
        Returns
        -------
        scipy.sparse matrix
            The discrete Laplacian operator
        """
        if self.N is None:
            raise ValueError("Grid must be set up first using setup_grid()")
            
        N = self.N
        num_points = N * N
        
        # Main diagonal: 2*(1/hx² + 1/hy²)
        diagonal = np.full(num_points, 2 * (1 / self.hx**2 + 1 / self.hy**2))
        
        # Off-diagonals for x-direction: -1/hx²
        off_diagonal_x = np.full(num_points - 1, -1 / self.hx**2)
        
        # Off-diagonals for y-direction: -1/hy²
        off_diagonal_y = np.full(num_points - N, -1 / self.hy**2)
        
        # Prevent wrap-around in x-direction (essential for rectangular domain)
        for i in range(1, N):
            off_diagonal_x[i * N - 1] = 0
            
        # Construct sparse matrix
        A = diags(
            [diagonal, off_diagonal_x, off_diagonal_x, off_diagonal_y, off_diagonal_y],
            [0, -1, 1, -N, N]
        ).tocsc()
        
        return A
    
    def solve(self, N: int, num_modes: int = 20):
        """
        Solve the eigenvalue problem for modes in the waveguide.
        
        Parameters
        ----------
        N : int
            Grid resolution parameter
        num_modes : int
            Number of eigenvalues/eigenvectors to compute
            
        Returns
        -------
        eigvals : ndarray
            Computed eigenvalues (k² values)
        eigvecs : ndarray
            Computed eigenvectors (mode shapes)
        """
        self.setup_grid(N)
        A = self.construct_matrix()
        
        # Solve eigenvalue problem for the smallest eigenvalues
        self.eigvals, self.eigvecs = eigsh(A, k=num_modes, which='SM')
        
        # Sort results
        sorted_indices = np.argsort(self.eigvals)
        self.eigvals = self.eigvals[sorted_indices]
        self.eigvecs = self.eigvecs[:, sorted_indices]
        
        self.solution = {
            'k_squared': self.eigvals,
            'modes': self.eigvecs,
            'N': N,
        }
        
        return self.eigvals, self.eigvecs
    
    def get_theoretical_k_squared(self, modes_list: list):
        """
        Compute theoretical k² values for given modes.
        
        Parameters
        ----------
        modes_list : list of tuples
            List of (m, n) mode indices for TE_mn modes
            
        Returns
        -------
        ndarray
            Theoretical k² values
        """
        k2_theoretical = np.array([
            ((np.pi * m / self.a)**2 + (np.pi * n / self.b)**2)
            for (m, n) in modes_list
        ])
        return k2_theoretical
    
    def compute_relative_error(self, theoretical_k2, num_modes=None):
        """
        Compute relative error between computed and theoretical k² values.
        
        Parameters
        ----------
        theoretical_k2 : ndarray
            Theoretical k² values
        num_modes : int, optional
            Number of modes to compare (default: all computed modes)
            
        Returns
        -------
        ndarray
            Relative errors
        """
        if self.eigvals is None:
            raise ValueError("Solve must be called first")
            
        if num_modes is None:
            num_modes = min(len(self.eigvals), len(theoretical_k2))
        
        computed_k2 = self.eigvals[:num_modes]
        theoretical_k2 = theoretical_k2[:num_modes]
        
        relative_error = np.abs(computed_k2 - theoretical_k2) / theoretical_k2
        return relative_error
    
    def reshape_mode(self, mode_index: int):
        """
        Reshape mode vector to 2D grid.
        
        Parameters
        ----------
        mode_index : int
            Index of the mode
            
        Returns
        -------
        ndarray
            2D mode field (padded with zeros on boundaries)
        """
        field = self.eigvecs[:, mode_index].reshape((self.N, self.N))
        # Pad with zeros to account for boundary conditions (u=0 on boundaries)
        field_padded = np.pad(field, pad_width=1, mode='constant')
        return field_padded
    
    def normalize_mode(self, field: np.ndarray):
        """
        Normalize a mode field to [-1, 1] range.
        
        Parameters
        ----------
        field : ndarray
            Mode field to normalize
            
        Returns
        -------
        ndarray
            Normalized field
        """
        max_val = np.max(np.abs(field))
        if max_val > 0:
            field = field / max_val
        return field
