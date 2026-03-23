"""
1D Burgers Equation Solver using Finite Difference Method

Solves: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

This is a classic test case for nonlinear PDE solvers combining:
- Advection (nonlinear term u∂u/∂x)
- Diffusion (viscous term ν∂²u/∂x²)

Can be transformed to linear heat equation via Cole-Hopf transformation.
"""

import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_banded
from .base_solver import BaseSolver


class BurgersSolver1D(BaseSolver):
    """
    Solver for 1D Burgers equation.
    
    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Supports multiple schemes:
    - Explicit: Simple finite difference, good for smooth solutions
    - Upwind: Stabilizes advection term for steep gradients
    - Cole-Hopf: Transform to linear heat equation for highly accurate solution
    """
    
    def __init__(self, length: float, viscosity: float = 0.01):
        """
        Initialize 1D Burgers equation solver.
        
        Parameters
        ----------
        length : float
            Domain length [0, L]
        viscosity : float
            Kinematic viscosity ν
        """
        super().__init__("1D Burgers Equation")
        self.length = length
        self.viscosity = viscosity
        self.x = None
        self.nx = None
        self.dx = None
        
    def setup_grid(self, nx: int):
        """
        Set up spatial grid.
        
        Parameters
        ----------
        nx : int
            Number of spatial grid points
        """
        self.nx = nx
        self.dx = self.length / (nx - 1)
        self.x = np.linspace(0, self.length, nx)
        self.grid = {'x': self.x, 'nx': nx, 'dx': self.dx}
    
    def construct_matrix(self, dt: float, scheme: str = 'upwind'):
        """
        Construct system matrix for one time step.
        
        Parameters
        ----------
        dt : float
            Time step size
        scheme : str
            'explicit', 'upwind', or 'implicit'
            
        Returns
        -------
        matrix_dict : dict
            Contains discretization parameters
        """
        if self.nx is None:
            raise ValueError("Grid must be set up first using setup_grid()")
        
        r = self.viscosity * dt / (self.dx ** 2)
        c = dt / self.dx  # Courant number or advection parameter
        
        return {
            'r': r,
            'c': c,
            'scheme': scheme,
            'dt': dt,
        }
    
    def solve_explicit_centered(self, initial_condition: np.ndarray, t_final: float,
                               dt: float = None, num_output: int = 10):
        """
        Solve Burgers equation with explicit centered difference scheme.
        
        Discretization:
        (u_{i}^{n+1} - u_{i}^{n})/dt + u_i^n (u_{i+1}^n - u_{i-1}^n)/(2dx) 
            = ν(u_{i+1}^n - 2u_i^n + u_{i-1}^n)/dx²
        
        Stability: Requires r ≤ 0.5 and C ≤ 2√r where C = |u|dt/dx
        """
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            dt = 0.5 * (self.dx ** 2) / self.viscosity
        
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.nx))
        u[0, :] = initial_condition
        
        r = self.viscosity * dt / (self.dx ** 2)
        
        for n in range(nt - 1):
            u[n+1, 0] = 0   # Dirichlet BC
            u[n+1, -1] = 0
            
            for i in range(1, self.nx - 1):
                advection = u[n, i] * (u[n, i+1] - u[n, i-1]) / (2 * self.dx)
                diffusion = r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                
                u[n+1, i] = u[n, i] - dt * advection + diffusion
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x}
        return u_output, t_output, self.x
    
    def solve_upwind(self, initial_condition: np.ndarray, t_final: float,
                    dt: float = None, num_output: int = 10):
        """
        Solve Burgers equation with upwind scheme for advection.
        
        Upwind discretization stabilizes the solution by using one-sided
        differences in the direction of flow:
        u_i^n (u_i^n - u_{i-1}^n)/dx  if u_i > 0
        u_i^n (u_{i+1}^n - u_i^n)/dx  if u_i < 0
        
        Stability: Improved for steep gradients andshocks.
        """
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            dt = 0.5 * (self.dx ** 2) / self.viscosity
        
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.nx))
        u[0, :] = initial_condition
        
        r = self.viscosity * dt / (self.dx ** 2)
        
        for n in range(nt - 1):
            u[n+1, 0] = 0
            u[n+1, -1] = 0
            
            for i in range(1, self.nx - 1):
                # Upwind advection
                if u[n, i] > 0:
                    advection = u[n, i] * (u[n, i] - u[n, i-1]) / self.dx
                else:
                    advection = u[n, i] * (u[n, i+1] - u[n, i]) / self.dx
                
                # Centered diffusion
                diffusion = r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                
                u[n+1, i] = u[n, i] - dt * advection + diffusion
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x}
        return u_output, t_output, self.x
    
    def solve_cole_hopf(self, initial_condition: np.ndarray, t_final: float,
                       dt: float = None, num_output: int = 10):
        """
        Solve Burgers equation using Cole-Hopf transformation.
        
        The Cole-Hopf transformation φ = exp(-∫u dx / (2ν)) converts Burgers
        equation to linear heat equation: ∂φ/∂t = ν ∂²φ/∂x²
        
        This approach is highly accurate and handles shocks naturally.
        """
        from scipy.integrate import cumtrapz
        
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            dt = 0.25 * (self.dx ** 2) / self.viscosity
        
        # Transform initial condition: φ = exp(-∫u/(2ν) dx)
        integrand = -initial_condition / (2 * self.viscosity)
        integral = np.zeros_like(integrand)
        integral[1:] = cumtrapz(integrand, self.x)  # Cumulative integration
        phi_0 = np.exp(integral)
        
        # Solve heat equation for φ with implicit method (unconditionally stable)
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        phi = np.zeros((nt, self.nx))
        phi[0, :] = phi_0
        
        r = self.viscosity * dt / (self.dx ** 2)
        
        for n in range(nt - 1):
            # Implicit scheme for heat equation
            ab = np.zeros((3, self.nx - 2))
            ab[0, 1:] = -r
            ab[1, :] = 1 + 2*r
            ab[2, :-1] = -r
            
            b = phi[n, 1:-1].copy()
            b[0] += r * phi[n, 0]
            b[-1] += r * phi[n, -1]
            
            phi[n+1, 0] = phi[n, 0]  # BC: φ boundary = constant
            phi[n+1, -1] = phi[n, -1]
            phi[n+1, 1:-1] = solve_banded((1, 1), ab, b)
        
        # Transform back: u = -2ν ∂(ln φ)/∂x
        u = np.zeros((nt, self.nx))
        for n in range(nt):
            # Centered difference for derivative
            dphi_dx = np.zeros(self.nx)
            dphi_dx[1:-1] = (phi[n, 2:] - phi[n, :-2]) / (2 * self.dx)
            dphi_dx[0] = (phi[n, 1] - phi[n, 0]) / self.dx
            dphi_dx[-1] = (phi[n, -1] - phi[n, -2]) / self.dx
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                u[n, :] = -2 * self.viscosity * dphi_dx / phi[n, :]
                u[n, phi[n, :] == 0] = 0
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x, 'phi': phi}
        return u_output, t_output, self.x
    
    def solve(self, initial_condition: np.ndarray, t_final: float,
              scheme: str = 'upwind', dt: float = None, num_output: int = 10):
        """
        Solve 1D Burgers equation with specified scheme.
        
        Parameters
        ----------
        initial_condition : ndarray
            Initial values u(x, 0)
        t_final : float
            Final time
        scheme : str
            'explicit' (centered), 'upwind', or 'cole_hopf'
        dt : float, optional
            Time step. If None, automatically selected for stability.
        num_output : int
            Number of output time steps to return
            
        Returns
        -------
        u : ndarray
            Solution at output times
        t : ndarray
            Output time points
        x : ndarray
            Spatial grid points
        """
        if scheme == 'explicit':
            return self.solve_explicit_centered(initial_condition, t_final, dt, num_output)
        elif scheme == 'upwind':
            return self.solve_upwind(initial_condition, t_final, dt, num_output)
        elif scheme == 'cole_hopf':
            return self.solve_cole_hopf(initial_condition, t_final, dt, num_output)
        else:
            raise ValueError(f"Unknown scheme: {scheme}. Use 'explicit', 'upwind', or 'cole_hopf'")
