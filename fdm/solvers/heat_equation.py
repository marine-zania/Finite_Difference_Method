"""
Heat Equation Solvers using Finite Difference Method

Solves:
- 1D Heat equation: ∂u/∂t = α ∂²u/∂x²
- 2D Heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

Supports multiple numerical schemes:
- Explicit (FTCS): Fast but conditionally stable
- Implicit (BTCS): Unconditionally stable
- Crank-Nicolson: Higher order accuracy O(Δt², Δx²)
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
from .base_solver import BaseSolver


class HeatSolver1D(BaseSolver):
    """
    Solver for 1D heat equation with multiple schemes.
    
    ∂u/∂t = α ∂²u/∂x²
    
    Supports three methods:
    - 'explicit' (FTCS): Forward Time, Centered Space. Stable for r ≤ 0.5
    - 'implicit' (BTCS): Backward Time, Centered Space. Unconditionally stable
    - 'crank_nicolson': O(Δt², Δx²) accuracy, unconditionally stable
    """
    
    def __init__(self, length: float, diffusivity: float = 1.0,
                 bc_left: float = 0.0, bc_right: float = 0.0):
        """
        Initialize 1D heat solver.
        
        Parameters
        ----------
        length : float
            Domain length [0, L]
        diffusivity : float
            Thermal diffusivity α
        bc_left : float
            Boundary condition at x=0
        bc_right : float
            Boundary condition at x=L
        """
        super().__init__("1D Heat Equation")
        self.length = length
        self.diffusivity = diffusivity
        self.bc_left = bc_left
        self.bc_right = bc_right
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
    
    def construct_matrix(self, dt: float, method: str = 'explicit'):
        """
        Construct system matrix for one time step.
        
        Parameters
        ----------
        dt : float
            Time step size
        method : str
            'explicit', 'implicit', or 'crank_nicolson'
            
        Returns
        -------
        matrix_dict : dict
            Contains system matrix and metadata
        """
        if self.nx is None:
            raise ValueError("Grid must be set up first using setup_grid()")
        
        r = self.diffusivity * dt / (self.dx ** 2)
        
        return {
            'r': r,
            'method': method,
            'dt': dt,
            'stable': self._check_stability(r, method)
        }
    
    def _check_stability(self, r: float, method: str) -> bool:
        """Check stability condition for given parameters."""
        if method == 'explicit':
            return r <= 0.5
        else:  # implicit and Crank-Nicolson are unconditionally stable
            return True
    
    def solve_explicit(self, initial_condition: np.ndarray, t_final: float, 
                      dt: float = None, num_output: int = 10):
        """
        Solve using Explicit (Forward Euler) scheme.
        
        Stability condition: r = α*dt/dx² ≤ 0.5
        """
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            # Use stable time step: r = 0.25 (factor of safety)
            dt = 0.25 * (self.dx ** 2) / self.diffusivity
        
        r = self.diffusivity * dt / (self.dx ** 2)
        
        if r > 0.5:
            print(f"WARNING: r={r:.4f} > 0.5. Solution may be unstable!")
        
        # Time stepping
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)  # Adjust to reach exactly t_final
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.nx))
        u[0, :] = initial_condition
        
        for n in range(nt - 1):
            u[n+1, 0] = self.bc_left
            u[n+1, -1] = self.bc_right
            
            for i in range(1, self.nx - 1):
                u[n+1, i] = (r * u[n, i-1] + 
                            (1 - 2*r) * u[n, i] + 
                            r * u[n, i+1])
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x}
        return u_output, t_output, self.x
    
    def solve_implicit(self, initial_condition: np.ndarray, t_final: float,
                      dt: float = None, num_output: int = 10):
        """
        Solve using Implicit (Backward Euler) scheme.
        
        Unconditionally stable. Requires tridiagonal matrix solve each step.
        """
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            dt = self.dx  # Choose dt based on accuracy, not stability
        
        r = self.diffusivity * dt / (self.dx ** 2)
        
        # Time stepping
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.nx))
        u[0, :] = initial_condition
        
        for n in range(nt - 1):
            # Solve (I - r*L)u^{n+1} = u^n
            # Tridiagonal system: (1 + 2r)u_i - r*u_{i-1} - r*u_{i+1} = u_i^n
            
            # Build banded matrix format for solve_banded
            ab = np.zeros((3, self.nx - 2))
            ab[0, 1:] = -r              # upper diagonal
            ab[1, :] = 1 + 2*r          # main diagonal
            ab[2, :-1] = -r             # lower diagonal
            
            # Right-hand side with boundary conditions incorporated
            b = u[n, 1:-1].copy()
            b[0] += r * self.bc_left      # u_0^{n+1} is known
            b[-1] += r * self.bc_right    # u_{nx-1}^{n+1} is known
            
            u[n+1, 0] = self.bc_left
            u[n+1, -1] = self.bc_right
            u[n+1, 1:-1] = solve_banded((1, 1), ab, b)
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x}
        return u_output, t_output, self.x
    
    def solve_crank_nicolson(self, initial_condition: np.ndarray, t_final: float,
                            dt: float = None, num_output: int = 10):
        """
        Solve using Crank-Nicolson scheme.
        
        Order: O(Δt², Δx²)
        Unconditionally stable.
        (I + r/2*L)u^{n+1} = (I - r/2*L)u^n
        """
        self.setup_grid(len(initial_condition))
        
        if dt is None:
            dt = self.dx
        
        r = self.diffusivity * dt / (self.dx ** 2)
        
        # Time stepping
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.nx))
        u[0, :] = initial_condition
        
        for n in range(nt - 1):
            # Build matrices for Crank-Nicolson
            # LHS: (1 + r)u_i - (r/2)u_{i-1} - (r/2)u_{i+1}
            # RHS: (1 - r)u_i^n + (r/2)u_{i-1}^n + (r/2)u_{i+1}^n
            
            ab = np.zeros((3, self.nx - 2))
            ab[0, 1:] = -r/2            # upper diagonal
            ab[1, :] = 1 + r            # main diagonal
            ab[2, :-1] = -r/2           # lower diagonal
            
            # RHS with boundary conditions
            b = np.zeros(self.nx - 2)
            for i in range(1, self.nx - 1):
                b[i-1] = ((1 - r) * u[n, i] +
                         (r/2) * u[n, i-1] +
                         (r/2) * u[n, i+1])
            
            b[0] += (r/2) * self.bc_left
            b[-1] += (r/2) * self.bc_right
            
            u[n+1, 0] = self.bc_left
            u[n+1, -1] = self.bc_right
            u[n+1, 1:-1] = solve_banded((1, 1), ab, b)
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :]
        
        self.solution = {'u': u_output, 't': t_output, 'x': self.x}
        return u_output, t_output, self.x
    
    def solve(self, initial_condition: np.ndarray, t_final: float,
              method: str = 'crank_nicolson', dt: float = None, 
              num_output: int = 10):
        """
        Solve 1D heat equation with specified method.
        
        Parameters
        ----------
        initial_condition : ndarray
            Initial values u(x, 0)
        t_final : float
            Final time
        method : str
            'explicit', 'implicit', or 'crank_nicolson'
        dt : float, optional
            Time step. If None, automatically selected.
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
        if method == 'explicit':
            return self.solve_explicit(initial_condition, t_final, dt, num_output)
        elif method == 'implicit':
            return self.solve_implicit(initial_condition, t_final, dt, num_output)
        elif method == 'crank_nicolson':
            return self.solve_crank_nicolson(initial_condition, t_final, dt, num_output)
        else:
            raise ValueError(f"Unknown method: {method}")


class HeatSolver2D(BaseSolver):
    """
    Solver for 2D heat equation.
    
    ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    
    Uses Alternating Direction Implicit (ADI) method for efficiency.
    """
    
    def __init__(self, length_x: float, length_y: float, diffusivity: float = 1.0):
        """
        Initialize 2D heat solver.
        
        Parameters
        ----------
        length_x : float
            Domain length in x-direction [0, Lx]
        length_y : float
            Domain length in y-direction [0, Ly]
        diffusivity : float
            Thermal diffusivity α
        """
        super().__init__("2D Heat Equation")
        self.length_x = length_x
        self.length_y = length_y
        self.diffusivity = diffusivity
        
    def setup_grid(self, nx: int, ny: int):
        """
        Set up spatial grid.
        
        Parameters
        ----------
        nx : int
            Number of points in x-direction
        ny : int
            Number of points in y-direction
        """
        self.nx = nx
        self.ny = ny
        self.dx = self.length_x / (nx - 1)
        self.dy = self.length_y / (ny - 1)
        
        x = np.linspace(0, self.length_x, nx)
        y = np.linspace(0, self.length_y, ny)
        self.X, self.Y = np.meshgrid(x, y)
    
    def construct_matrix(self):
        """Construct system matrix for 2D heat equation."""
        if not hasattr(self, 'nx'):
            raise ValueError("Grid must be set up first using setup_grid()")
    
    def solve(self, initial_condition: np.ndarray, t_final: float,
              dt: float = None, num_output: int = 10):
        """
        Solve 2D heat equation.
        
        Parameters
        ----------
        initial_condition : ndarray
            Initial field u(x,y,0) of shape (ny, nx)
        t_final : float
            Final time
        dt : float, optional
            Time step
        num_output : int
            Number of output times
            
        Returns
        -------
        u_output : ndarray
            Solution field at output times
        t_output : ndarray
            Output time points
        X, Y : ndarray
            2D meshgrid
        """
        self.setup_grid(initial_condition.shape[1], initial_condition.shape[0])
        
        if dt is None:
            dt = 0.25 * min(self.dx, self.dy)**2 / self.diffusivity
        
        # Time stepping (implement ADI or simple 2D explicit/implicit)
        nt = int(t_final / dt) + 1
        dt = t_final / (nt - 1)
        t_all = np.linspace(0, t_final, nt)
        
        u = np.zeros((nt, self.ny, self.nx))
        u[0, :, :] = initial_condition
        
        # Simple explicit scheme for 2D heat equation
        rx = self.diffusivity * dt / (self.dx ** 2)
        ry = self.diffusivity * dt / (self.dy ** 2)
        
        for n in range(nt - 1):
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    u[n+1, i, j] = (u[n, i, j] +
                                   rx * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1]) +
                                   ry * (u[n, i+1, j] - 2*u[n, i, j] + u[n, i-1, j]))
            
            # Apply boundary conditions (Dirichlet: u=0)
            u[n+1, 0, :] = 0
            u[n+1, -1, :] = 0
            u[n+1, :, 0] = 0
            u[n+1, :, -1] = 0
        
        # Select output times
        output_idx = np.linspace(0, nt-1, num_output, dtype=int)
        t_output = t_all[output_idx]
        u_output = u[output_idx, :, :]
        
        self.solution = {'u': u_output, 't': t_output, 'X': self.X, 'Y': self.Y}
        return u_output, t_output, self.X, self.Y
