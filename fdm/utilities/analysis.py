"""Analysis utilities for convergence and validation"""

import numpy as np
import pandas as pd


def convergence_analysis(solver, N_values: list, num_modes: int, 
                         theoretical_k2: np.ndarray = None):
    """
    Perform convergence analysis by solving for multiple grid resolutions.
    
    Parameters
    ----------
    solver : BaseSolver
        Solver instance (e.g., HelmholtzSolver2D)
    N_values : list
        List of grid resolution parameters to test
    num_modes : int
        Number of eigenvalues to compute
    theoretical_k2 : ndarray, optional
        Theoretical k² values for error comparison
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'N_values': Grid resolutions tested
        - 'h_values': Corresponding grid spacings
        - 'computed_k2': Array of computed k² for each N and mode
        - 'relative_errors': Relative errors (if theoretical_k2 provided)
    """
    results = {
        'N_values': N_values,
        'h_values': [],
        'computed_k2': [],
        'relative_errors': [] if theoretical_k2 is not None else None
    }
    
    for N in N_values:
        # Solve for this grid resolution
        eigvals, _ = solver.solve(N, num_modes=num_modes)
        
        h = solver.a / (N + 1)  # assuming solver has 'a' attribute
        results['h_values'].append(h)
        results['computed_k2'].append(eigvals)
        
        # Compute errors if theoretical values provided
        if theoretical_k2 is not None:
            rel_errors = np.abs(eigvals - theoretical_k2[:num_modes]) / theoretical_k2[:num_modes]
            results['relative_errors'].append(rel_errors)
    
    results['computed_k2'] = np.array(results['computed_k2'])
    results['h_values'] = np.array(results['h_values'])
    if results['relative_errors'] is not None:
        results['relative_errors'] = np.array(results['relative_errors'])
    
    return results


def validate_against_theoretical(solver, theoretical_k2: np.ndarray, 
                                  num_modes: int = None, decimal_places: int = 4):
    """
    Validate computed k² values against theoretical values.
    
    Parameters
    ----------
    solver : BaseSolver
        Solver instance with computed solution
    theoretical_k2 : ndarray
        Theoretical k² values
    num_modes : int, optional
        Number of modes to compare (default: all)
    decimal_places : int
        Precision for error display
        
    Returns
    -------
    validation_table : pandas.DataFrame
        Table with mode, computed k², theoretical k², and relative error
    """
    if solver.eigvals is None:
        raise ValueError("Solver must be solved first")
    
    if num_modes is None:
        num_modes = min(len(solver.eigvals), len(theoretical_k2))
    
    computed = solver.eigvals[:num_modes]
    theoretical = theoretical_k2[:num_modes]
    rel_errors = np.abs(computed - theoretical) / theoretical
    
    data = {
        'Mode': [f"Mode {i}" for i in range(num_modes)],
        'Computed k²': np.round(computed, decimal_places),
        'Theoretical k²': np.round(theoretical, decimal_places),
        'Relative Error': np.round(rel_errors, -int(np.log10(10**(-decimal_places))))
    }
    
    return pd.DataFrame(data)


def compute_energy_norm(field: np.ndarray, dx: float, dy: float = None):
    """
    Compute L2 energy norm of a field.
    
    Parameters
    ----------
    field : ndarray
        2D field (or 1D for temporal fields)
    dx : float
        Grid spacing
    dy : float, optional
        Grid spacing in y-direction (for 2D fields)
        
    Returns
    -------
    norm : float
        L2 energy norm
    """
    if dy is None:
        # 1D case
        norm = np.sqrt(np.sum(field**2) * dx)
    else:
        # 2D case
        norm = np.sqrt(np.sum(field**2) * dx * dy)
    
    return norm
