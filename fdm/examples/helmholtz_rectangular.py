"""Example: 2D Helmholtz Equation - Rectangular Waveguide

This example solves the 2D Helmholtz equation for a rectangular waveguide
(WR90 standard) using the Finite Difference Method and compares with
analytical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from fdm.solvers import HelmholtzSolver2D
from fdm.utilities import plot_modes, plot_convergence, validate_against_theoretical


def example_rectangular_modes():
    """
    Solve for modes in a rectangular waveguide (WR90: 19.05 × 9.525 mm).
    """
    print("=" * 60)
    print("Example: 2D Helmholtz in Rectangular Waveguide (WR90)")
    print("=" * 60)
    
    # WR90 waveguide dimensions (in mm)
    a = 19.05  # x-dimension
    b = 9.525  # y-dimension
    
    # Define TE modes for comparison
    modes = [(1, 1), (2, 1), (3, 1), (1, 2), (4, 1), (2, 2), (3, 2), (5, 1)]
    
    # Create solver
    solver = HelmholtzSolver2D(length_a=a, length_b=b)
    
    # Compute theoretical k² values
    k2_theoretical = solver.get_theoretical_k_squared(modes)
    
    print(f"\nWaveguide dimensions: {a} mm × {b} mm")
    print(f"Modes to analyze: {modes}\n")
    
    # Solve eigenvalue problem with moderate grid resolution
    print("Solving eigenvalue problem (N=150)...")
    eigvals, eigvecs = solver.solve(N=150, num_modes=20)
    
    # Print comparison table
    print("\nComputed vs. Theoretical k² Values:")
    print("-" * 70)
    print(f"{'Mode':<15} {'Computed k²':<20} {'Theoretical k²':<20}")
    print("-" * 70)
    
    for idx, (m, n) in enumerate(modes):
        theo_k2 = ((np.pi * m / a)**2 + (np.pi * n / b)**2)
        comp_k2 = eigvals[idx]
        rel_err = np.abs(comp_k2 - theo_k2) / theo_k2
        print(f"TE({m},{n})       {comp_k2:.6e}       {theo_k2:.6e}  ({rel_err:.3e})")
    
    # Plot first 8 modes
    print("\nGenerating mode plots...")
    plot_modes(solver, list(range(8)), modes_per_row=4, figsize=(16, 8))
    
    plt.savefig('helmholtz_rectangular_modes.png', dpi=150, bbox_inches='tight')
    print("Mode plot saved as 'helmholtz_rectangular_modes.png'")
    plt.show()


def example_convergence_analysis():
    """
    Perform convergence analysis for different grid resolutions.
    """
    print("\n" + "=" * 60)
    print("Example: Convergence Analysis")
    print("=" * 60)
    
    a = 19.05
    b = 9.525
    modes = [(1, 1), (2, 1), (3, 1), (1, 2)]
    
    solver = HelmholtzSolver2D(length_a=a, length_b=b)
    k2_theoretical = solver.get_theoretical_k_squared(modes)
    
    # Test multiple grid resolutions
    N_values = [50, 75, 100, 150, 200, 300]
    
    print(f"Testing grid resolutions: {N_values}\n")
    
    errors_by_mode = {f"TE({m},{n})": [] for m, n in modes}
    h_values = []
    
    for N in N_values:
        print(f"Solving with N={N}...", end=" ")
        eigvals, _ = solver.solve(N, num_modes=len(modes))
        h = a / (N + 1)
        h_values.append(h)
        
        # Compute relative errors
        rel_errors = solver.compute_relative_error(k2_theoretical, num_modes=len(modes))
        for idx, (m, n) in enumerate(modes):
            errors_by_mode[f"TE({m},{n})"].append(rel_errors[idx])
        
        print(f"h={h:.4e}, Relative errors: {rel_errors}")
    
    # Plot convergence
    print("\nGenerating convergence plot...")
    plot_convergence(h_values, errors_by_mode, figsize=(10, 6))
    
    plt.savefig('helmholtz_convergence.png', dpi=150, bbox_inches='tight')
    print("Convergence plot saved as 'helmholtz_convergence.png'")
    plt.show()


if __name__ == "__main__":
    # Run examples
    example_rectangular_modes()
    example_convergence_analysis()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
