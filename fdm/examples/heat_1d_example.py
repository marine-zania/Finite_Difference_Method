"""Example: 1D Heat Equation

Solves the 1D heat equation using three different numerical schemes:
- Explicit (FTCS): Forward Time, Centered Space
- Implicit (BTCS): Backward Time, Centered Space  
- Crank-Nicolson: Higher-order accurate scheme

Demonstrates the stability/accuracy tradeoffs of each method.
"""

import numpy as np
import matplotlib.pyplot as plt
from fdm.solvers import HeatSolver1D


def example_1d_heat_comparison():
    """
    Compare three numerical methods for the 1D heat equation.
    
    Problem:
    ∂u/∂t = α ∂²u/∂x²  on [0, L] × [0, T]
    u(x, 0) = 0.5*x*(L-x)  (initial condition)
    u(0, t) = 0, u(L, t) = 0  (boundary conditions)
    """
    print("=" * 70)
    print("Example: 1D Heat Equation - Comparison of Methods")
    print("=" * 70)
    
    # Problem parameters
    L = 4.0  # Domain length
    nx = 41  # Number of spatial points
    alpha = 1/16  # Thermal diffusivity
    T = 0.5  # Final time
    
    # Initial condition: u(x,0) = 0.5*x*(L-x)
    x = np.linspace(0, L, nx)
    u_initial = 0.5 * x * (L - x)
    
    print(f"\nProblem setup:")
    print(f"  Domain: [0, {L}] × [0, {T}]")
    print(f"  Grid points: {nx}")
    print(f"  Diffusivity: α = {alpha}")
    print(f"  Initial condition: u(x,0) = 0.5*x*({L}-x)")
    print(f"  Boundary conditions: u(0,t) = 0, u({L},t) = 0")
    
    # Solve with three methods
    methods = ['explicit', 'implicit', 'crank_nicolson']
    solutions = {}
    
    for method in methods:
        print(f"\nSolving with {method.upper()} method...")
        
        solver = HeatSolver1D(length=L, diffusivity=alpha)
        u, t, x = solver.solve(u_initial, T, method=method, num_output=5)
        
        solutions[method] = {
            'u': u,
            't': t,
            'x': x,
        }
        
        print(f"  Max solution value: {np.max(u):.6f}")
        print(f"  Min solution value: {np.min(u):.6f}")
        print(f"  Final time reached: {t[-1]:.6f}")
    
    # Plot comparison
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        u, t, x = solutions[method]['u'], solutions[method]['t'], solutions[method]['x']
        
        # Plot solution at different times
        for i, (time, solution) in enumerate(zip(t, u)):
            color = plt.cm.viridis(i / len(t))
            ax.plot(x, solution, label=f't={time:.3f}', color=color, marker='o', markersize=3)
        
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"{method.upper()} Method")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_1d_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: heat_1d_methods_comparison.png")
    plt.show()


def example_1d_heat_stability_study():
    """
    Study stability and accuracy vs. time step for explicit method.
    """
    print("\n" + "=" * 70)
    print("Example: Stability Study for Explicit Method")
    print("=" * 70)
    
    L = 4.0
    nx = 41
    alpha = 1/16
    T = 0.5
    
    x = np.linspace(0, L, nx)
    u_initial = 0.5 * x * (L - x)
    dx = L / (nx - 1)
    
    # Test different time steps
    dt_max_stable = 0.5 * (dx ** 2) / alpha  # Stability limit: r ≤ 0.5
    dt_values = [0.1 * dt_max_stable, 0.25 * dt_max_stable, 0.45 * dt_max_stable, 0.55 * dt_max_stable]
    
    print(f"\nGrid spacing: dx = {dx:.6f}")
    print(f"Maximum stable dt: {dt_max_stable:.6f} (for r = 0.5)")
    print(f"Testing dt values: {[f'{dt:.6f}' for dt in dt_values]}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, dt in enumerate(dt_values):
        ax = axes[idx]
        r = alpha * dt / (dx ** 2)
        
        solver = HeatSolver1D(length=L, diffusivity=alpha)
        u, t, x_plot = solver.solve(u_initial, T, method='explicit', dt=dt, num_output=5)
        
        # Plot
        for i, (time, solution) in enumerate(zip(t, u)):
            color = plt.cm.viridis(i / len(t))
            ax.plot(x_plot, solution, label=f't={time:.3f}', color=color, marker='o', markersize=3)
        
        status = "✓ Stable" if r <= 0.5 else "✗ Unstable (numerical oscillations expected)"
        ax.set_title(f"dt={dt:.6f}, r={r:.3f} {status}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_1d_stability_study.png', dpi=150, bbox_inches='tight')
    print("\nSaved: heat_1d_stability_study.png")
    plt.show()


if __name__ == "__main__":
    example_1d_heat_comparison()
    example_1d_heat_stability_study()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
