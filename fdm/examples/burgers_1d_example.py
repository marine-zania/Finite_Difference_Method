"""Example: 1D Burgers Equation

Demonstrates three approaches to solving the 1D Burgers equation:

∂u/∂t + u∂u/∂x = ν∂²u/∂x²

This is a classic nonlinear PDE that combines:
- Advection (u∂u/∂x)
- Diffusion (ν∂²u/∂x²)

Three solution methods:
1. Explicit centered differences (simple, less stable)
2. Upwind scheme (more stable for steep gradients)
3. Cole-Hopf transformation (highly accurate, handles shocks)
"""

import numpy as np
import matplotlib.pyplot as plt
from fdm.solvers import BurgersSolver1D


def example_burgers_comparison():
    """
    Compare three numerical schemes for Burgers equation.
    """
    print("=" * 70)
    print("Example: 1D Burgers Equation - Comparison of Schemes")
    print("=" * 70)
    
    # Problem parameters
    L = 1.0
    nx = 101
    viscosity = 0.01
    T = 0.5
    
    # Initial condition: Gaussian bump
    x = np.linspace(0, L, nx)
    u_initial = np.exp(-100 * (x - 0.5)**2)
    
    print(f"\nProblem setup:")
    print(f"  Domain: [0, {L}] × [0, {T}]")
    print(f"  Grid points: {nx}")
    print(f"  Viscosity: ν = {viscosity}")
    print(f"  Initial condition: Gaussian bump")
    print(f"  Boundary conditions: u(0,t) = 0, u(L,t) = 0")
    
    # Solve with three schemes
    schemes = ['upwind', 'explicit', 'cole_hopf']
    solutions = {}
    
    for scheme in schemes:
        print(f"\nSolving with {scheme.upper()} scheme...")
        
        solver = BurgersSolver1D(length=L, viscosity=viscosity)
        u, t, x_plot = solver.solve(u_initial, T, scheme=scheme, num_output=5)
        
        solutions[scheme] = {
            'u': u,
            't': t,
            'x': x_plot,
        }
        
        print(f"  Max solution value: {np.max(u):.6f}")
        print(f"  Min solution value: {np.min(u):.6f}")
        print(f"  Time steps: {len(t)}")
    
    # Plot comparison
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, scheme in enumerate(schemes):
        ax = axes[idx]
        u, t, x_plot = solutions[scheme]['u'], solutions[scheme]['t'], solutions[scheme]['x']
        
        # Plot solution at different times
        for i, (time, solution) in enumerate(zip(t, u)):
            color = plt.cm.viridis(i / len(t))
            ax.plot(x_plot, solution, label=f't={time:.3f}', color=color, linewidth=2)
        
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"{scheme.upper()} Scheme")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('burgers_1d_schemes_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: burgers_1d_schemes_comparison.png")
    plt.show()


def example_burgers_cole_hopf():
    """
    Demonstrate the Cole-Hopf transformation for Burgers equation.
    
    The transformation u = -2ν (∂ln(φ)/∂x) converts Burgers to linear heat.
    """
    print("\n" + "=" * 70)
    print("Example: Cole-Hopf Transformation for Burgers Equation")
    print("=" * 70)
    
    L = 1.0
    nx = 101
    viscosity = 0.01
    T = 0.5
    
    x = np.linspace(0, L, nx)
    # Smooth initial condition
    u_initial = np.sin(2 * np.pi * x) * np.exp(-2 * x)
    
    print(f"\nProblem:")
    print(f"  Burgers equation with smooth initial condition")
    print(f"  Viscosity: ν = {viscosity}")
    print(f"  Cole-Hopf transformation converts to linear heat equation")
    
    print(f"\nSolving with Cole-Hopf transformation...")
    solver = BurgersSolver1D(length=L, viscosity=viscosity)
    u, t, x_plot = solver.solve(u_initial, T, scheme='cole_hopf', num_output=6)
    
    # Plot evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Solution evolution
    for i, (time, solution) in enumerate(zip(t, u)):
        color = plt.cm.plasma(i / len(t))
        ax1.plot(x_plot, solution, label=f't={time:.3f}', color=color, linewidth=2)
    
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("u(x,t)", fontsize=12)
    ax1.set_title("Cole-Hopf Solution: Burgers Equation", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Contour/heatmap of full solution
    u_full = solver.solution['u']
    # Note: u_full already has correct dimensions from solve()
    im = ax2.contourf(x_plot, t, u_full, levels=20, cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('u(x,t)')
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("t", fontsize=12)
    ax2.set_title("Solution Space-Time Domain", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('burgers_1d_cole_hopf.png', dpi=150, bbox_inches='tight')
    print("Saved: burgers_1d_cole_hopf.png")
    plt.show()


if __name__ == "__main__":
    example_burgers_comparison()
    example_burgers_cole_hopf()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
