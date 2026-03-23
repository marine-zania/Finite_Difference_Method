"""Example: 2D Heat Equation

Solves the 2D heat equation on a square domain using FDM.

∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) on [0,1]² × [0, T]

Initial condition: u(x,y,0) = sin(πx)*sin(πy)
Boundary conditions: u = 0 on all boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fdm.solvers import HeatSolver2D


def example_2d_heat():
    """
    Solve 2D heat equation on unit square.
    """
    print("=" * 70)
    print("Example: 2D Heat Equation on Unit Square")
    print("=" * 70)
    
    # Problem parameters
    Lx = 1.0
    Ly = 1.0
    nx = 31
    ny = 31
    alpha = 0.1
    T = 0.2
    
    # Initial condition: u(x,y,0) = sin(πx)*sin(πy)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    u_initial = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    print(f"\nProblem setup:")
    print(f"  Domain: [0, {Lx}] × [0, {Ly}] × [0, {T}]")
    print(f"  Grid: {nx} × {ny} points")
    print(f"  Diffusivity: α = {alpha}")
    print(f"  Initial condition: u(x,y,0) = sin(πx)sin(πy)")
    print(f"  Boundary conditions: u = 0 on all edges")
    
    print(f"\nSolving 2D heat equation...")
    solver = HeatSolver2D(Lx, Ly, diffusivity=alpha)
    u, t, X_plot, Y_plot = solver.solve(u_initial, T, num_output=5)
    
    print(f"  Max solution value: {np.max(u):.6f}")
    print(f"  Min solution value: {np.min(u):.6f}")
    print(f"  Time steps computed: {len(t)}")
    
    # Plot solution at different times
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(15, 10))
    
    for i, (time, solution) in enumerate(zip(t, u)):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        surf = ax.plot_surface(X_plot, Y_plot, solution, cmap='viridis', 
                              vmin=0, vmax=np.max(u_initial))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y,t)')
        ax.set_zlim(0, np.max(u_initial) * 1.1)
        ax.set_title(f't = {time:.4f}')
    
    plt.tight_layout()
    plt.savefig('heat_2d_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: heat_2d_evolution.png")
    plt.show()


if __name__ == "__main__":
    example_2d_heat()
    print("\nExample completed successfully!")
