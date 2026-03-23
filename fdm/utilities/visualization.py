"""Visualization utilities for FDM results"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_modes(solver, mode_indices: list, modes_per_row: int = 4, 
               figsize: tuple = None, cmap: str = 'jet', vmin: float = -1, vmax: float = 1):
    """
    Plot mode shapes from Helmholtz solver.
    
    Parameters
    ----------
    solver : HelmholtzSolver2D
        Solver object with computed modes
    mode_indices : list
        Indices of modes to plot
    modes_per_row : int
        Number of subplots per row
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name
    vmin, vmax : float
        Color axis limits
    """
    num_modes = len(mode_indices)
    num_cols = min(modes_per_row, num_modes)
    num_rows = (num_modes + modes_per_row - 1) // modes_per_row
    
    if figsize is None:
        figsize = (4 * num_cols, 3.5 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()  # flatten for uniform indexing
    
    for idx, mode_idx in enumerate(mode_indices):
        ax = axes[idx]
        
        # Get mode field
        field = solver.reshape_mode(mode_idx)
        field = solver.normalize_mode(field)
        
        # Plot
        im = ax.contourf(solver.grid['x'], solver.grid['y'], field.T, 
                         levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=r"$H_z(x,y)$")
        
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Mode {mode_idx}\n$k^2 = {solver.eigvals[mode_idx]:.3e}$")
    
    # Hide unused subplots
    for idx in range(len(mode_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_field(x, y, field, title: str = "Field", xlabel: str = "x", 
               ylabel: str = "y", cmap: str = 'jet'):
    """
    Plot a 2D scalar field.
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    field : ndarray
        2D field values
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    cmap : str
        Colormap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(x, y, field, levels=50, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Field Value")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_convergence(h_values: list, errors: dict, figsize: tuple = (10, 6)):
    """
    Plot convergence analysis (error vs. grid spacing).
    
    Parameters
    ----------
    h_values : list
        Grid spacing values
    errors : dict
        Dictionary with mode names as keys and error lists as values
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'x', '+', '|']
    
    for idx, (mode_name, error_list) in enumerate(errors.items()):
        marker = markers[idx % len(markers)]
        ax.loglog(h_values, error_list, marker=marker, linestyle='-', label=mode_name)
    
    ax.set_xlabel(r"Grid spacing $h$ (mm)", fontsize=12)
    ax.set_ylabel("Relative Error in $k^2$", fontsize=12)
    ax.set_title("FDM Convergence Analysis", fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
