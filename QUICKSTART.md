# Quick Start Guide

## 5-Minute Setup

### 1. Install the Package

```bash
cd /home/gazania/zania_folder/2D_Helmholtz_eqnFDM
pip install -e .
```

### 2. Run Your First Example

```bash
python run.py helmholtz
```

You'll see:
- Helmholtz equation solved for WR90 waveguide  
- Mode shapes plotted
- Eigenvalues compared with theory

### 3. Try Heat Equation

```bash
python run.py heat-1d
```

Compares three numerical methods.

### 4. Explore Burgers Equation

```bash
python run.py burgers
```

Shows nonlinear PDE with Cole-Hopf transformation.

## Usage Examples

### Example 1: Solve Helmholtz Equation

```python
from fdm.solvers import HelmholtzSolver2D

# Create solver for rectangular waveguide
solver = HelmholtzSolver2D(length_a=19.05, length_b=9.525)

# Solve eigenvalue problem
eigvals, eigvecs = solver.solve(N=150, num_modes=20)

# Get theoretical values
modes = [(1,1), (2,1), (3,1)]
k2_theory = solver.get_theoretical_k_squared(modes)

# Check accuracy
errors = solver.compute_relative_error(k2_theory, num_modes=3)
print(f"Relative errors: {errors}")
```

### Example 2: Heat Equation with Different Methods

```python
from fdm.solvers import HeatSolver1D
import numpy as np

# Domain and parameters
L = 4.0
x = np.linspace(0, L, 41)
u_init = 0.5 * x * (L - x)

solver = HeatSolver1D(length=L, diffusivity=1/16)

# Method 1: Explicit (FTCS)
u1, t1, x1 = solver.solve(u_init, t_final=0.5, 
                          method='explicit', dt=0.001)

# Method 2: Implicit (BTCS)  
u2, t2, x2 = solver.solve(u_init, t_final=0.5,
                          method='implicit')

# Method 3: Crank-Nicolson (best accuracy)
u3, t3, x3 = solver.solve(u_init, t_final=0.5,
                          method='crank_nicolson')
```

### Example 3: Burgers Equation with Cole-Hopf

```python
from fdm.solvers import BurgersSolver1D
import numpy as np

L = 1.0
x = np.linspace(0, L, 101)
u_init = np.exp(-100 * (x - 0.5)**2)  # Gaussian bump

solver = BurgersSolver1D(length=L, viscosity=0.01)

# Cole-Hopf transformation (most accurate)
u, t, x_grid = solver.solve(u_init, t_final=0.5, 
                            scheme='cole_hopf')

print(f"Computed at {len(t)} time points")
print(f"Max solution: {np.max(u):.6f}")
```

### Example 4: Visualization

```python
from fdm.solvers import HelmholtzSolver2D
from fdm.utilities import plot_modes, plot_convergence

solver = HelmholtzSolver2D(19.05, 9.525)
solver.solve(N=150, num_modes=8)

# Plot first 8 modes
plot_modes(solver, mode_indices=list(range(8)))

# Convergence analysis
from fdm.utilities import convergence_analysis
results = convergence_analysis(solver, N_values=[50,100,150,200],
                              num_modes=5)
```

## Command Line Usage

```bash
# List all available examples
python run.py list

# Run specific example
python run.py helmholtz
python run.py heat-1d
python run.py heat-2d
python run.py burgers
python run.py convergence

# Run everything
python run.py all
```

## Common Workflows

### Workflow A: Mode Analysis & Visualization

```python
from fdm.solvers import HelmholtzSolver2D
from fdm.utilities import plot_modes, validate_against_theoretical

solver = HelmholtzSolver2D(19.05, 9.525)
eigvals, eigvecs = solver.solve(N=150, num_modes=20)

# Visualize modes
plot_modes(solver, mode_indices=[0,1,2,3], modes_per_row=2)

# Validate
modes = [(1,1), (2,1), (3,1), (1,2)]
theory = solver.get_theoretical_k_squared(modes)
table = validate_against_theoretical(solver, theory, num_modes=4)
print(table)
```

### Workflow B: Convergence Study

```python
from fdm.solvers import HelmholtzSolver2D
from fdm.utilities import convergence_analysis, plot_convergence

solver = HelmholtzSolver2D(19.05, 9.525)
modes = [(1,1), (2,1), (3,1), (1,2)]
k2_theory = solver.get_theoretical_k_squared(modes)

results = convergence_analysis(solver, 
                              N_values=[50, 75, 100, 150, 200],
                              num_modes=4,
                              theoretical_k2=k2_theory)

plot_convergence(results['h_values'], 
                {f"TE{m}": results['relative_errors'][:, i]
                 for i, m in enumerate(modes)})
```

### Workflow C: Comparing Heat Equation Methods

```python
from fdm.solvers import HeatSolver1D
import matplotlib.pyplot as plt
import numpy as np

L, nx = 4.0, 41
x = np.linspace(0, L, nx)
u_init = 0.5 * x * (L - x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, method in enumerate(['explicit', 'implicit', 'crank_nicolson']):
    solver = HeatSolver1D(L, 1/16)
    u, t, x_grid = solver.solve(u_init, 0.5, method=method, num_output=5)
    
    ax = axes[idx]
    for i, (ti, ui) in enumerate(zip(t, u)):
        ax.plot(x_grid, ui, label=f't={ti:.3f}')
    ax.set_title(f"{method.upper()}")
    ax.legend()
    ax.grid()
```

## Documentation References

Quick navigation to find what you need:

| Need | File |
|------|------|
| How to install? | README.md |
| API reference? | MODULE_INDEX.md |
| Algorithm details? | ALGORITHM_REFERENCE.md |
| Examples? | See `fdm/examples/` |
| Tests? | Run `pytest tests/` |
| Project overview? | REPO_GUIDE.md |
| Future plans? | ROADMAP.md |

## Troubleshooting

**Import Error: "No module named 'fdm'"**
```bash
# Install in development mode
pip install -e .
```

**Stability Warning in Heat Solver**
```python
# FTCS method requires r = α·dt/dx² ≤ 0.5
# If you see the warning, either:
# 1. Use smaller dt
# 2. Use 'implicit' or 'crank_nicolson' method
```

**Slow Eigenvalue Computation**
```python
# For large N, eigensolvers take time:
# N=100: ~10ms
# N=200: ~50ms  
# N=300: ~200ms
# Use N=150 as default balance
```

## Important Parameters

### Helmholtz Solver
- `length_a`, `length_b`: Waveguide dimensions
- `N`: Grid points per direction (50-300 typical)
- `num_modes`: How many eigenvalues to find (5-50 typical)

### Heat Solver 1D
- `length`: Domain length
- `diffusivity`: α parameter (0.01-1.0 typical)
- `dt`: Time step (auto-selected if None)
- `method`: 'explicit', 'implicit', or 'crank_nicolson'

### Heat Solver 2D
- `length_x`, `length_y`: Domain dimensions
- `diffusivity`: α parameter
- `dt`: Time step

### Burgers Solver
- `length`: Domain length
- `viscosity`: ν parameter (0.001-0.1 typical)
- `scheme`: 'explicit', 'upwind', or 'cole_hopf'

## Key Features

✓ Eigenvalue solver for Helmholtz
✓ Three methods for 1D heat (explicit/implicit/CN)
✓ Upwind scheme for nonlinear advection
✓ Cole-Hopf transformation for Burgers
✓ Automatic stability checking
✓ Built-in visualization
✓ Convergence analysis tools
✓ Unit tests
✓ Complete documentation

## What's Next?

1. **Try the examples**: `python run.py all`
2. **Read the docs**: Start with README.md
3. **Explore the code**: Look at `fdm/solvers/`
4. **Modify examples**: Create your own problems
5. **Extend solvers**: Add new equations following the pattern

## Getting Help

- **API Questions**: See MODULE_INDEX.md
- **Algorithm Details**: Check ALGORITHM_REFERENCE.md
- **Code Examples**: Run `fdm/examples/`
- **Errors**: Check error message, run tests
- **More Info**: Read docstrings in source code

---

**You're all set!** 🚀

Start with: `python run.py helmholtz`
