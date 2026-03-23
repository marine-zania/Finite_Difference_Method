# FDM Package Configuration
Readme to understand the consolidated FDM repository structure.

## What's in This Repository?

This is a **unified repository for all Finite Difference Method (FDM) PDE solvers** previously scattered across multiple GitHub projects. It consolidates:

- **2D Helmholtz Equation** solver for waveguide analysis
- **1D & 2D Heat Equations** with multiple numerical schemes
- **1D Burgers Equation** with advanced solution methods
- **Comprehensive utilities** for visualization and convergence analysis

## Quick Navigation

- **Main Package**: `fdm/` - Core solver implementations
- **Solvers**: `fdm/solvers/` - Individual PDE solver modules
- **Utilities**: `fdm/utilities/` - Plotting and analysis tools
- **Examples**: `fdm/examples/` - Runnable demonstration scripts
- **Tests**: `tests/` - Unit tests and validation

## Installation

```bash
pip install -e .
```

## Running Examplesor

```bash
# List all available examples
python run.py list

# Run specific example
python run.py helmholtz
python run.py heat-1d
python run.py heat-2d
python run.py burgers

# Run everything
python run.py all
```

## Key Features

✓ Modular, object-oriented solver design
✓ Efficient sparse matrix operations
✓ Multiple numerical schemes per equation
✓ Built-in visualization utilities
✓ Convergence analysis tools
✓ Production-ready code with tests
✓ Comprehensive documentation

## Problem Classes Supported

### 1. **2D Helmholtz Equation**
Eigenvalue problem for waveguide modes
- Rectangular waveguides (WR90, etc.)
- Arbitrary dimensions  
- Convergence analysis tools
- Comparison with analytical solutions

### 2. **1D Heat Equation**
Three time integration methods:
- **Explicit (FTCS)**: Fast, conditionally stable
- **Implicit (BTCS)**: Unconditionally stable
- **Crank-Nicolson**: Higher-order accuracy

### 3. **2D Heat Equation**
Diffusion on 2D domains:
- Simple explicit scheme (for education)
- Ready for ADI implementation

### 4. **1D Burgers Equation**
Nonlinear advection-diffusion:
- **Explicit centered** scheme
- **Upwind** scheme (stable for steep gradients)
- **Cole-Hopf transformation** (highly accurate, handles shocks)

## Solver Base Classes

All solvers inherit from `BaseSolver` with consistent interface:
- `setup_grid()` - Define computational grid
- `construct_matrix()` - Build FDM system
- `solve()` - Execute solver
- `get_solution()` - Retrieve result

## Documentation

- **Mathematical formulations**: See docstrings in solver classes
- **Implementation details**: Check CODE_MODULES_READY.md
- **Algorithm reference**: See ALGORITHM_REFERENCE.md
- **extraction details**: See GITHUB_CODE_EXTRACTION_SUMMARY.md

## Performance Notes

- Sparse matrix eigensolvers for Helmholtz
- Banded matrix solvers for heat/Burgers
- Automatic stability checking for explicit schemes
- Typical runtimes <1s for standard grids

## Contributing

Add new solvers by:
1. Extending `BaseSolver`
2. Implementing required methods
3. Adding example script
4. Creating unit tests
5. Updating this documentation

## License

MIT License

## References

- Strikwerda, J. C. (2004). Finite Difference Schemes and Partial Differential Equations.
- Trefethen, L. N. (2000). Spectral Methods in MATLAB.
- Haberman, R. (2012). Applied Partial Differential Equations.

---

**This repository was consolidated from:**
- 2D_Helmholtz_eqnFDM
- heat_equation_FDM
- 2D_heat_equation
- 1DBurgers_equation

**Maintained by**: Zania (@marine-zania)
