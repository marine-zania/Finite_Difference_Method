# Jupyter Notebooks

This folder contains the original Jupyter notebook files from the project.

## Contents

- **Analytical_Rectangular.ipynb** - Analytical solution for 2D Helmholtz in rectangular waveguides
- **FDM_Rectangular (1).ipynb** - Finite Difference Method solution for rectangular waveguides
- **FDM_Elliptical.ipynb** - FDM solution for elliptical waveguide cross-sections
- **FDM_Trapezoidal.ipynb** - FDM solution for trapezoidal waveguide cross-sections

## Note

These notebooks have been converted to Python modules in the `fdm/` package for better organization and reusability. The core code from these notebooks is now available as:

- `fdm/solvers/helmholtz.py` - HelmholtzSolver2D class
- `fdm/examples/helmholtz_rectangular.py` - Rectangular waveguide example
- `fdm/examples/helmholtz_elliptical.py` - Elliptical waveguide (stub for implementation)

## Usage

**For quick exploration**: Use the Jupyter notebooks in this folder

**For production code**: Import solvers from the `fdm/` package

```python
from fdm.solvers import HelmholtzSolver2D
solver = HelmholtzSolver2D(19.05, 9.525)
eigvals, eigvecs = solver.solve(N=150, num_modes=20)
```

**For running examples**: Execute the CLI tool

```bash
python run.py helmholtz
```

## Converting Notebooks to Scripts

To convert a notebook to a Python script:

```bash
jupyter nbconvert --to script <notebook>.ipynb
```

This will create a `.py` file that can be run directly or integrated into the package.
