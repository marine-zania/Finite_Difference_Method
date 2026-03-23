"""
Tests for FDM solvers
"""

import pytest
import numpy as np
from fdm.solvers import HelmholtzSolver2D
from fdm.utilities import analysis, visualization


class TestHelmholtzSolver2D:
    """Tests for 2D Helmholtz solver."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.a = 19.05  # WR90 x dimension
        self.b = 9.525  # WR90 y dimension
        self.solver = HelmholtzSolver2D(self.a, self.b)
    
    def test_initialization(self):
        """Test solver initialization."""
        assert self.solver.a == self.a
        assert self.solver.b == self.b
        assert self.solver.eigvals is None
    
    def test_setup_grid(self):
        """Test grid setup."""
        N = 100
        self.solver.setup_grid(N)
        
        assert self.solver.N == N
        expected_h = self.a / (N + 1)
        assert np.isclose(self.solver.hx, expected_h)
        assert np.isclose(self.solver.hy, self.b / (N + 1))
    
    def test_construct_matrix(self):
        """Test finite-difference matrix construction."""
        N = 50
        self.solver.setup_grid(N)
        A = self.solver.construct_matrix()
        
        # Check sparse matrix properties
        assert A.shape == (N*N, N*N)
        assert A.format == 'csc'  # Compressed sparse column format
    
    def test_solve(self):
        """Test eigenvalue solver."""
        N = 50
        num_modes = 8
        
        eigvals, eigvecs = self.solver.solve(N, num_modes=num_modes)
        
        assert len(eigvals) == num_modes
        assert eigvecs.shape == (N*N, num_modes)
        # Eigenvalues should be positive and increasing
        assert np.all(eigvals > 0)
        assert np.all(eigvals[:-1] <= eigvals[1:])
    
    def test_theoretical_k_squared(self):
        """Test theoretical k² computation."""
        modes = [(1, 1), (2, 1), (3, 1)]
        k2_theory = self.solver.get_theoretical_k_squared(modes)
        
        assert len(k2_theory) == len(modes)
        
        # First mode should have smallest k²
        assert k2_theory[0] < k2_theory[1]
        assert k2_theory[1] < k2_theory[2]
        
        # Check manual calculation for first mode
        expected_k2_11 = (np.pi / self.a)**2 + (np.pi / self.b)**2
        assert np.isclose(k2_theory[0], expected_k2_11)
    
    def test_relative_error(self):
        """Test relative error computation."""
        N = 100
        modes = [(1, 1), (2, 1), (3, 1)]
        
        self.solver.solve(N, num_modes=10)
        k2_theory = self.solver.get_theoretical_k_squared(modes)
        errors = self.solver.compute_relative_error(k2_theory, num_modes=3)
        
        assert len(errors) == 3
        assert np.all(errors > 0)  # Errors should be positive
        assert np.all(errors < 0.1)  # Should be reasonably small
    
    def test_reshape_mode(self):
        """Test mode reshaping."""
        N = 50
        self.solver.solve(N, num_modes=5)
        
        field = self.solver.reshape_mode(0)
        
        # Should be padded to (N+2) x (N+2)
        assert field.shape == (N+2, N+2)
        # Boundaries should be zero
        assert np.all(field[0, :] == 0)
        assert np.all(field[-1, :] == 0)
        assert np.all(field[:, 0] == 0)
        assert np.all(field[:, -1] == 0)
    
    def test_normalize_mode(self):
        """Test mode normalization."""
        field = np.array([[-1, 0.5], [0.25, 1.0]])
        normalized = self.solver.normalize_mode(field)
        
        # Max absolute value should be 1
        assert np.isclose(np.max(np.abs(normalized)), 1.0)


class TestAnalysisUtils:
    """Tests for analysis utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = HelmholtzSolver2D(19.05, 9.525)
        self.solver.solve(100, num_modes=5)
    
    def test_convergence_analysis(self):
        """Test convergence analysis function."""
        N_values = [50, 75, 100]
        k2_theory = self.solver.get_theoretical_k_squared([(1,1), (2,1), (3,1)])
        
        results = analysis.convergence_analysis(
            self.solver, N_values, num_modes=3, theoretical_k2=k2_theory
        )
        
        assert 'N_values' in results
        assert 'h_values' in results
        assert 'computed_k2' in results
        assert 'relative_errors' in results
        
        assert len(results['N_values']) == len(N_values)
        assert len(results['h_values']) == len(N_values)


class TestVisualization:
    """Tests for visualization utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = HelmholtzSolver2D(19.05, 9.525)
        self.solver.solve(50, num_modes=8)
    
    def test_plot_modes_no_crash(self):
        """Test that plot_modes doesn't crash."""
        import matplotlib.pyplot as plt
        fig = visualization.plot_modes(self.solver, [0, 1, 2, 3])
        assert fig is not None
        plt.close(fig)
    
    def test_plot_convergence_no_crash(self):
        """Test that plot_convergence doesn't crash."""
        import matplotlib.pyplot as plt
        errors = {
            'Mode 1': [0.1, 0.05, 0.01],
            'Mode 2': [0.2, 0.08, 0.02],
        }
        fig = visualization.plot_convergence([0.1, 0.05, 0.01], errors)
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
