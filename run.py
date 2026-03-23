"""
Main entry point for running FDM solver examples and utilities.

This script provides a command-line interface to run various examples and analyses.
"""

import sys
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from fdm.examples.helmholtz_rectangular import example_rectangular_modes, example_convergence_analysis
from fdm.examples.heat_1d_example import example_1d_heat_comparison, example_1d_heat_stability_study
from fdm.examples.heat_2d_example import example_2d_heat
from fdm.examples.burgers_1d_example import example_burgers_comparison, example_burgers_cole_hopf


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FDM PDE Solver - Finite Difference Method for Partial Differential Equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Helmholtz examples
  python run.py helmholtz
  
  # Run Heat equation examples
  python run.py heat-1d
  python run.py heat-2d
  
  # Run Burgers equation examples
  python run.py burgers
  
  # Run convergence analysis
  python run.py convergence
  
  # Run all examples
  python run.py all
  
  # List available examples
  python run.py list
        """
    )
    
    parser.add_argument(
        'task',
        nargs='?',
        default='helmholtz',
        choices=['helmholtz', 'heat-1d', 'heat-2d', 'burgers', 'convergence', 'all', 'list'],
        help='Task to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FDM PDE SOLVER - Finite Difference Method Examples")
    print("="*70 + "\n")
    
    try:
        if args.task == 'list':
            print("Available examples:")
            print("  helmholtz        - 2D Helmholtz equation in rectangular waveguide")
            print("  heat-1d          - 1D Heat equation (explicit, implicit, Crank-Nicolson)")
            print("  heat-2d          - 2D Heat equation evolution")
            print("  burgers          - 1D Burgers equation (multiple schemes)")
            print("  convergence      - Helmholtz convergence analysis")
            print("  all              - Run all examples")
            return 0
        
        elif args.task == 'helmholtz':
            print("Running: 2D Helmholtz Equation Examples")
            print("-" * 70)
            example_rectangular_modes()
            
        elif args.task == 'heat-1d':
            print("Running: 1D Heat Equation Examples")
            print("-" * 70)
            example_1d_heat_comparison()
            example_1d_heat_stability_study()
            
        elif args.task == 'heat-2d':
            print("Running: 2D Heat Equation Example")
            print("-" * 70)
            example_2d_heat()
            
        elif args.task == 'burgers':
            print("Running: 1D Burgers Equation Examples")
            print("-" * 70)
            example_burgers_comparison()
            example_burgers_cole_hopf()
            
        elif args.task == 'convergence':
            print("Running: Convergence Analysis")
            print("-" * 70)
            example_convergence_analysis()
            
        elif args.task == 'all':
            print("Running all examples...")
            print("-" * 70)
            print("\n>>> Helmholtz Examples:")
            example_rectangular_modes()
            example_convergence_analysis()
            
            print("\n>>> 1D Heat Equation:")
            example_1d_heat_comparison()
            example_1d_heat_stability_study()
            
            print("\n>>> 2D Heat Equation:")
            example_2d_heat()
            
            print("\n>>> Burgers Equation:")
            example_burgers_comparison()
            example_burgers_cole_hopf()
        
        print("\n" + "="*70)
        print("✓ All tasks completed successfully!")
        print("="*70)
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
