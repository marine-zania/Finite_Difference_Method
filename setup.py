"""Setup script for FDM package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fdm-pde-solvers",
    version="1.0.0",
    author="Zania",
    description="Modular Finite Difference Method solvers for PDEs (Helmholtz, Heat, Burgers)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marine-zania/fdm-pde-solvers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "pandas>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
        ],
    },
)
