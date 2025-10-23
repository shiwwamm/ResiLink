#!/usr/bin/env python3
"""
Enhanced ResiLink Setup Script
=============================

Installation script for Enhanced ResiLink hybrid network optimization.
"""

from setuptools import setup, find_packages

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-resilink",
    version="2.0.0",
    author="Research Team",
    author_email="research@university.edu",
    description="Hybrid GNN+RL network resilience optimization with complete academic justification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/enhanced-resilink",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords for discovery
    keywords="network resilience optimization SDN GNN reinforcement learning graph neural networks",
    
    # Scripts
    scripts=["hybrid_resilink_implementation.py"],
    
    # Additional files to include
    include_package_data=True,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/research-team/enhanced-resilink/issues",
        "Source": "https://github.com/research-team/enhanced-resilink",
    },
)