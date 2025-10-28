"""
Enhanced ResiLink: SDN Controller Integration
===========================================

SDN controller integration for Enhanced ResiLink with comprehensive monitoring
and academic-grade data collection for network resilience research.

This module provides:
- Academic-grade Ryu controller implementation
- Real-time network monitoring and metrics collection
- Integration with Enhanced ResiLink optimization algorithm
- Performance profiling and scalability analysis

Usage:
    from enhanced_resilink.sdn_controller import AcademicResilientController
    
    # Use with Ryu framework
    ryu-manager academic_controller.py

Authors: Research Team
License: MIT
"""

# Import controllers with optional dependencies
try:
    from .enhanced_academic_controller import EnhancedAcademicController
    __all__ = ["EnhancedAcademicController"]
except ImportError as e:
    print(f"Warning: Could not import EnhancedAcademicController: {e}")
    __all__ = []

try:
    from .academic_controller import AcademicResilientController
    __all__.append("AcademicResilientController")
except ImportError as e:
    print(f"Warning: Could not import AcademicResilientController (Flask required): {e}")
    pass