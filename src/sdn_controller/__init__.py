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

from .academic_controller import AcademicResilientController

__all__ = ["AcademicResilientController"]