#!/usr/bin/env python3
"""
Spray Coverage Simulation - Main Entry Point

A professional spray nozzle coverage analysis tool with interactive visualization.

Usage:
    python main.py

Features:
    - Interactive parameter adjustment via sliders
    - Real-time spray pattern visualization
    - Grid-based nozzle arrangement
    - Professional modular architecture

Requirements:
    - numpy
    - matplotlib
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import main


if __name__ == "__main__":
    """Entry point when script is run directly."""
    main()
