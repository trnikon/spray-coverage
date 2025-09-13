"""
Configuration constants for the spray coverage simulation.
"""

# Simulation defaults
DEFAULT_N_NOZZLES = 4
DEFAULT_DISTANCE = 10.0
DEFAULT_CONE_ANGLE = 30
DEFAULT_N_DROPLETS = 2000
DEFAULT_SURFACE_SIZE = 20.0
DEFAULT_GRAVITY = 9.81
DEFAULT_NOZZLE_SPACING = 5.0
DEFAULT_SPRAY_DENSITY = 1.0

# Grid resolution
GRID_RESOLUTION = 200

# Droplet simulation parameters
DROPLET_SIZE = 2  # Radius in pixels for droplet impact
DROPLET_WEIGHT = 3  # How much each droplet contributes

# Spray pattern types - most common for liquid nozzles (defined early for use in SLIDER_RANGES)
SPRAY_PATTERNS = {
    'full_cone': 'Full Cone (uniform circular)',
    'flat_fan': 'Flat Fan (elliptical - most common)',
    'hollow_cone': 'Hollow Cone (ring pattern)',
    'solid_stream': 'Solid Stream (concentrated)'
}

# Default spray pattern (full cone for general purpose applications)
DEFAULT_SPRAY_PATTERN = 'full_cone'

# UI configuration
SLIDER_RANGES = {
    'nozzles': {'min': 1, 'max': 25, 'default': DEFAULT_N_NOZZLES, 'step': 1},
    'distance': {'min': 5.0, 'max': 30.0, 'default': DEFAULT_DISTANCE},
    'angle': {'min': 5, 'max': 80, 'default': DEFAULT_CONE_ANGLE},
    'spacing': {'min': 1.0, 'max': 10.0, 'default': DEFAULT_NOZZLE_SPACING},
    'density': {'min': 0.1, 'max': 3.0, 'default': DEFAULT_SPRAY_DENSITY},
    'pattern': {'options': list(SPRAY_PATTERNS.keys()), 'default': DEFAULT_SPRAY_PATTERN}
}

# Visualization settings
COLORMAP = 'hot'
SURFACE_EXTENT = [-10, 10, -10, 10]
UI_BACKGROUND_COLOR = 'lightgoldenrodyellow'
