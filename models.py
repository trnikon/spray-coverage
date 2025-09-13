"""
Spray coverage simulation model.

This module contains the core simulation logic for spray nozzle coverage analysis.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import multiprocessing as mp
from functools import partial
from config import (
    DEFAULT_N_NOZZLES, DEFAULT_DISTANCE, DEFAULT_CONE_ANGLE, DEFAULT_N_DROPLETS,
    DEFAULT_SURFACE_SIZE, DEFAULT_GRAVITY, DEFAULT_NOZZLE_SPACING,
    DEFAULT_SPRAY_DENSITY, GRID_RESOLUTION, DROPLET_SIZE, DROPLET_WEIGHT
)


class SpraySimulator:
    """Handles spray coverage simulation with configurable parameters."""

    def __init__(self, surface_size: float = DEFAULT_SURFACE_SIZE,
                 grid_resolution: int = GRID_RESOLUTION):
        """
        Initialize the spray simulator.

        Args:
            surface_size: Size of the surface to simulate (units)
            grid_resolution: Resolution of the simulation grid
        """
        self.surface_size = surface_size
        self.grid_resolution = grid_resolution
        self._setup_grid()

    def _setup_grid(self) -> None:
        """Set up the coordinate grid for simulation."""
        self.x = np.linspace(-self.surface_size/2, self.surface_size/2, self.grid_resolution)
        self.y = np.linspace(-self.surface_size/2, self.surface_size/2, self.grid_resolution)

    def simulate(self, n_nozzles: int = DEFAULT_N_NOZZLES,
                 distance: float = DEFAULT_DISTANCE,
                 cone_angle: float = DEFAULT_CONE_ANGLE,
                 n_droplets: int = DEFAULT_N_DROPLETS,
                 gravity: float = DEFAULT_GRAVITY,
                 nozzle_spacing: float = DEFAULT_NOZZLE_SPACING,
                 spray_density: float = DEFAULT_SPRAY_DENSITY,
                 spray_pattern: str = 'flat_fan') -> np.ndarray:
        """
        Simulate spray coverage for given parameters.

        Args:
            n_nozzles: Number of spray nozzles
            distance: Height of nozzles above surface
            cone_angle: Spray cone angle in degrees
            n_droplets: Base number of droplets per nozzle
            gravity: Gravitational acceleration
            nozzle_spacing: Distance between nozzles in grid
            spray_density: Multiplier for droplet density
            spray_pattern: Type of spray pattern ('flat_fan', 'full_cone', 'hollow_cone', 'solid_stream')

        Returns:
            2D array representing spray coverage intensity
        """
        surface = np.zeros((self.grid_resolution, self.grid_resolution))
        adjusted_droplets = int(n_droplets * spray_density)

        nozzle_positions = self._calculate_nozzle_positions(n_nozzles, nozzle_spacing)

        for nozzle_x, nozzle_y in nozzle_positions:
            self._simulate_single_nozzle(
                surface, nozzle_x, nozzle_y, distance, cone_angle,
                adjusted_droplets, gravity, spray_pattern
            )

        return surface

    def _calculate_nozzle_positions(self, n_nozzles: int,
                                  spacing: float) -> List[Tuple[float, float]]:
        """
        Calculate positions for nozzles in a grid pattern with proper center alignment.

        Args:
            n_nozzles: Number of nozzles to position
            spacing: Distance between adjacent nozzles

        Returns:
            List of (x, y) positions for each nozzle
        """
        if n_nozzles == 1:
            # Single nozzle at center
            return [(0.0, 0.0)]

        # Calculate optimal grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_nozzles)))

        # Create all possible positions in the grid
        all_positions = []
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate position relative to grid center
                x = (col - (grid_size - 1) / 2) * spacing
                y = (row - (grid_size - 1) / 2) * spacing
                all_positions.append((x, y, row, col))

        # For non-perfect squares, select the most centered positions
        if n_nozzles == grid_size * grid_size:
            # Perfect square grid - use all positions
            positions = [(x, y) for x, y, _, _ in all_positions]
        else:
            # Select positions that create the most balanced/centered pattern
            positions = self._select_centered_positions(all_positions, n_nozzles, grid_size, spacing)

        return positions[:n_nozzles]

    def _select_centered_positions(self, all_positions: List[Tuple[float, float, int, int]],
                                 n_nozzles: int, grid_size: int, spacing: float) -> List[Tuple[float, float]]:
        """
        Select the most centered positions from available grid positions.

        Args:
            all_positions: List of (x, y, row, col) tuples for all grid positions
            n_nozzles: Number of nozzles needed
            grid_size: Size of the grid
            spacing: Distance between adjacent nozzles

        Returns:
            List of (x, y) positions that create the most centered pattern
        """
        # Calculate distance from center for each position
        center = (grid_size - 1) / 2
        positions_with_distance = []

        for x, y, row, col in all_positions:
            # Distance from grid center
            distance_from_center = np.sqrt((row - center)**2 + (col - center)**2)
            positions_with_distance.append((x, y, distance_from_center, row, col))

        # Sort by distance from center (closest first)
        positions_with_distance.sort(key=lambda p: p[2])

        # For common cases, use optimized patterns
        if n_nozzles == 2:
            # Two nozzles: place them symmetrically
            return [(-spacing/2, 0.0), (spacing/2, 0.0)]
        elif n_nozzles == 3:
            # Three nozzles: triangle pattern
            return [(0.0, 0.0), (-spacing/2, -spacing/2), (spacing/2, -spacing/2)]
        elif n_nozzles == 5:
            # Five nozzles: center plus four corners of inner square
            return [(0.0, 0.0), (-spacing, 0.0), (spacing, 0.0), (0.0, -spacing), (0.0, spacing)]
        elif n_nozzles == 6:
            # Six nozzles: 2x3 rectangle centered
            return [(-spacing/2, -spacing), (-spacing/2, 0.0), (-spacing/2, spacing),
                   (spacing/2, -spacing), (spacing/2, 0.0), (spacing/2, spacing)]
        elif n_nozzles == 7:
            # Seven nozzles: hexagonal pattern with center
            positions = [(0.0, 0.0)]  # Center
            for i in range(6):
                angle = i * np.pi / 3
                x = spacing * np.cos(angle)
                y = spacing * np.sin(angle)
                positions.append((x, y))
            return positions[:7]
        elif n_nozzles == 8:
            # Eight nozzles: 3x3 grid without center
            return [(x, y) for x, y, _, _, _ in positions_with_distance
                   if not (abs(x) < spacing/4 and abs(y) < spacing/4)][:8]
        else:
            # Default: use closest to center positions
            return [(x, y) for x, y, _, _, _ in positions_with_distance]

    def _simulate_single_nozzle(self, surface: np.ndarray, nozzle_x: float, nozzle_y: float,
                               distance: float, cone_angle: float, n_droplets: int,
                               gravity: float, spray_pattern: str) -> None:
        """
        Simulate spray coverage from a single nozzle.

        Args:
            surface: Surface grid to accumulate coverage
            nozzle_x, nozzle_y: Nozzle position
            distance: Height of nozzle above surface
            cone_angle: Spray cone angle in degrees
            n_droplets: Number of droplets to simulate
            gravity: Gravitational acceleration
            spray_pattern: Type of spray pattern ('flat_fan', 'full_cone', 'hollow_cone', 'solid_stream')
        """
        # Generate droplet directions based on spray pattern
        if spray_pattern == 'flat_fan':
            # Flat fan: droplets in a wide, thin fan shape (mostly in one plane)
            theta = np.random.uniform(-np.pi/6, np.pi/6, n_droplets)  # 60-degree fan
            phi = np.random.uniform(0, np.radians(cone_angle), n_droplets)
        elif spray_pattern == 'full_cone':
            # Full cone: uniform distribution across entire cone
            theta = np.random.uniform(0, 2*np.pi, n_droplets)
            phi = np.random.uniform(0, np.radians(cone_angle), n_droplets)
        elif spray_pattern == 'hollow_cone':
            # Hollow cone: ring pattern - no droplets in center
            theta = np.random.uniform(0, 2*np.pi, n_droplets)
            min_phi = np.radians(cone_angle * 0.6)  # Inner boundary (60% of max angle)
            max_phi = np.radians(cone_angle)        # Outer boundary (100% of max angle)
            phi = np.random.uniform(min_phi, max_phi, n_droplets)
        elif spray_pattern == 'solid_stream':
            # Solid stream: very tight, concentrated stream
            theta = np.random.normal(0, np.radians(2), n_droplets)  # Very narrow spread
            phi = np.random.normal(0, np.radians(cone_angle * 0.1), n_droplets)  # Minimal cone angle
            # Clamp values to prevent negative angles
            phi = np.abs(phi)
            theta = np.clip(theta, -np.pi/12, np.pi/12)  # Limit to ±15 degrees
        else:
            # Default to flat fan
            theta = np.random.uniform(-np.pi/6, np.pi/6, n_droplets)
            phi = np.random.uniform(0, np.radians(cone_angle), n_droplets)

        # Calculate direction vectors
        dx = np.sin(phi) * np.cos(theta)
        dy = np.sin(phi) * np.sin(theta)
        dz = -np.cos(phi)

        # Calculate impact points
        t = -distance / dz
        hit_x = nozzle_x + dx * t
        hit_y = nozzle_y + dy * t

        # Apply gravity correction (simplified)
        if gravity > 0:
            # Basic gravity effect - could be enhanced
            pass

        # Convert to grid indices and apply droplet impacts
        self._apply_droplet_impacts(surface, hit_x, hit_y, spray_pattern)

    def _apply_droplet_impacts(self, surface: np.ndarray, hit_x: np.ndarray,
                              hit_y: np.ndarray, spray_pattern: str) -> None:
        """
        Apply droplet impacts to the surface grid with realistic spreading.

        Args:
            surface: Surface grid to modify
            hit_x, hit_y: Arrays of impact coordinates
            spray_pattern: Type of spray pattern ('flat_fan', 'full_cone', 'hollow_cone', 'solid_stream')
        """
        xi = np.searchsorted(self.x, hit_x)
        yi = np.searchsorted(self.y, hit_y)
        valid = ((xi >= 0) & (xi < surface.shape[1]) &
                 (yi >= 0) & (yi < surface.shape[0]))

        for x_i, y_i in zip(xi[valid], yi[valid]):
            self._add_droplet_impact(surface, x_i, y_i, spray_pattern)

    def _add_droplet_impact(self, surface: np.ndarray, center_x: int, center_y: int, spray_pattern: str) -> None:
        """
        Add a single droplet impact with realistic spreading.

        Args:
            surface: Surface grid to modify
            center_x, center_y: Center of impact
            spray_pattern: Type of spray pattern ('flat_fan', 'full_cone', 'hollow_cone', 'solid_stream')
        """
        if spray_pattern == 'flat_fan':
            self._add_flat_fan_impact(surface, center_x, center_y)
        elif spray_pattern == 'full_cone':
            self._add_full_cone_impact(surface, center_x, center_y)
        elif spray_pattern == 'hollow_cone':
            self._add_hollow_cone_impact(surface, center_x, center_y)
        elif spray_pattern == 'solid_stream':
            self._add_solid_stream_impact(surface, center_x, center_y)
        else:
            # Default to flat fan if unknown pattern
            self._add_flat_fan_impact(surface, center_x, center_y)

    def _add_flat_fan_impact(self, surface: np.ndarray, center_x: int, center_y: int) -> None:
        """
        Add impact pattern for flat fan nozzle - creates an elongated oval pattern.

        Args:
            surface: Surface grid to modify
            center_x, center_y: Center of impact
        """
        # Flat fan pattern - elongated oval shape (wider horizontally)
        for dx in range(-DROPLET_SIZE * 2, DROPLET_SIZE * 2 + 1):
            for dy in range(-DROPLET_SIZE, DROPLET_SIZE + 1):
                # Create elliptical pattern (2:1 aspect ratio)
                ellipse_test = (dx / (DROPLET_SIZE * 2))**2 + (dy / DROPLET_SIZE)**2
                if ellipse_test <= 1.0:
                    new_x = center_x + dx
                    new_y = center_y + dy

                    if (0 <= new_x < surface.shape[1] and
                        0 <= new_y < surface.shape[0]):
                        # Weight based on distance from center, with higher intensity at edges
                        distance_from_center = np.sqrt((dx / 2)**2 + dy**2)
                        # Flat fans have more uniform distribution with slight edge weighting
                        weight = DROPLET_WEIGHT * (0.7 + 0.3 * (1 - ellipse_test))
                        surface[new_y, new_x] += weight

    def _add_full_cone_impact(self, surface: np.ndarray, center_x: int, center_y: int) -> None:
        """
        Add impact pattern for full cone nozzle - creates uniform circular pattern.

        Args:
            surface: Surface grid to modify
            center_x, center_y: Center of impact
        """
        # Full cone pattern - uniform circular distribution with center concentration
        for dx in range(-DROPLET_SIZE, DROPLET_SIZE + 1):
            for dy in range(-DROPLET_SIZE, DROPLET_SIZE + 1):
                distance = np.sqrt(dx*dx + dy*dy)
                if distance <= DROPLET_SIZE:
                    new_x = center_x + dx
                    new_y = center_y + dy

                    if (0 <= new_x < surface.shape[1] and
                        0 <= new_y < surface.shape[0]):
                        # Full cone has highest intensity at center, decreasing outward
                        center_weight = 1.5 if distance < DROPLET_SIZE * 0.3 else 1.0
                        weight = DROPLET_WEIGHT * center_weight * (1 - distance / (DROPLET_SIZE + 1))
                        surface[new_y, new_x] += weight

    def _add_hollow_cone_impact(self, surface: np.ndarray, center_x: int, center_y: int) -> None:
        """
        Add impact pattern for hollow cone nozzle - creates a ring pattern with hollow center.

        Args:
            surface: Surface grid to modify
            center_x, center_y: Center of impact
        """
        # Hollow cone pattern - ring pattern with hollow center
        # Inner radius should be at least 40% of outer radius to create visible hollow effect
        outer_radius = DROPLET_SIZE
        inner_radius = DROPLET_SIZE * 0.4  # 40% inner radius for visible hollow center

        for dx in range(-DROPLET_SIZE, DROPLET_SIZE + 1):
            for dy in range(-DROPLET_SIZE, DROPLET_SIZE + 1):
                distance = np.sqrt(dx*dx + dy*dy)
                # Only add droplets in the ring area (between inner and outer radius)
                if inner_radius <= distance <= outer_radius:
                    new_x = center_x + dx
                    new_y = center_y + dy

                    if (0 <= new_x < surface.shape[1] and
                        0 <= new_y < surface.shape[0]):
                        # Weight decreases with distance from optimal ring position
                        ring_center = (inner_radius + outer_radius) / 2
                        ring_width = outer_radius - inner_radius
                        distance_from_ring_center = abs(distance - ring_center)
                        weight_factor = max(0, 1 - (distance_from_ring_center / (ring_width / 2)))
                        weight = DROPLET_WEIGHT * weight_factor
                        surface[new_y, new_x] += weight

    def _add_solid_stream_impact(self, surface: np.ndarray, center_x: int, center_y: int) -> None:
        """
        Add impact pattern for solid stream nozzle - creates very concentrated circular pattern.

        Args:
            surface: Surface grid to modify
            center_x, center_y: Center of impact
        """
        # Solid stream pattern - very concentrated, small circular area
        small_radius = max(1, DROPLET_SIZE // 2)  # Much smaller than other patterns

        for dx in range(-small_radius, small_radius + 1):
            for dy in range(-small_radius, small_radius + 1):
                distance = np.sqrt(dx*dx + dy*dy)
                if distance <= small_radius:
                    new_x = center_x + dx
                    new_y = center_y + dy

                    if (0 <= new_x < surface.shape[1] and
                        0 <= new_y < surface.shape[0]):
                        # Very high intensity, concentrated at center
                        if distance == 0:
                            weight = DROPLET_WEIGHT * 3.0  # Maximum intensity at center
                        elif distance <= small_radius * 0.5:
                            weight = DROPLET_WEIGHT * 2.0  # High intensity near center
                        else:
                            weight = DROPLET_WEIGHT * 1.0  # Normal intensity at edges
                        surface[new_y, new_x] += weight

    def optimize_nozzle_layout(self, initial_nozzle_count: int = DEFAULT_N_NOZZLES,
                              max_nozzles: int = 100, target_coverage: float = 0.95) -> Tuple[int, float]:
        """
        Optimize the nozzle layout for uniform spray coverage.

        Args:
            initial_nozzle_count: Starting number of nozzles for optimization
            max_nozzles: Maximum number of nozzles to consider
            target_coverage: Desired coverage fraction (0 to 1)

        Returns:
            Tuple of (optimal nozzle count, achieved coverage)
        """
        best_nozzle_count = initial_nozzle_count
        best_coverage = 0.0

        for n_nozzles in range(1, max_nozzles + 1):
            simulated_surface = self.simulate(n_nozzles=n_nozzles)
            coverage = np.sum(simulated_surface)

            if coverage >= target_coverage * self.surface_size**2:
                return n_nozzles, coverage

            # Update best found solution
            if coverage > best_coverage:
                best_coverage = coverage
                best_nozzle_count = n_nozzles

        return best_nozzle_count, best_coverage

    def optimize_for_uniformity(self, current_params: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """
        Optimize spray parameters for maximum uniformity using multiprocessing.

        Args:
            current_params: Current parameter values from sliders
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with optimized parameters
        """
        print("Starting parallel optimization for uniform coverage...")

        # Search ranges for optimization
        angle_range = np.linspace(10, 70, 12)  # Reduced for faster optimization
        spacing_range = np.linspace(1.0, 8.0, 15)  # Reduced for faster optimization

        # Create parameter combinations
        param_combinations = []
        for angle in angle_range:
            for spacing in spacing_range:
                test_params = current_params.copy()
                test_params['cone_angle'] = angle
                test_params['nozzle_spacing'] = spacing
                param_combinations.append(test_params)

        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")

        # Use multiprocessing for parallel optimization
        n_workers = min(mp.cpu_count(), 4)  # Limit workers to prevent overwhelming system

        results = []
        completed = 0

        # Process in chunks to provide progress updates
        chunk_size = max(1, total_combinations // 20)  # 20 progress updates

        with mp.Pool(processes=n_workers) as pool:
            # Submit all jobs
            jobs = [pool.apply_async(self._evaluate_parameters, (params,)) for params in param_combinations]

            # Collect results with progress tracking
            for job in jobs:
                uniformity = job.get()
                results.append(uniformity)
                completed += 1

                # Update progress every chunk_size completions
                if completed % chunk_size == 0 or completed == total_combinations:
                    progress = completed / total_combinations
                    if progress_callback:
                        progress_callback(progress)
                    print(f"Progress: {completed}/{total_combinations} ({progress*100:.1f}%)")

        # Find best result
        best_uniformity = max(results)
        best_index = results.index(best_uniformity)
        best_params = param_combinations[best_index]

        print(f"Optimization complete! Best uniformity: {best_uniformity:.3f}")
        return best_params

    def _evaluate_parameters(self, test_params: Dict[str, Any]) -> float:
        """
        Evaluate uniformity for given parameters (used by multiprocessing).

        Args:
            test_params: Parameters to test

        Returns:
            Uniformity score
        """
        surface = self.simulate(**test_params)
        return self._calculate_uniformity(surface)

    def _calculate_uniformity(self, surface: np.ndarray) -> float:
        """
        Calculate uniformity metric for spray coverage.
        Higher values indicate more uniform coverage.

        Args:
            surface: 2D array of spray coverage

        Returns:
            Uniformity score (0-1, where 1 is perfectly uniform)
        """
        # Ignore areas with no coverage
        covered_areas = surface[surface > 0]

        if len(covered_areas) == 0:
            return 0.0

        # Calculate coefficient of variation (lower is more uniform)
        mean_coverage = np.mean(covered_areas)
        std_coverage = np.std(covered_areas)

        if mean_coverage == 0:
            return 0.0

        cv = std_coverage / mean_coverage

        # Convert to uniformity score (inverse of coefficient of variation)
        # Add small constant to avoid division by zero
        uniformity = 1.0 / (1.0 + cv)

        # Bonus for higher coverage area
        coverage_ratio = len(covered_areas) / surface.size

        return uniformity * coverage_ratio
