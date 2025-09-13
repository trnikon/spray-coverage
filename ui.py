"""
User interface components for the spray coverage simulation.

This module handles the matplotlib-based GUI with interactive sliders and visualization.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, Any

matplotlib.use('TkAgg')

from config import SLIDER_RANGES, COLORMAP, SURFACE_EXTENT, UI_BACKGROUND_COLOR
from models import SpraySimulator


class SprayVisualizationUI:
    """Interactive UI for spray coverage simulation."""

    def __init__(self, simulator: SpraySimulator):
        """
        Initialize the visualization UI.

        Args:
            simulator: SpraySimulator instance to use for calculations
        """
        self.simulator = simulator
        self.fig = None
        self.ax = None
        self.image = None
        self.sliders: Dict[str, Slider] = {}
        self.reset_button = None
        self.optimize_button = None
        self.progress_bar = None
        self.metrics_display_ax = None
        self.pattern_button = None
        self.current_pattern = 'full_cone'  # Default spray pattern changed to full cone

    def setup_ui(self) -> None:
        """Set up the matplotlib figure and UI elements."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('Spray Coverage Simulation', fontsize=14, fontweight='bold')

        # Adjust layout: center graph horizontally with more space below it
        # Left margin reduced to 0.15, right margin to 0.85 to center the graph better
        # Bottom margin increased to 0.48 to give more space for UI elements
        plt.subplots_adjust(left=0.15, bottom=0.48, top=0.85, right=0.85)

        # Create sliders first
        self._create_sliders()

        # Create reset and optimize buttons
        self._create_reset_button()
        self._create_optimize_button()
        self._create_pattern_button()

        # Create progress bar
        self._create_progress_bar()

        # Create coverage metrics display (positioned directly below graph)
        self._create_metrics_display()

        # Set up event handlers
        self._setup_event_handlers()

        # Initialize with default simulation (after sliders are created)
        self._update_visualization()

        # Set up colorbar
        plt.colorbar(self.image, ax=self.ax, label="Spray Intensity")

    def _create_sliders(self) -> None:
        """Create all parameter control sliders."""
        # Position sliders lower to increase gap from metrics - moved further down
        slider_configs = [
            ('nozzles', 'Number of Nozzles', 0.27),  # Moved down from 0.30 to increase gap from metrics
            ('distance', 'Height Above Surface', 0.23),
            ('angle', 'Spray Cone Angle (°)', 0.19),
            ('spacing', 'Nozzle Spacing', 0.15),
            ('density', 'Spray Density', 0.11)  # Moved down to reduce gap to buttons
        ]

        for param_key, label, y_pos in slider_configs:
            ax_slider = plt.axes([0.25, y_pos, 0.65, 0.03], facecolor=UI_BACKGROUND_COLOR)
            config = SLIDER_RANGES[param_key]

            if 'step' in config:
                slider = Slider(
                    ax_slider, label, config['min'], config['max'],
                    valinit=config['default'], valstep=config['step']
                )
            else:
                slider = Slider(
                    ax_slider, label, config['min'], config['max'],
                    valinit=config['default']
                )

            self.sliders[param_key] = slider

    def _create_reset_button(self) -> None:
        """Create the reset button."""
        reset_ax = plt.axes([0.65, 0.02, 0.1, 0.04])  # Reduced top margin from 0.05 to 0.02
        self.reset_button = Button(
            reset_ax, 'Reset',
            color=UI_BACKGROUND_COLOR,
            hovercolor='0.975'
        )

    def _create_optimize_button(self) -> None:
        """Create the optimize button."""
        optimize_ax = plt.axes([0.8, 0.02, 0.15, 0.04])  # Reduced top margin from 0.05 to 0.02
        self.optimize_button = Button(
            optimize_ax, 'Optimize',
            color='lightgreen',
            hovercolor='0.8'
        )

    def _create_pattern_button(self) -> None:
        """Create the spray pattern button with current pattern name."""
        pattern_ax = plt.axes([0.25, 0.02, 0.1, 0.04])  # Reduced top margin from 0.05 to 0.02

        # Get the display name for the current pattern
        from config import SPRAY_PATTERNS
        current_display_name = SPRAY_PATTERNS[self.current_pattern].split(' (')[0]

        self.pattern_button = Button(
            pattern_ax, current_display_name,  # Show current pattern name
            color=UI_BACKGROUND_COLOR,
            hovercolor='0.975'
        )

    def _create_progress_bar(self) -> None:
        """Create a progress bar for optimization."""
        # Progress bar positioned closer to sliders - reduced gap
        self.progress_bg_ax = plt.axes([0.25, 0.06, 0.5, 0.02], facecolor='lightgray')  # Moved up from 0.07 to 0.06
        self.progress_bg_ax.set_xlim(0, 1)
        self.progress_bg_ax.set_ylim(0, 1)
        self.progress_bg_ax.set_xticks([])
        self.progress_bg_ax.set_yticks([])

        # Progress bar fill
        self.progress_fill = Rectangle((0, 0), 0, 1, color='green', alpha=0.7)
        self.progress_bg_ax.add_patch(self.progress_fill)

        # Progress text
        self.progress_text = self.progress_bg_ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=8)

        # Hide progress bar initially
        self.progress_bg_ax.set_visible(False)

    def _create_metrics_display(self) -> None:
        """Create the coverage metrics display area closer to graph but with larger gap to sliders."""
        # Position metrics display closer to graph and further from sliders
        # Moved up from 0.33 to 0.35 to be closer to graph, creating larger gap to sliders at 0.27
        self.metrics_display_ax = plt.axes([0.15, 0.35, 0.7, 0.04], facecolor='white')
        self.metrics_display_ax.set_xticks([])
        self.metrics_display_ax.set_yticks([])
        self.metrics_display_ax.set_xlim(0, 1)
        self.metrics_display_ax.set_ylim(0, 1)

        # Add border
        for spine in self.metrics_display_ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    def _update_visualization(self) -> None:
        """Update the spray coverage visualization."""
        coverage = self.simulator.simulate(**self._get_current_parameters())

        if self.image is None:
            self.image = self.ax.imshow(
                coverage,
                extent=SURFACE_EXTENT,
                origin='lower',
                cmap=COLORMAP
            )
            self.ax.set_xlabel('X Position (units)')
            self.ax.set_ylabel('Y Position (units)')
            self.ax.set_title('Spray Coverage Pattern')
        else:
            self.image.set_data(coverage)
            self.image.set_clim(vmin=0, vmax=coverage.max())

        self.fig.canvas.draw_idle()

        # Update metrics display
        self._update_metrics_display(coverage)

    def _update_metrics_display(self, coverage: np.ndarray) -> None:
        """Update the metrics display with only uniformity and coverage percentage."""
        if self.metrics_display_ax is not None:
            # Clear previous text
            self.metrics_display_ax.clear()
            self.metrics_display_ax.set_xticks([])
            self.metrics_display_ax.set_yticks([])
            self.metrics_display_ax.set_xlim(0, 1)
            self.metrics_display_ax.set_ylim(0, 1)

            # Calculate only the requested metrics
            covered_area = np.count_nonzero(coverage)
            total_area = coverage.size
            coverage_percentage = (covered_area / total_area) * 100

            # Calculate uniformity metric (fixed logic)
            if covered_area > 0:
                covered_values = coverage[coverage > 0]
                mean_intensity = np.mean(covered_values)
                if mean_intensity > 0:
                    std_intensity = np.std(covered_values)
                    # Coefficient of variation (CV)
                    cv = std_intensity / mean_intensity
                    # Convert to uniformity percentage: lower CV = higher uniformity
                    # Use exponential decay to convert CV to 0-100% scale
                    uniformity = 100 * np.exp(-cv * 2)  # Multiply by 2 for better scaling
                    uniformity = min(100, max(0, uniformity))  # Clamp to 0-100%
                else:
                    uniformity = 100  # Perfect uniformity if all values are the same
            else:
                uniformity = 0  # No coverage = no uniformity

            # Display metrics in a larger, more prominent format
            self.metrics_display_ax.text(0.25, 0.5, f'Coverage: {coverage_percentage:.1f}%',
                                       fontsize=12, ha='center', va='center', weight='bold')
            self.metrics_display_ax.text(0.75, 0.5, f'Uniformity: {uniformity:.1f}%',
                                       fontsize=12, ha='center', va='center', weight='bold')

            # Add background styling
            self.metrics_display_ax.set_facecolor('lightblue')
            for spine in self.metrics_display_ax.spines.values():
                spine.set_edgecolor('darkblue')
                spine.set_linewidth(2)

    def _update_progress(self, progress: float) -> None:
        """Update the progress bar."""
        if hasattr(self, 'progress_fill') and hasattr(self, 'progress_text'):
            # Update progress fill width
            self.progress_fill.set_width(progress)

            # Update progress text
            self.progress_text.set_text(f'{progress*100:.1f}%')

            # Redraw the figure
            self.fig.canvas.draw_idle()

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for sliders and buttons."""
        # Connect a single mouse release event for all sliders for better responsiveness
        self.fig.canvas.mpl_connect('button_release_event', self._on_slider_release)

        # Connect reset button with immediate update
        self.reset_button.on_clicked(self._on_reset)
        # Connect optimize button
        self.optimize_button.on_clicked(self._on_optimize)
        # Connect pattern button
        self.pattern_button.on_clicked(self._on_pattern_change)

    def _on_slider_release(self, event) -> None:
        """Handle slider release events - only update when slider is dropped."""
        # Check if the release event happened over any slider
        for slider in self.sliders.values():
            if slider.ax.contains(event)[0]:
                self._update_visualization()
                break

    def _on_reset(self, event) -> None:
        """Handle reset button click."""
        for slider in self.sliders.values():
            slider.reset()
        # Immediately update visualization after reset
        self._update_visualization()

    def _on_optimize(self, event) -> None:
        """Handle optimize button click with progress tracking - fixed to run in main thread."""
        print("Starting optimization... This may take a moment.")

        # Disable the optimize button during optimization
        self.optimize_button.label.set_text('Optimizing...')
        self.optimize_button.color = 'orange'

        # Show and reset progress bar
        self.progress_bg_ax.set_visible(True)
        self.progress_fill.set_width(0)
        self.progress_text.set_text('0.0%')
        self.fig.canvas.draw_idle()

        def progress_callback(progress: float):
            """Callback function to update progress bar from optimization."""
            self._update_progress(progress)

        try:
            # Get current parameters from sliders
            current_params = self._get_current_parameters()

            # Perform optimization routine with progress callback
            # Run in main thread to avoid matplotlib threading issues
            optimized_params = self.simulator.optimize_for_uniformity(
                current_params,
                progress_callback=progress_callback
            )

            # Update sliders to match optimized parameters
            if 'cone_angle' in optimized_params:
                self.sliders['angle'].set_val(optimized_params['cone_angle'])
            if 'nozzle_spacing' in optimized_params:
                self.sliders['spacing'].set_val(optimized_params['nozzle_spacing'])

            # Update visualization immediately
            self._update_visualization()

            print("Optimization completed! Sliders updated with optimal values.")

        except Exception as e:
            print(f"Optimization failed: {e}")

        finally:
            # Re-enable the optimize button
            self.optimize_button.label.set_text('Optimize')
            self.optimize_button.color = 'lightgreen'

            # Hide progress bar
            self.progress_bg_ax.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_pattern_change(self, event) -> None:
        """Handle spray pattern button click - cycle through all available patterns."""
        from config import SPRAY_PATTERNS

        # Get list of available patterns
        pattern_list = list(SPRAY_PATTERNS.keys())

        # Find current pattern index and cycle to next
        try:
            current_index = pattern_list.index(self.current_pattern)
            next_index = (current_index + 1) % len(pattern_list)
            self.current_pattern = pattern_list[next_index]
        except ValueError:
            # If current pattern not found, default to full cone
            self.current_pattern = 'full_cone'

        # Update button label to show current pattern
        pattern_name = SPRAY_PATTERNS[self.current_pattern].split(' (')[0]  # Get short name
        self.pattern_button.label.set_text(f'{pattern_name}')

        # Update visualization with new pattern
        self._update_visualization()

        print(f"Spray pattern changed to: {SPRAY_PATTERNS[self.current_pattern]}")

    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values from sliders."""
        if not self.sliders:
            # Use defaults from config when sliders don't exist yet
            from config import (DEFAULT_N_NOZZLES, DEFAULT_DISTANCE, DEFAULT_CONE_ANGLE,
                               DEFAULT_NOZZLE_SPACING, DEFAULT_SPRAY_DENSITY, DEFAULT_SPRAY_PATTERN)
            return {
                'n_nozzles': DEFAULT_N_NOZZLES,
                'distance': DEFAULT_DISTANCE,
                'cone_angle': DEFAULT_CONE_ANGLE,
                'nozzle_spacing': DEFAULT_NOZZLE_SPACING,
                'spray_density': DEFAULT_SPRAY_DENSITY,
                'spray_pattern': DEFAULT_SPRAY_PATTERN  # Use the config default (now full_cone)
            }

        return {
            'n_nozzles': int(self.sliders['nozzles'].val),
            'distance': self.sliders['distance'].val,
            'cone_angle': self.sliders['angle'].val,
            'nozzle_spacing': self.sliders['spacing'].val,
            'spray_density': self.sliders['density'].val,
            'spray_pattern': self.current_pattern  # Use current selected pattern
        }

    def show(self) -> None:
        """Display the UI."""
        plt.show()


def create_application() -> SprayVisualizationUI:
    """
    Factory function to create a complete spray simulation application.

    Returns:
        Configured SprayVisualizationUI instance
    """
    simulator = SpraySimulator()
    ui = SprayVisualizationUI(simulator)
    ui.setup_ui()
    return ui
