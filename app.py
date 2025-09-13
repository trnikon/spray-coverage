"""
Spray Coverage Simulation Application

A spray nozzle coverage analysis tool with interactive visualization.
This application simulates spray patterns from multiple nozzles arranged in a grid
pattern and provides real-time parameter adjustment through an intuitive UI.

Date: 2025
"""

from ui import create_application


def main():
    """
    Main entry point for the spray coverage simulation application.

    This function initializes and runs the interactive spray simulation
    with a professional matplotlib-based GUI.
    """
    try:
        # Create and configure the application
        app = create_application()

        # Display the application
        print("Starting Spray Coverage Simulation...")
        print("Use the sliders to adjust parameters in real-time.")
        print("Click 'Reset' to return to default values.")

        app.show()

    except Exception as e:
        print(f"Error starting application: {e}")
        raise


if __name__ == "__main__":
    main()
