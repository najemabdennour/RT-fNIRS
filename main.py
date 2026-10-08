"""Application entry point: launches the PyQt5 LSL fNIRS visualizer window."""
import sys
from PyQt5.QtWidgets import QApplication
from ui.exception_handler import install_exception_hook
from ui.main_window import LSLVisualizerApp


def main():
    """Create the Qt application, install the global exception handler, show the main window, and run the event loop."""
    app = QApplication(sys.argv)
    install_exception_hook()
    window = LSLVisualizerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
