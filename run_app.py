"""
MAIN ENTRY POINT - VAR GOLF STUDIO (AI GOLF SWING ANALYZER)
Launches borderless fullscreen Modern White Bento UI via PySide6 (no console window).
"""

import argparse
import os
import sys

# Ensure ASCII working directory for MediaPipe C++ runfiles loader
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from PySide6.QtWidgets import QApplication
import config
from src.ui.main_window import GolfAssistantWindow


def main():
    parser = argparse.ArgumentParser(description="VAR Golf Studio - AI Golf Swing Analyzer")
    parser.add_argument(
        "--camera",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="USB Camera / Webcam Index (default: 0)",
    )
    args, _unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setApplicationName("VAR Golf Studio")

    window = GolfAssistantWindow(camera_index=args.camera)
    window.showFullScreen()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
