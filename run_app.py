"""
CHƯƠNG TRÌNH CHÍNH (MAIN ENTRY POINT)
HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)
Khởi chạy giao diện Desktop trực quan PySide6 (Không cửa sổ CMD đen).
"""

import argparse
import os
import sys

# Thiết lập thư mục làm việc chuẩn về thư mục chứa code (đường dẫn ASCII an toàn cho MediaPipe C++)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from PySide6.QtWidgets import QApplication
import config
from src.ui.main_window import GolfAssistantWindow


def main():
    parser = argparse.ArgumentParser(description="He thong Tro ly AI Golf (VAR Golf)")
    parser.add_argument(
        "--camera",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Chi so Camera USB / Webcam (mac dinh 0)",
    )
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Trợ Lý AI Golf - VAR Golf")

    window = GolfAssistantWindow(camera_index=args.camera)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
