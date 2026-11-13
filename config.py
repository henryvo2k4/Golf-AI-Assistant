"""
CẤU HÌNH THÔNG SỐ TOÀN DỰ ÁN TRỢ LÝ AI GOLF (VAR GOLF)
Giao diện Desktop trực quan + Phân tích toàn diện 5 lỗi sinh trắc học Golf
"""

import os

# Đường dẫn gốc dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "pose_landmarker_full.task")
RECORDINGS_DIR = os.path.join(BASE_DIR, "assets", "videos")
LAST_SWING_VIDEO_PATH = os.path.join(RECORDINGS_DIR, "last_swing.mp4")

# ==============================
# CẤU HÌNH CAMERA & GHI HÌNH
# ==============================
DEFAULT_CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
TARGET_FPS = 30

# Thời gian đếm ngược & ghi hình theo yêu cầu thực chiến
COUNTDOWN_SECONDS = 5       # Đếm ngược 5 giây để golfer bước lên thảm vào vị trí
RECORD_SECONDS = 10         # Ghi hình tối đa 10 giây cho mỗi lượt đánh
MIRROR_CAMERA_DEFAULT = True

# ==============================
# CẤU HÌNH PHÂN TÍCH SINH TRẮC HỌC GOLF (CHUẨN PGA)
# ==============================
GOLFER_HANDEDNESS = "right"          # Mặc định thuận tay phải

# 1. Góc khuỷu tay dẫn hướng (Tay trái cho người thuận tay phải)
LEAD_ARM_MIN_ANGLE = 150.0           # Chuẩn PGA >= 150° tại Đỉnh Swing & Chạm bóng

# 2. Nhấp nhô / Ngẩng đầu sớm (Head Lift / Dip)
HEAD_DRIFT_MAX_RATIO = 0.07          # Dịch chuyển dọc đầu tối đa <= 7% chiều cao cơ thể

# 3. Trượt / Lắc hông ngang (Pelvic Sway)
HIP_SWAY_MAX_RATIO = 0.14            # Trượt hông ngang qua trục chân sau tối đa <= 14% độ rộng thế đứng

# 4. Ngược trục cột sống (Reverse Spine Angle)
REVERSE_SPINE_MAX_DEG = 4.0          # Độ nghiêng cột sống ngược về phía mục tiêu tối đa <= 4° tại đỉnh swing

# 5. Độ ổn định đầu gối chân trụ (Knee Stability)
KNEE_MIN_ANGLE = 128.0               # Không chùng sụp gối dưới 128°
KNEE_MAX_ANGLE = 178.0               # Không duỗi thẳng cứng sớm (Early Extension) ở đỉnh swing/impact

# ==============================
# CẤU HÌNH FONT CHỮ TIẾNG VIỆT (WINDOWS)
# ==============================
FONT_BOLD_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REGULAR_PATH = r"C:\Windows\Fonts\segoeui.ttf"
FALLBACK_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
