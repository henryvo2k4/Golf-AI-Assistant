"""
CẤU HÌNH THÔNG SỐ TOÀN DỰ ÁN TRỢ LÝ AI GOLF (VAR GOLF)
Tích hợp chuẩn sinh trắc học PGA / TPI (3D World Landmarks + Face-On / Down-the-Line)
"""

import os

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

COUNTDOWN_SECONDS = 5
RECORD_SECONDS = 10
SLIDE_HOLD_SECONDS = 4.0
MIRROR_CAMERA_DEFAULT = True

# ==============================
# NGƯỠNG SINH TRẮC HỌC CHUẨN PGA / TPI (3D + 2D)
# ==============================
GOLFER_HANDEDNESS = "right"

# 1. Góc duỗi khuỷu tay dẫn hướng (3D Elbow Angle tại Đỉnh Swing -> Chạm bóng)
# Sử dụng pose_world_landmarks (3D mét) để không bị sai lệch do góc nghiêng camera
LEAD_ARM_MIN_ANGLE_3D = 150.0        # Chuẩn PGA >= 150° (Tối ưu 165° - 180°)

# 2. Độ ổn định trục đầu (Head Stability: Setup -> Impact)
HEAD_VERTICAL_DRIFT_MAX_RATIO = 0.07 # Dịch chuyển lên/xuống tối đa <= 7% chiều cao cơ thể
HEAD_LATERAL_DRIFT_MAX_RATIO = 0.08  # Dịch chuyển ngang đầu tối đa <= 8% chiều cao cơ thể

# 3. Trượt / Lắc hông ngang (Pelvic Sway - Góc quay Chính diện Face-On)
HIP_SWAY_MAX_RATIO = 0.14            # Hông sau không trượt quá trục chân sau > 14% bề rộng thế đứng

# 4. Ngược trục cột sống (Reverse Spine Angle - Góc quay Chính diện Face-On)
REVERSE_SPINE_MAX_DEG = 4.0          # Thân trên ở đỉnh swing không đổ ngược về mục tiêu > 4°

# 5. Xoay vai ở Đỉnh Swing (3D Shoulder Turn - Mặt phẳng ngang X-Z)
SHOULDER_TURN_MIN_DEG = 65.0         # Chuẩn golfer xoay vai đạt 70° - 105° so với lúc Setup

# 6. Ổn định đầu gối chân trụ (3D Knee Flexion Angle)
KNEE_MIN_ANGLE_3D = 128.0            # Không chùng sụp gối dưới 128°
KNEE_MAX_ANGLE_3D = 176.0            # Không duỗi thẳng cứng sớm (Early Extension) trước/tại Impact

# 7. Giữ góc gập lưng Setup (Spine Flexion - Dành cho góc quay Nhìn ngang Down-the-Line)
DTL_SPINE_FLEXION_MIN_DEG = 20.0     # Cúi người tối thiểu >= 20° (tránh đứng thẳng đơ)
DTL_POSTURE_LOSS_MAX_DEG = 12.0      # Không nhổm lưng mất góc quá 12° khi chạm bóng (Early Extension)

# ==============================
# FONT CHỮ TIẾNG VIỆT (WINDOWS)
# ==============================
FONT_BOLD_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REGULAR_PATH = r"C:\Windows\Fonts\segoeui.ttf"
FONT_MONO_PATH = r"C:\Windows\Fonts\consolab.ttf"
FALLBACK_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
