"""
CẤU HÌNH THÔNG SỐ TOÀN DỰ ÁN TRỢ LÝ AI GOLF (VAR GOLF)
Hiển thị trực quan toàn bộ kết quả lên màn hình (Visual HUD & On-screen Alerts)
"""

# ==============================
# CẤU HÌNH CAMERA & VIDEO NGUỒN
# ==============================
DEFAULT_VIDEO_SOURCE = 0              # 0 là Webcam mặc định, hoặc đường dẫn file .mp4
TARGET_FPS = 60                       # Tần số quét mục tiêu của EMEET C60E
FRAME_WIDTH = 1280                    # Độ phân giải hiển thị chuẩn HD
FRAME_HEIGHT = 720

# ==============================
# CẤU HÌNH PHÂN TÍCH GOLF SWING
# ==============================
# Người chơi tay thuận (Right-handed): Tay dẫn hướng (Lead Arm) là TAY TRÁI
# Điểm xương MediaPipe:
# 11: Vai trái (Left Shoulder)
# 13: Cùi chỏ trái (Left Elbow)
# 15: Cổ tay trái (Left Wrist)
GOLFER_HANDEDNESS = "right"

# Ngưỡng góc uốn tay trái tối thiểu ở đỉnh swing (Chuẩn PGA >= 150 độ)
# Dưới 150 độ là bị lỗi co tay / gập cùi chỏ (Bent Lead Arm / Chicken Wing)
LEAD_ARM_MIN_ANGLE = 150.0

# ==============================
# CẤU HÌNH BÀN ĐẠP CHÂN (FOOT PEDAL)
# ==============================
# Bàn đạp chân USB vỏ kim loại TFS gửi tín hiệu mã phím Spacebar (ASCII 32)
TRIGGER_KEY_CODE = 32                 # Mã phím Space trong OpenCV (ord(' '))
RESET_KEY_CODE = ord('r')             # Phím R để reset trạng thái
QUIT_KEY_CODE = ord('q')              # Phím Q để thoát

# ==============================
# CẤU HÌNH BỘ NHỚ ĐỆM (BUFFER)
# ==============================
BUFFER_SECONDS = 5
MAX_BUFFER_FRAMES = TARGET_FPS * BUFFER_SECONDS

# ==============================
# MÀU SẮC GIAO DIỆN (ĐỊNH DẠNG BGR)
# ==============================
COLOR_GREEN = (0, 230, 64)            # Màu xanh lá đạt chuẩn PGA
COLOR_RED = (0, 0, 255)               # Màu đỏ cảnh báo lỗi
COLOR_YELLOW = (0, 215, 255)          # Màu vàng trạng thái chuẩn bị / chờ
COLOR_CYAN = (255, 255, 0)            # Màu xanh dương nhạt
COLOR_WHITE = (255, 255, 255)         # Màu trắng chữ
COLOR_DARK_BG = (20, 20, 20)          # Màu nền bảng điều khiển HUD
