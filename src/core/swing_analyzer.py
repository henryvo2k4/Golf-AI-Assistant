"""
MODULE BỘ NÃO PHÂN TÍCH CÚ SWING (FINITE STATE MACHINE & BIOMECHANICS)
Tự động tìm kiếm thời điểm Đỉnh Swing (Top of Swing) và đánh giá độ uốn cánh tay.
"""

from collections import deque
from src.utils.geometry import calculate_angle_2d


class SwingAnalyzer:
    # Các trạng thái của máy trạng thái (FSM)
    STATE_IDLE = "IDLE"              # Đang chờ dậm chân (Space)
    STATE_RECORDING = "RECORDING"    # Đang ghi nhận & theo dõi chuyển động swing
    STATE_ANALYZED = "ANALYZED"      # Đã phân tích xong và hiển thị kết quả

    def __init__(self, min_arm_angle=150.0, handedness="right", max_buffer_frames=300):
        self.min_arm_angle = min_arm_angle
        self.handedness = handedness
        self.max_buffer_frames = max_buffer_frames

        self.state = self.STATE_IDLE
        self.current_arm_angle = 180.0
        self.peak_arm_angle = 0.0
        self.is_error = False

        # Bộ đệm lưu vết chuyển động của cổ tay và góc
        self.wrist_y_history = deque(maxlen=60)   # Lưu 60 frame gần nhất để tìm cực tiểu Y
        self.angle_history = deque(maxlen=60)
        self.min_wrist_y = float("inf")
        self.peak_frame_idx = -1
        self.frame_counter = 0

        # Đếm frame sau đỉnh swing để hoàn tất cú đánh
        self.frames_after_peak = 0
        self.post_peak_threshold = 45             # Ghi tiếp 45 frame (~0.75s ở 60FPS) sau đỉnh

    def trigger_swing_start(self):
        """
        Kích hoạt khi người chơi dậm chân vào bàn đạp (hoặc bấm Space).
        """
        self.state = self.STATE_RECORDING
        self.current_arm_angle = 180.0
        self.peak_arm_angle = 0.0
        self.is_error = False
        self.wrist_y_history.clear()
        self.angle_history.clear()
        self.min_wrist_y = float("inf")
        self.peak_frame_idx = -1
        self.frame_counter = 0
        self.frames_after_peak = 0

    def reset(self):
        """
        Reset về trạng thái chờ IDLE để thực hiện cú đánh mới.
        """
        self.state = self.STATE_IDLE
        self.current_arm_angle = 180.0
        self.peak_arm_angle = 0.0
        self.is_error = False

    def update(self, shoulder_pt, elbow_pt, wrist_pt) -> dict:
        """
        Cập nhật phân tích từng khung hình video.
        """
        result = {
            "state": self.state,
            "current_angle": self.current_arm_angle,
            "peak_angle": self.peak_arm_angle,
            "is_error": self.is_error,
            "just_finished": False
        }

        if shoulder_pt is None or elbow_pt is None or wrist_pt is None:
            return result

        # 1. Luôn tính góc tay hiện tại theo thời gian thực
        angle = calculate_angle_2d(shoulder_pt, elbow_pt, wrist_pt)
        self.current_arm_angle = angle
        result["current_angle"] = angle

        # 2. Xử lý logic khi đang ở trạng thái RECORDING
        if self.state == self.STATE_RECORDING:
            self.frame_counter += 1
            wrist_y = wrist_pt[1]  # Tọa độ Y trong OpenCV (nhỏ hơn = cao hơn)

            self.wrist_y_history.append(wrist_y)
            self.angle_history.append(angle)

            # Tìm điểm cổ tay lên cao nhất (tọa độ Y nhỏ nhất)
            if wrist_y < self.min_wrist_y:
                self.min_wrist_y = wrist_y
                self.peak_arm_angle = angle
                self.peak_frame_idx = self.frame_counter
                self.frames_after_peak = 0
            else:
                # Cổ tay đang bắt đầu hạ xuống (Downswing)
                self.frames_after_peak += 1

            # Khi cổ tay đã qua đỉnh và đi xuống đủ ngưỡng -> Hoàn tất phân tích cú swing
            if self.frames_after_peak >= self.post_peak_threshold and self.peak_arm_angle > 0:
                self.state = self.STATE_ANALYZED
                # Kiểm tra lỗi theo luật chuyên gia PGA
                self.is_error = bool(self.peak_arm_angle < self.min_arm_angle)
                result["state"] = self.state
                result["peak_angle"] = self.peak_arm_angle
                result["is_error"] = self.is_error
                result["just_finished"] = True

        elif self.state == self.STATE_ANALYZED:
            result["peak_angle"] = self.peak_arm_angle
            result["is_error"] = self.is_error

        return result
