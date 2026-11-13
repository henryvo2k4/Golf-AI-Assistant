"""
MODULE NHẬN DIỆN KHUNG XƯƠNG NGƯỜI (GOOGLE MEDIAPIPE TASKS POSELANDMARKER)
Tương thích chuẩn MediaPipe 1.2.2+ bằng API PoseLandmarker mới nhất.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


# Bảng ánh xạ các khớp xương trọng tâm trong Golf
LANDMARK_IDS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Các cặp khớp nối để vẽ khung xương cơ thể (Skeleton)
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class GolfPoseDetector:
    """
    Bộ nhận diện 33 khớp xương sử dụng Google MediaPipe PoseLandmarker.
    Đọc file model dưới dạng buffer byte để không bao giờ lỗi đường dẫn trên Windows.
    """

    def __init__(self, model_path: str = None, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.model_path = model_path or config.MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Khong tim thay file model MediaPipe tai: {self.model_path}"
            )

        with open(self.model_path, "rb") as f:
            self._model_bytes = f.read()

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def _create_landmarker(self, running_mode):
        base_options = python.BaseOptions(model_asset_buffer=self._model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _extract_landmarks_dict(pose_landmarks_list, frame_shape) -> dict | None:
        """
        Chuyển kết quả MediaPipe PoseLandmarker thành từ điển tọa độ pixel:
        {
            "left_shoulder": (x_px, y_px, z, visibility),
            ...
        }
        """
        if not pose_landmarks_list:
            return None

        lm_list = pose_landmarks_list[0]
        h, w, _ = frame_shape
        extracted = {}

        for name, idx in LANDMARK_IDS.items():
            lm = lm_list[idx]
            extracted[name] = (
                float(lm.x * w),
                float(lm.y * h),
                float(lm.z),
                float(getattr(lm, "visibility", 1.0) or 1.0),
            )
        return extracted

    def detect_video_sequence(self, frames_bgr: list, timestamps_ms: list, progress_callback=None) -> list:
        """
        Phân tích chuỗi khung hình video đã ghi (RunningMode.VIDEO cho độ mượt cao).
        Trả về danh sách các từ điển khớp xương tương ứng từng khung hình (hoặc None nếu không thấy người).
        """
        total = len(frames_bgr)
        results_per_frame = []

        if total == 0:
            return results_per_frame

        landmarker = self._create_landmarker(vision.RunningMode.VIDEO)
        try:
            last_ts = -1
            for idx, (frame_bgr, ts_ms) in enumerate(zip(frames_bgr, timestamps_ms)):
                # Đảm bảo timestamp luôn tăng nghiêm ngặt (yêu cầu của MediaPipe VIDEO mode)
                safe_ts = int(ts_ms)
                if safe_ts <= last_ts:
                    safe_ts = last_ts + 1
                last_ts = safe_ts

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                detection = landmarker.detect_for_video(mp_image, safe_ts)
                lm_dict = self._extract_landmarks_dict(detection.pose_landmarks, frame_bgr.shape)
                results_per_frame.append(lm_dict)

                if progress_callback and (idx % 5 == 0 or idx == total - 1):
                    progress_callback(int(((idx + 1) / total) * 100))
        finally:
            landmarker.close()

        return results_per_frame

    def detect_single_image(self, frame_bgr) -> dict | None:
        """Nhận diện tư thế trên 1 khung hình đơn lẻ (RunningMode.IMAGE)."""
        landmarker = self._create_landmarker(vision.RunningMode.IMAGE)
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection = landmarker.detect(mp_image)
            return self._extract_landmarks_dict(detection.pose_landmarks, frame_bgr.shape)
        finally:
            landmarker.close()
