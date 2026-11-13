"""
MODULE NHẬN DIỆN KHUNG XƯƠNG NGƯỜI 2D + 3D METRIC (GOOGLE MEDIAPIPE TASKS POSELANDMARKER)
Trích xuất đồng thời:
- pose_landmarks (2D Pixel): Dùng để vẽ khung xương, vòng tròn khoanh vùng lỗi chính xác lên ảnh.
- pose_world_landmarks (3D Mét): Dùng để đo góc khớp thật (không bị méo phối cảnh khi xoay vai).
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


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
    """

    def __init__(self, model_path: str = None, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.model_path = model_path or config.MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Khong tim thay file model MediaPipe tai: {self.model_path}")

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
    def _extract_pair(detection_result, frame_shape) -> tuple[dict | None, dict | None]:
        """
        Trả về cặp (landmarks_2d_px, landmarks_3d_world).
        """
        if not detection_result.pose_landmarks:
            return None, None

        lm_2d_raw = detection_result.pose_landmarks[0]
        lm_3d_raw = detection_result.pose_world_landmarks[0] if detection_result.pose_world_landmarks else None

        h, w, _ = frame_shape
        dict_2d = {}
        dict_3d = {}

        for name, idx in LANDMARK_IDS.items():
            p2 = lm_2d_raw[idx]
            vis = float(getattr(p2, "visibility", 1.0) or 1.0)
            dict_2d[name] = (float(p2.x * w), float(p2.y * h), float(p2.z * w), vis)

            if lm_3d_raw is not None:
                p3 = lm_3d_raw[idx]
                vis3 = float(getattr(p3, "visibility", vis) or vis)
                dict_3d[name] = (float(p3.x), float(p3.y), float(p3.z), vis3)
            else:
                dict_3d[name] = (float(p2.x), float(p2.y), float(p2.z), vis)

        return dict_2d, dict_3d

    def detect_video_sequence(
        self, frames_bgr: list, timestamps_ms: list, progress_callback=None
    ) -> tuple[list[dict | None], list[dict | None]]:
        """
        Phân tích toàn bộ chuỗi khung hình video theo chế độ RunningMode.VIDEO.
        Trả về: (landmarks_2d_per_frame, landmarks_3d_per_frame)
        """
        total = len(frames_bgr)
        list_2d: list[dict | None] = []
        list_3d: list[dict | None] = []

        if total == 0:
            return list_2d, list_3d

        landmarker = self._create_landmarker(vision.RunningMode.VIDEO)
        try:
            last_ts = -1
            for idx, (frame_bgr, ts_ms) in enumerate(zip(frames_bgr, timestamps_ms)):
                safe_ts = int(ts_ms)
                if safe_ts <= last_ts:
                    safe_ts = last_ts + 1
                last_ts = safe_ts

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection = landmarker.detect_for_video(mp_image, safe_ts)
                d2, d3 = self._extract_pair(detection, frame_bgr.shape)
                list_2d.append(d2)
                list_3d.append(d3)

                if progress_callback and (idx % 5 == 0 or idx == total - 1):
                    progress_callback(int(((idx + 1) / total) * 100))
        finally:
            landmarker.close()

        return list_2d, list_3d

    def detect_single_image(self, frame_bgr) -> dict | None:
        """Nhận diện tư thế trên 1 khung hình đơn lẻ (RunningMode.IMAGE)."""
        landmarker = self._create_landmarker(vision.RunningMode.IMAGE)
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection = landmarker.detect(mp_image)
            d2, _ = self._extract_pair(detection, frame_bgr.shape)
            return d2
        finally:
            landmarker.close()
