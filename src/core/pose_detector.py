"""
MODULE NHẬN DIỆN KHUNG XƯƠNG NGƯỜI (GOOGLE MEDIAPIPE POSE)
Trích xuất 33 điểm mốc (Keypoints) trong thời gian thực tốc độ 60+ FPS.
"""

import cv2
import mediapipe as mp


class GolfPoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        """
        Khởi tạo mô hình Google MediaPipe Pose.
        - model_complexity=1: Cân bằng hoàn hảo giữa tốc độ (60 FPS trên CPU i5) và độ chính xác.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            smooth_landmarks=True
        )

        # Chỉ số các điểm khớp MediaPipe quan trọng
        self.LM_LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        self.LM_LEFT_ELBOW = self.mp_pose.PoseLandmark.LEFT_ELBOW
        self.LM_LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST

        self.LM_RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        self.LM_RIGHT_ELBOW = self.mp_pose.PoseLandmark.RIGHT_ELBOW
        self.LM_RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST

    def process_frame(self, frame_bgr):
        """
        Xử lý 1 khung hình BGR từ OpenCV và trả về kết quả landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Khóa writeable=False để tăng tốc độ xử lý của MediaPipe
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def get_lead_arm_points(self, landmarks, frame_shape, handedness="right"):
        """
        Trích xuất tọa độ pixel [x, y, z] của 3 điểm khớp cánh tay dẫn hướng (Lead Arm).
        - Golfer tay phải (Right-handed): Tay dẫn hướng là TAY TRÁI (Vai 11, Cùi chỏ 13, Cổ tay 15).
        - Golfer tay trái (Left-handed): Tay dẫn hướng là TAY PHẢI (Vai 12, Cùi chỏ 14, Cổ tay 16).
        """
        if not landmarks:
            return None, None, None

        h, w, _ = frame_shape
        lm = landmarks.landmark

        if handedness == "right":
            s_lm = lm[self.LM_LEFT_SHOULDER.value]
            e_lm = lm[self.LM_LEFT_ELBOW.value]
            w_lm = lm[self.LM_LEFT_WRIST.value]
        else:
            s_lm = lm[self.LM_RIGHT_SHOULDER.value]
            e_lm = lm[self.LM_RIGHT_ELBOW.value]
            w_lm = lm[self.LM_RIGHT_WRIST.value]

        # Kiểm tra độ tin cậy hiển thị (visibility)
        if s_lm.visibility < 0.4 or e_lm.visibility < 0.4 or w_lm.visibility < 0.4:
            return None, None, None

        shoulder_px = (int(s_lm.x * w), int(s_lm.y * h), s_lm.z)
        elbow_px = (int(e_lm.x * w), int(e_lm.y * h), e_lm.z)
        wrist_px = (int(w_lm.x * w), int(w_lm.y * h), w_lm.z)

        return shoulder_px, elbow_px, wrist_px

    def draw_full_body_skeleton(self, frame_bgr, landmarks):
        """
        Vẽ toàn bộ 33 điểm mốc cơ thể người mờ phía sau.
        """
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
