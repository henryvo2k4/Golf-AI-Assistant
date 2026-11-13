"""
KIỂM THỬ TÍCH HỢP TOÀN HỆ THỐNG (AI + ANALYZER + RENDERER + PYSIDE6 UI)
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
os.chdir(PARENT_DIR)

from src.core.pose_detector import GolfPoseDetector
from src.core.swing_analyzer import BatchSwingAnalyzer
from src.utils.drawing import draw_skeleton_overlay, render_annotated_snapshot


def _make_pose(
    nose=(320, 100),
    l_sh=(350, 160), r_sh=(290, 160),
    l_el=(360, 210), r_el=(280, 210),
    l_wr=(350, 260), r_wr=(290, 260),
    l_hip=(340, 290), r_hip=(300, 290),
    l_knee=(345, 380), r_knee=(295, 380),
    l_ank=(350, 460), r_ank=(290, 460),
):
    pts = {
        "nose": nose, "left_ear": (nose[0] + 15, nose[1]), "right_ear": (nose[0] - 15, nose[1]),
        "left_shoulder": l_sh, "right_shoulder": r_sh,
        "left_elbow": l_el, "right_elbow": r_el,
        "left_wrist": l_wr, "right_wrist": r_wr,
        "left_hip": l_hip, "right_hip": r_hip,
        "left_knee": l_knee, "right_knee": r_knee,
        "left_ankle": l_ank, "right_ankle": r_ank,
    }
    return {k: (float(v[0]), float(v[1]), 0.0, 0.99) for k, v in pts.items()}


def test_mediapipe_pose_landmarker_init_and_detect():
    detector = GolfPoseDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect_single_image(blank)
    assert res is None  # Không có người trong ảnh đen


def test_batch_analyzer_detects_faults_and_renders_snapshot():
    # Tạo chuỗi 25 khung hình giả lập 1 cú swing có lỗi:
    # - Address (0..9)
    # - Top of Swing (10..15): Gập khuỷu tay trái 90 độ + Nhổm đầu lên + Trượt hông + Ngược trục cột sống
    # - Impact (16..24): Hạ tay xuống
    fake_landmarks = []
    for i in range(25):
        if i < 10:
            fake_landmarks.append(_make_pose())
        elif i <= 15:
            # Đỉnh swing: cổ tay cao (y=90), khuỷu tay gập 90 độ, đầu ngẩng cao (y=50), hông phải trượt sang x=250, vai đổ sang phải (x=370)
            fake_landmarks.append(_make_pose(
                nose=(320, 50),
                l_sh=(390, 150), r_sh=(350, 150),
                l_el=(350, 110),
                l_wr=(310, 110),
                l_hip=(330, 290), r_hip=(250, 290),
            ))
        else:
            fake_landmarks.append(_make_pose(l_wr=(350, 270)))

    analyzer = BatchSwingAnalyzer(handedness="right")
    # Thay detector.detect_video_sequence bằng mock trả về chuỗi khớp xương giả lập
    analyzer.detector.detect_video_sequence = lambda frames, ts, progress_callback=None: fake_landmarks

    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(25)]
    dummy_ts = [i * 33.3 for i in range(25)]

    result = analyzer.analyze_clip(dummy_frames, dummy_ts)
    assert result.person_detected is True
    assert result.top_idx in range(10, 16)

    error_codes = {c.code for c in result.criteria if c.status == "error"}
    assert "bent_lead_arm" in error_codes
    assert "head_movement" in error_codes
    assert "pelvic_sway" in error_codes
    assert "reverse_spine" in error_codes

    # Kiểm tra render ảnh chú thích tiếng Việt và overlay slow-motion
    first_err = next(c for c in result.criteria if c.status == "error")
    snap = render_annotated_snapshot(
        dummy_frames[first_err.frame_idx],
        result.landmarks_per_frame[first_err.frame_idx],
        first_err,
        phase_label="ĐỈNH SWING"
    )
    assert snap.shape == (480, 640, 3)

    slowmo_frame = draw_skeleton_overlay(
        dummy_frames[result.top_idx],
        result.landmarks_per_frame[result.top_idx],
        arm_angle=result.arm_angles_per_frame[result.top_idx],
        handedness="right"
    )
    assert slowmo_frame.shape == (480, 640, 3)


def test_pyside6_ui_import():
    from src.ui.main_window import GolfAssistantWindow
    assert GolfAssistantWindow is not None
