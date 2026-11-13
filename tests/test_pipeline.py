"""
KIỂM THỬ TÍCH HỢP TOÀN HỆ THỐNG (AI 2D+3D + ANALYZER + RENDERER + OLED UI)
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
from src.core.swing_analyzer import BatchSwingAnalyzer, FaultCriterion
from src.utils.drawing import draw_skeleton_overlay, render_annotated_snapshot


def _make_pose_pair(
    nose=(320, 100),
    l_sh=(350, 160), r_sh=(290, 160),
    l_el=(360, 210), r_el=(280, 210),
    l_wr=(350, 260), r_wr=(290, 260),
    l_hip=(340, 290), r_hip=(300, 290),
    l_knee=(345, 380), r_knee=(295, 380),
    l_ank=(350, 460), r_ank=(290, 460),
    z_offsets=None,
):
    pts = {
        "nose": nose,
        "left_ear": (nose[0] + 15, nose[1]),
        "right_ear": (nose[0] - 15, nose[1]),
        "left_shoulder": l_sh,
        "right_shoulder": r_sh,
        "left_elbow": l_el,
        "right_elbow": r_el,
        "left_wrist": l_wr,
        "right_wrist": r_wr,
        "left_hip": l_hip,
        "right_hip": r_hip,
        "left_knee": l_knee,
        "right_knee": r_knee,
        "left_ankle": l_ank,
        "right_ankle": r_ank,
    }
    z_offsets = z_offsets or {}
    lm_2d = {k: (float(v[0]), float(v[1]), 0.0, 0.99) for k, v in pts.items()}
    # Hệ 3D mét quy đổi tuyến tính quanh tâm hông (320, 290)
    lm_3d = {
        k: ((float(v[0]) - 320.0) * 0.004, (float(v[1]) - 290.0) * 0.004, float(z_offsets.get(k, 0.0)), 0.99)
        for k, v in pts.items()
    }
    return lm_2d, lm_3d


def test_mediapipe_pose_landmarker_init_and_detect():
    detector = GolfPoseDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect_single_image(blank)
    assert res is None  # Không có người trong ảnh đen


def test_batch_analyzer_detects_faults_and_renders_snapshot():
    # Chuỗi 25 khung hình giả lập Face-On swing có nhiều lỗi kỹ thuật:
    # - Address (0..9)
    # - Top of Swing (10..15): Gập khuỷu tay trái 90°, nhổm đầu 50px, trượt hông phải x=250,
    #   ngược trục cột sống (Reverse Spine), xoay vai ít (~10°)
    # - Impact & Finish (16..24): Hạ tay xuống
    fake_2d, fake_3d = [], []
    for i in range(25):
        if i < 10:
            p2, p3 = _make_pose_pair()
        elif i <= 15:
            p2, p3 = _make_pose_pair(
                nose=(320, 50),
                l_sh=(390, 150),
                r_sh=(350, 150),
                l_el=(350, 110),
                l_wr=(310, 110),
                r_wr=(315, 115),
                l_hip=(330, 290),
                r_hip=(250, 290),
            )
        else:
            p2, p3 = _make_pose_pair(l_wr=(350, 270), r_wr=(340, 270))
        fake_2d.append(p2)
        fake_3d.append(p3)

    analyzer = BatchSwingAnalyzer(handedness="right")
    analyzer.detector.detect_video_sequence = (
        lambda frames, ts, progress_callback=None: (fake_2d, fake_3d)
    )

    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(25)]
    dummy_ts = [i * 33.3 for i in range(25)]

    result = analyzer.analyze_clip(dummy_frames, dummy_ts)
    assert result.person_detected is True
    assert result.camera_view == "face_on"
    assert result.top_idx in range(10, 16)
    assert result.finish_idx >= result.impact_idx

    error_codes = {c.code for c in result.criteria if c.status == "error"}
    assert "bent_lead_arm" in error_codes
    assert "head_movement" in error_codes
    assert "pelvic_sway" in error_codes
    assert "reverse_spine" in error_codes
    assert "shoulder_turn" in error_codes

    # Render snapshot & slow-motion overlay
    first_err = next(c for c in result.criteria if c.status == "error")
    snap = render_annotated_snapshot(
        dummy_frames[first_err.frame_idx],
        result.landmarks_per_frame[first_err.frame_idx],
        first_err,
        phase_label="ĐỈNH SWING | Frame 12/25",
    )
    assert snap.shape == (480, 640, 3)

    slowmo_frame = draw_skeleton_overlay(
        dummy_frames[result.top_idx],
        result.landmarks_per_frame[result.top_idx],
        arm_angle=result.arm_angles_per_frame[result.top_idx],
        handedness="right",
    )
    assert slowmo_frame.shape == (480, 640, 3)


def test_ui_theme_icons_and_mirror_criterion():
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import GolfAssistantWindow, mirror_criterion
    from src.ui.theme import build_app_stylesheet
    from src.ui.icons import build_svg_bytes

    app = QApplication.instance() or QApplication([])
    assert app is not None
    assert "QFrame#TopBar" in build_app_stylesheet()
    assert b"<svg" in build_svg_bytes("target", color="#10B981")

    crit = FaultCriterion(
        code="head_movement",
        title="Test",
        status="error",
        measured_text="10%",
        standard_text="<= 8%",
        frame_idx=5,
        description="desc",
        correction_tip="tip",
        annotation_type="head_drift",
        annotation_data={"nose": (100.0, 80.0), "addr_nose": (120.0, 90.0), "addr_nose_y": 90.0},
    )
    mirrored = mirror_criterion(crit, width=640)
    assert mirrored.annotation_data["nose"] == (540.0, 80.0)
    assert mirrored.annotation_data["addr_nose"] == (520.0, 90.0)
    assert GolfAssistantWindow is not None
