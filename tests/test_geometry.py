"""
KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CHO MODULE TOÁN HỌC HÌNH HỌC VECTOR
"""

import math
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from src.utils.geometry import (
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_shoulder_turn_xz_deg,
    calculate_spine_away_tilt_deg,
    calculate_spine_forward_flexion_deg,
    smooth_landmarks_sequence,
)


def test_straight_line_180_degrees():
    angle = calculate_angle_2d((0, 100), (0, 50), (0, 0))
    assert math.isclose(angle, 180.0, abs_tol=1e-3)


def test_right_angle_90_degrees():
    angle = calculate_angle_2d((0, 10), (0, 0), (10, 0))
    assert math.isclose(angle, 90.0, abs_tol=1e-3)


def test_pga_threshold_150_degrees():
    b = (0.0, 0.0)
    a = (1.0, 0.0)
    c = (math.cos(math.radians(150)), math.sin(math.radians(150)))
    angle = calculate_angle_2d(a, b, c)
    assert math.isclose(angle, 150.0, abs_tol=1e-3)


def test_zero_division_guard():
    assert calculate_angle_2d((10, 10), (10, 10), (20, 20)) == 0.0


def test_3d_angle():
    angle = calculate_angle_3d((1, 0, 0), (0, 0, 0), (0, 1, 0))
    assert math.isclose(angle, 90.0, abs_tol=1e-3)


def test_spine_away_tilt_positive_when_leaning_away():
    # Mục tiêu bên phải (+1.0). Vai nghiêng sang trái (-10px) -> Nghiêng ra xa mục tiêu -> Dương (+)
    tilt = calculate_spine_away_tilt_deg((100, 300), (90, 200), target_dir_sign=1.0)
    assert tilt > 0.0


def test_spine_away_tilt_negative_when_reverse_spine():
    # Mục tiêu bên phải (+1.0). Vai đổ sang phải (+15px) -> Ngược trục về phía mục tiêu -> Âm (-)
    tilt = calculate_spine_away_tilt_deg((100, 300), (115, 200), target_dir_sign=1.0)
    assert tilt < -5.0


def test_shoulder_turn_xz_90_degrees():
    # Setup: trục vai dọc theo X; Top: trục vai xoay 90 độ dọc theo Z
    turn = calculate_shoulder_turn_xz_deg(
        addr_left_sh=(0.2, 0.0, 0.0),
        addr_right_sh=(-0.2, 0.0, 0.0),
        top_left_sh=(0.0, 0.0, 0.2),
        top_right_sh=(0.0, 0.0, -0.2),
    )
    assert math.isclose(turn, 90.0, abs_tol=1e-3)


def test_spine_forward_flexion_45_deg():
    flex = calculate_spine_forward_flexion_deg((100, 300), (200, 200))
    assert math.isclose(flex, 45.0, abs_tol=1e-3)


def test_smooth_landmarks_sequence_reduces_spike():
    seq = [
        {"nose": (100.0, 100.0, 0.0, 0.99)},
        {"nose": (100.0, 100.0, 0.0, 0.99)},
        {"nose": (150.0, 100.0, 0.0, 0.99)},  # Nhiễu tăng vọt 50px ở frame giữa
        {"nose": (100.0, 100.0, 0.0, 0.99)},
        {"nose": (100.0, 100.0, 0.0, 0.99)},
    ]
    smoothed = smooth_landmarks_sequence(seq)
    # 0.4 * 150 + 0.6 * 100 = 120.0
    assert math.isclose(smoothed[2]["nose"][0], 120.0, abs_tol=1e-3)
