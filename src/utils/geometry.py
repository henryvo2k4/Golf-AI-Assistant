"""
MODULE TOÁN HỌC HÌNH HỌC VECTOR (VECTOR GEOMETRY)
Tính toán góc uốn khớp xương, khoảng cách chuẩn hóa và độ nghiêng trục cột sống.
"""

import math
import numpy as np


def calculate_angle_2d(point_a, point_b, point_c) -> float:
    """
    Tính góc uốn (Độ) tại đỉnh B tạo bởi 3 điểm A, B, C trong mặt phẳng 2D (x, y).
    """
    a = np.array(point_a[:2], dtype=np.float64)
    b = np.array(point_b[:2], dtype=np.float64)
    c = np.array(point_c[:2], dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_angle_3d(point_a, point_b, point_c) -> float:
    """
    Tính góc uốn (Độ) tại đỉnh B tạo bởi 3 điểm A, B, C trong không gian 3D (x, y, z).
    """
    a = np.array(point_a[:3], dtype=np.float64)
    b = np.array(point_b[:3], dtype=np.float64)
    c = np.array(point_c[:3], dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def midpoint_2d(p1, p2) -> tuple[float, float]:
    """Tính trung điểm 2D giữa 2 tọa độ."""
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def distance_2d(p1, p2) -> float:
    """Tính khoảng cách Euclidean 2D."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def calculate_spine_away_tilt_deg(mid_hip, mid_shoulder, target_dir_sign: float) -> float:
    """
    Tính góc nghiêng ngang của trục cột sống (tính từ Tâm hông lên Tâm vai) so với phương thẳng đứng.
    - Trả về giá trị DƯƠNG (+) nếu thân trên nghiêng RA XA mục tiêu (tư thế chuẩn ở đỉnh swing).
    - Trả về giá trị ÂM (-) nếu thân trên nghiêng NGƯỢC VỀ PHÍA MỤC TIÊU (Lỗi Reverse Spine Angle).

    Tham số:
        mid_hip: (x, y) tâm hông
        mid_shoulder: (x, y) tâm vai
        target_dir_sign: +1.0 nếu hướng mục tiêu là chiều dương X (bên phải ảnh), -1.0 nếu bên trái ảnh.
    """
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]  # Dương khi vai ở phía trên hông (OpenCV y hướng xuống)

    if dy <= 1e-6:
        return 0.0

    # Góc nghiêng theo phương X dương (radian -> độ)
    angle_from_vertical = math.degrees(math.atan2(dx, dy))
    # Nếu nghiêng cùng chiều với hướng mục tiêu (dx * target_dir_sign > 0) -> đó là nghiêng VỀ mục tiêu (âm)
    # Nếu nghiêng ngược chiều hướng mục tiêu -> nghiêng RA XA mục tiêu (dương)
    return -angle_from_vertical * target_dir_sign
