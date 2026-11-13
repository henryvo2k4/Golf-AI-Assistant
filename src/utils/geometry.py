"""
MODULE TOÁN HỌC HÌNH HỌC VECTOR 2D/3D & XỬ LÝ TÍN HIỆU (VECTOR GEOMETRY)
Tính toán góc uốn khớp 3D (bất biến theo góc camera), xoay vai X-Z,
nghiêng trục cột sống và bộ lọc làm mượt chuỗi tọa độ theo thời gian.
"""

import math
import numpy as np


def calculate_angle_2d(point_a, point_b, point_c) -> float:
    """Tính góc (Độ) tại đỉnh B tạo bởi A-B-C trong mặt phẳng 2D (x, y)."""
    a = np.array(point_a[:2], dtype=np.float64)
    b = np.array(point_b[:2], dtype=np.float64)
    c = np.array(point_c[:2], dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_angle_3d(point_a, point_b, point_c) -> float:
    """
    Tính góc uốn (Độ) tại đỉnh B tạo bởi A-B-C trong không gian 3D thực (X, Y, Z).
    Giúp đo góc khuỷu tay và đầu gối chính xác tuyệt đối ngay cả khi vai xoay ngang.
    """
    a = np.array(point_a[:3], dtype=np.float64)
    b = np.array(point_b[:3], dtype=np.float64)
    c = np.array(point_c[:3], dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def midpoint_2d(p1, p2) -> tuple[float, float]:
    """Tính trung điểm 2D."""
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def midpoint_3d(p1, p2) -> tuple[float, float, float]:
    """Tính trung điểm 3D."""
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5)


def distance_2d(p1, p2) -> float:
    """Tính khoảng cách Euclidean 2D."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def calculate_spine_away_tilt_deg(mid_hip, mid_shoulder, target_dir_sign: float) -> float:
    """
    Tính góc nghiêng ngang của trục cột sống (Tâm hông -> Tâm vai) so với phương thẳng đứng.
    - Giá trị DƯƠNG (+): Nghiêng RA XA mục tiêu (tư thế chuẩn ở đỉnh swing).
    - Giá trị ÂM (-): Nghiêng NGƯỢC VỀ PHÍA MỤC TIÊU (Lỗi Reverse Spine Angle).
    """
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]  # OpenCV y hướng xuống

    if dy <= 1e-6:
        return 0.0

    angle_from_vertical = math.degrees(math.atan2(dx, dy))
    return -angle_from_vertical * target_dir_sign


def calculate_shoulder_turn_xz_deg(
    addr_left_sh, addr_right_sh, top_left_sh, top_right_sh
) -> float:
    """
    Tính góc xoay vai (độ) trên mặt phẳng nằm ngang (X, Z) trong không gian 3D mét
    giữa thời điểm chuẩn bị (Setup) và Đỉnh Swing (Top of Swing).
    """
    v_addr = np.array([addr_left_sh[0] - addr_right_sh[0], addr_left_sh[2] - addr_right_sh[2]], dtype=np.float64)
    v_top = np.array([top_left_sh[0] - top_right_sh[0], top_left_sh[2] - top_right_sh[2]], dtype=np.float64)

    n1 = np.linalg.norm(v_addr)
    n2 = np.linalg.norm(v_top)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cos_val = float(np.clip(np.dot(v_addr, v_top) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def calculate_spine_forward_flexion_deg(mid_hip_2d, mid_sh_2d) -> float:
    """
    Tính góc cúi lưng về phía trước so với trục thẳng đứng (dành cho góc quay Nhìn ngang Down-the-Line).
    """
    dx = abs(mid_sh_2d[0] - mid_hip_2d[0])
    dy = max(mid_hip_2d[1] - mid_sh_2d[1], 1e-6)
    return float(math.degrees(math.atan2(dx, dy)))


def smooth_landmarks_sequence(landmarks_list: list[dict | None]) -> list[dict | None]:
    """
    Bộ lọc Gaussian 5 khung hình [0.1, 0.2, 0.4, 0.2, 0.1] khử nhiễu rung khung xương.
    Giữ nguyên độ tin cậy (visibility) của khung hình tâm.
    """
    n = len(landmarks_list)
    if n < 5:
        return landmarks_list

    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    offsets = [-2, -1, 0, 1, 2]
    smoothed = []

    for i in range(n):
        center = landmarks_list[i]
        if center is None:
            smoothed.append(None)
            continue

        new_frame = {}
        for key, (cx, cy, cz, cvis) in center.items():
            sum_w = 0.0
            sx, sy, sz = 0.0, 0.0, 0.0
            for off, w in zip(offsets, weights):
                j = i + off
                if 0 <= j < n and landmarks_list[j] is not None and key in landmarks_list[j]:
                    jx, jy, jz, jvis = landmarks_list[j][key]
                    if jvis >= 0.35:
                        sx += jx * w
                        sy += jy * w
                        sz += jz * w
                        sum_w += w
            if sum_w > 0:
                new_frame[key] = (sx / sum_w, sy / sum_w, sz / sum_w, cvis)
            else:
                new_frame[key] = (cx, cy, cz, cvis)
        smoothed.append(new_frame)

    return smoothed
