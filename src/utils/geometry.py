"""
MODULE TOÁN HỌC HÌNH HỌC VECTOR (VECTOR GEOMETRY)
Chịu trách nhiệm tính toán góc uốn khớp xương trong không gian 2D và 3D.
Dành cho: Phân tích góc khuỷu tay, góc vai, góc hông trong cú Swing Golf.
"""

import numpy as np


def calculate_angle_2d(point_a, point_b, point_c) -> float:
    """
    Tính góc uốn (đơn vị: Độ) tại đỉnh B tạo bởi 3 điểm A, B, C trong mặt phẳng 2D (x, y).

    Trong bài toán Swing Golf:
        - Điểm A: Khớp vai (Shoulder)
        - Điểm B: Khớp khuỷu tay / cùi chỏ (Elbow) - Đỉnh góc
        - Điểm C: Khớp cổ tay (Wrist)

    Công thức Toán học Vector:
        Vector BA = A - B
        Vector BC = C - B
        cos(theta) = (BA . BC) / (|BA| * |BC|)
        theta = arccos(cos(theta)) chuyển sang độ (0° đến 180°)

    Tham số:
        point_a: Tọa độ [x, y] của điểm đầu (Vai)
        point_b: Tọa độ [x, y] của đỉnh góc (Cùi chỏ)
        point_c: Tọa độ [x, y] của điểm cuối (Cổ tay)

    Trả về:
        float: Góc theta tính bằng độ (từ 0.0° đến 180.0°)
    """
    a = np.array(point_a[:2], dtype=np.float64)
    b = np.array(point_b[:2], dtype=np.float64)
    c = np.array(point_c[:2], dtype=np.float64)

    # Tạo 2 vector từ đỉnh B
    ba = a - b
    bc = c - b

    # Độ dài chuẩn (Norm / Magnitude) của 2 vector
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Tránh chia cho 0 nếu các điểm trùng nhau
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

    # Tích vô hướng (Dot Product)
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)

    # Giới hạn giá trị cos trong đoạn [-1.0, 1.0] để tránh lỗi số học làm tròn
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Tính góc arccos và đổi sang độ
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def calculate_angle_3d(point_a, point_b, point_c) -> float:
    """
    Tính góc uốn (đơn vị: Độ) tại đỉnh B tạo bởi 3 điểm A, B, C trong không gian 3D (x, y, z).
    Tận dụng dữ liệu tọa độ chiều sâu Z từ MediaPipe Pose 3D landmarks.

    Tham số:
        point_a: Tọa độ [x, y, z]
        point_b: Tọa độ [x, y, z] (Đỉnh góc)
        point_c: Tọa độ [x, y, z]

    Trả về:
        float: Góc không gian 3D (0.0° đến 180.0°)
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

    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))
