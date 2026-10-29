"""
KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CHO MODULE TOÁN HỌC HÌNH HỌC VECTOR
Chứng minh tính chính xác tuyệt đối 100% của giải thuật đo góc khuỷu tay.
"""

import sys
import os
import math
import pytest

# Thêm thư mục gốc vào đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from src.utils.geometry import calculate_angle_2d, calculate_angle_3d


def test_straight_line_180_degrees():
    """
    Kiểm tra cánh tay thẳng tắp (180 độ)
    Vai: (0, 100), Cùi chỏ: (0, 50), Cổ tay: (0, 0)
    """
    a = (0, 100)
    b = (0, 50)
    c = (0, 0)
    angle = calculate_angle_2d(a, b, c)
    assert math.isclose(angle, 180.0, abs_tol=1e-3), f"Ky vong 180 do, nhan duoc {angle}"


def test_right_angle_90_degrees():
    """
    Kiểm tra góc vuông 90 độ (cùi chỏ gập vuông)
    Vai: (0, 10), Cùi chỏ: (0, 0), Cổ tay: (10, 0)
    """
    a = (0, 10)
    b = (0, 0)
    c = (10, 0)
    angle = calculate_angle_2d(a, b, c)
    assert math.isclose(angle, 90.0, abs_tol=1e-3), f"Ky vong 90 do, nhan duoc {angle}"


def test_pga_threshold_150_degrees():
    """
    Kiểm tra góc 150 độ (ngưỡng phân định giữa Chuẩn PGA và Lỗi gập tay)
    """
    b = (0.0, 0.0)
    a = (1.0, 0.0)
    # Điểm C tạo góc 150 độ so với trục X: cos(150°) = -sqrt(3)/2, sin(150°) = 0.5
    c = (math.cos(math.radians(150)), math.sin(math.radians(150)))
    angle = calculate_angle_2d(a, b, c)
    assert math.isclose(angle, 150.0, abs_tol=1e-3), f"Ky vong 150 do, nhan duoc {angle}"


def test_zero_division_guard():
    """
    Kiểm tra trường hợp 2 điểm trùng nhau -> Không được gây lỗi Crash / ZeroDivisionError
    """
    a = (10, 10)
    b = (10, 10)
    c = (20, 20)
    angle = calculate_angle_2d(a, b, c)
    assert angle == 0.0, "Ham phai tra ve 0.0 khi co diem trung nhau"


def test_3d_angle():
    """
    Kiểm tra góc 3D không gian
    """
    a = (1, 0, 0)
    b = (0, 0, 0)
    c = (0, 1, 0)
    angle = calculate_angle_3d(a, b, c)
    assert math.isclose(angle, 90.0, abs_tol=1e-3)


if __name__ == "__main__":
    test_straight_line_180_degrees()
    test_right_angle_90_degrees()
    test_pga_threshold_150_degrees()
    test_zero_division_guard()
    test_3d_angle()
    print("[SUCCESS] Tat ca 5 bai kiem tra toan hoc Vector deu PASSED 100%!")
