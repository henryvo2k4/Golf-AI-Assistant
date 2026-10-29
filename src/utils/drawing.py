"""
MODULE HIỂN THỊ GIAO DIỆN & VẼ HUD (HEADS-UP DISPLAY)
Vẽ thông số phân tích tư thế trực tiếp lên khung hình video (OpenCV).
"""

import cv2
import numpy as np


def draw_hud(frame, fps: float, current_angle: float, peak_angle: float, state: str, is_error: bool):
    """
    Vẽ thanh thông tin Dashboard (HUD) ở góc trên màn hình.
    """
    h, w, _ = frame.shape

    # Vẽ thanh nền bán trong suốt trên cùng
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 1. Trạng thái hệ thống (State)
    state_color = (0, 255, 255)  # Vàng
    if state == "RECORDING":
        state_color = (0, 140, 255)  # Cam
    elif state == "ANALYZED":
        state_color = (0, 0, 255) if is_error else (0, 230, 64)

    cv2.putText(frame, f"TRANG THAI: {state}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)

    # 2. Góc tay hiện tại & Góc đỉnh (Peak Angle)
    angle_color = (0, 0, 255) if current_angle < 150.0 else (0, 230, 64)
    cv2.putText(frame, f"Goc tay trai: {current_angle:.1f} do", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2, cv2.LINE_AA)

    peak_text = f"Dinh Swing: {peak_angle:.1f} do (Chuan PGA >= 150 do)" if peak_angle > 0 else "Dinh Swing: Dang doi..."
    cv2.putText(frame, peak_text, (380, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. FPS & Hướng dẫn phím bấm
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "[SPACE]: Dam chan | [R]: Reset | [Q]: Thoat", (380, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # 4. Nếu phát hiện lỗi: Hiện Banner cảnh báo đỏ nổi bật ở giữa phía trên
    if is_error and state == "ANALYZED":
        banner_w, banner_h = 560, 45
        bx = (w - banner_w) // 2
        by = 100
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 200), -1)
        cv2.putText(frame, "CANH BAO: GAP TAY TRAI O DINH SWING!", (bx + 15, by + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_skeleton_with_alerts(frame, shoulder_pt, elbow_pt, wrist_pt, angle: float, is_error: bool):
    """
    Vẽ đường nối 3 khớp tay trái (Vai -> Cùi chỏ -> Cổ tay) và vòng tròn cảnh báo.
    """
    line_color = (0, 0, 255) if is_error else (0, 230, 64)
    thickness = 4

    p1 = (int(shoulder_pt[0]), int(shoulder_pt[1]))
    p2 = (int(elbow_pt[0]), int(elbow_pt[1]))
    p3 = (int(wrist_pt[0]), int(wrist_pt[1]))

    # Vẽ đường nối Vai - Cùi chỏ và Cùi chỏ - Cổ tay
    cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)
    cv2.line(frame, p2, p3, line_color, thickness, cv2.LINE_AA)

    # Vẽ các khớp tròn
    cv2.circle(frame, p1, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p3, 7, (255, 255, 255), -1, cv2.LINE_AA)

    if is_error:
        # Cảnh báo lỗi: Vẽ 2 vòng tròn đồng tâm màu đỏ quanh cùi chỏ
        cv2.circle(frame, p2, 14, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{angle:.0f} do (Loi)", (p2[0] + 15, p2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, p2, 8, (0, 230, 64), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{angle:.0f} do", (p2[0] + 15, p2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 64), 2, cv2.LINE_AA)
