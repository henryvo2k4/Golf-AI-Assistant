"""
MODULE HIỂN THỊ GIAO DIỆN & VẼ HUD (HEADS-UP DISPLAY)
Toàn bộ phản hồi hiển thị trực quan 100% trên màn hình.
"""

import cv2
import numpy as np


def draw_hud(frame, fps: float, current_angle: float, peak_angle: float, state: str, is_error: bool, min_angle: float = 150.0):
    """
    Vẽ Dashboard trực quan trên màn hình (Visual HUD).
    """
    h, w, _ = frame.shape

    # 1. Thanh Menu ngang trên cùng (Bán trong suốt)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Trạng thái hoạt động
    if state == "IDLE":
        status_text = "SAN SANG (DAM BAN DAP DE DANH)"
        status_color = (0, 230, 255)  # Vàng
    elif state == "RECORDING":
        status_text = "DANG QUAY CU SWING... (DUA GAY LEN DINH)"
        status_color = (0, 140, 255)  # Cam
    elif state == "ANALYZED":
        status_text = "DA PHAN TICH XONG CU SWING"
        status_color = (0, 0, 255) if is_error else (0, 230, 64)
    else:
        status_text = state
        status_color = (255, 255, 255)

    cv2.putText(frame, f"TRANG THAI: {status_text}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # 3. Góc tay thời gian thực
    angle_color = (0, 0, 255) if current_angle < min_angle else (0, 230, 64)
    cv2.putText(frame, f"Goc tay truc tiep: {current_angle:.1f} do", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, angle_color, 2, cv2.LINE_AA)

    # 4. Góc tại đỉnh Swing
    if peak_angle > 0:
        peak_color = (0, 0, 255) if is_error else (0, 230, 64)
        peak_text = f"Dinh Swing: {peak_angle:.1f} do (PGA >= {min_angle:.0f} do)"
    else:
        peak_color = (180, 180, 180)
        peak_text = f"Dinh Swing: --.- do (PGA >= {min_angle:.0f} do)"
    cv2.putText(frame, peak_text, (380, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, peak_color, 2, cv2.LINE_AA)

    # 5. FPS & Phím điều khiển
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    # 6. BANNER KẾT QUẢ NỔI BẬT KHI PHÂN TÍCH XONG (Thay thế hoàn toàn giọng nói)
    if state == "ANALYZED":
        banner_w = 680
        banner_h = 75
        bx = (w - banner_w) // 2
        by = 100

        # Nền banner
        bg_color = (0, 0, 180) if is_error else (0, 160, 40)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), bg_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (255, 255, 255), 2)

        if is_error:
            cv2.putText(frame, "LOI: GAP TAY TRAI O DINH SWING!", (bx + 25, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Goc dat: {peak_angle:.1f} do - Khuyen nghi: Giu thang tay >= {min_angle:.0f} do",
                        (bx + 25, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 240, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "XUAT SAC: TAY TRAI THANG DAT CHUAN PGA!", (bx + 20, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Goc dat: {peak_angle:.1f} do (Toi uu cho cu danh uy luc)",
                        (bx + 20, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Hướng dẫn dậm chân ở góc dưới màn hình
    guide_overlay = frame.copy()
    cv2.rectangle(guide_overlay, (0, h - 35), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(guide_overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "[SPACE / DAM CHAN]: Bat dau danh | [R]: Reset | [Q]: Thoat",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


def draw_skeleton_with_alerts(frame, shoulder_pt, elbow_pt, wrist_pt, angle: float, is_error: bool, min_angle: float = 150.0):
    """
    Vẽ đường nối 3 khớp tay trái (Vai -> Cùi chỏ -> Cổ tay) và các vòng tròn cảnh báo trực quan.
    """
    line_color = (0, 0, 255) if is_error else (0, 230, 64)
    thickness = 4

    p1 = (int(shoulder_pt[0]), int(shoulder_pt[1]))
    p2 = (int(elbow_pt[0]), int(elbow_pt[1]))
    p3 = (int(wrist_pt[0]), int(wrist_pt[1]))

    # Vẽ đường nối cánh tay
    cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)
    cv2.line(frame, p2, p3, line_color, thickness, cv2.LINE_AA)

    # Khớp vai và cổ tay
    cv2.circle(frame, p1, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p3, 7, (255, 255, 255), -1, cv2.LINE_AA)

    # Khớp cùi chỏ (Elbow) - Điểm trọng tâm đo góc
    if is_error:
        # Vòng tròn đỏ nhấp nháy nổi bật
        cv2.circle(frame, p2, 16, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, p2, 8, (0, 0, 255), -1, cv2.LINE_AA)

        # Hộp nhãn góc lỗi
        label_text = f"{angle:.0f} do (GAP)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (p2[0] + 10, p2[1] - 25), (p2[0] + 15 + tw, p2[1] - 5), (0, 0, 180), -1)
        cv2.putText(frame, label_text, (p2[0] + 12, p2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, p2, 9, (0, 230, 64), -1, cv2.LINE_AA)
        label_text = f"{angle:.0f} do"
        cv2.putText(frame, label_text, (p2[0] + 12, p2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 64), 2, cv2.LINE_AA)
