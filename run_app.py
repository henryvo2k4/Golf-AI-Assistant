"""
CHƯƠNG TRÌNH CHÍNH (MAIN ENTRY POINT)
HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)

Cách chạy:
- Chạy với Webcam:          python run_app.py --source 0
- Chạy với Video clip test: python run_app.py --source assets/videos/sample_swing.mp4
"""

import cv2
import time
import argparse
import sys
import os

# Thêm thư mục gốc vào sys.path để import chuẩn module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from src.core.pose_detector import GolfPoseDetector
from src.core.swing_analyzer import SwingAnalyzer
from src.core.voice_alert import VoiceAlertService
from src.utils.drawing import draw_hud, draw_skeleton_with_alerts


def main():
    parser = argparse.ArgumentParser(description="He thong Tro ly AI Golf (VAR Golf)")
    parser.add_argument("--source", type=str, default=str(config.DEFAULT_VIDEO_SOURCE),
                        help="Nguon video: 0 cho Webcam mac dinh hoac duong dan file .mp4")
    parser.add_argument("--min-angle", type=float, default=config.LEAD_ARM_MIN_ANGLE,
                        help="Nguong goc tay trai toi thieu (mac dinh 150 do)")
    parser.add_argument("--handedness", type=str, default=config.GOLFER_HANDEDNESS,
                        choices=["right", "left"], help="Tay thuan golfer: right hoac left")
    args = parser.parse_args()

    # Chuyển đổi nguồn video (nếu là số thì ép kiểu int cho Webcam ID)
    video_source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  HE THONG TRO LY AI PHAN TICH TU THE GOLF (VAR GOLF) ")
    print("=" * 60)
    print(f"[*] Nguon video: {video_source}")
    print(f"[*] Nguoi choi thuan: Tay {args.handedness.upper()} (Tay dan huong: {'TRAI' if args.handedness == 'right' else 'PHAI'})")
    print(f"[*] Nguong goc chuan PGA: >= {args.min_angle} do")
    print(f"[*] Dieu khien: [SPACE] Dam ban dap chan | [R] Reset | [Q] Thoat")
    print("=" * 60)

    # 1. Khởi tạo Camera
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"\n[!] LOI: Khong the mo nguon video: {video_source}")
        print("[!] Hay kiem tra lai ket noi Webcam hoac duong dan file video.\n")
        return

    # Tối ưu kích thước khung hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    # 2. Khởi tạo các phân hệ
    print("[*] Dang khoi tao mo hinh Google MediaPipe Pose...")
    detector = GolfPoseDetector()
    analyzer = SwingAnalyzer(min_arm_angle=args.min_angle, handedness=args.handedness)
    voice_service = VoiceAlertService(speech_rate=config.TTS_SPEECH_RATE, enabled=config.ENABLE_VOICE_ALERT)

    prev_time = time.time()
    fps = 0.0

    print("[+] He thong da san sang! Hay nhan [SPACE] de bat dau danh...")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # Nếu chạy bằng file video clip và hết video -> Tự động tua lại từ đầu để lặp lại
                if isinstance(video_source, str) and os.path.exists(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[!] Khong the doc khung hinh tu camera.")
                    break

            # Tính FPS thời gian thực
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # 3. Phân tích tư thế bằng MediaPipe Pose
            pose_results = detector.process_frame(frame)

            shoulder_pt, elbow_pt, wrist_pt = None, None, None
            if pose_results.pose_landmarks:
                shoulder_pt, elbow_pt, wrist_pt = detector.get_lead_arm_points(
                    pose_results.pose_landmarks, frame.shape, handedness=args.handedness
                )

            # 4. Cập nhật bộ não phân tích Swing
            analysis = analyzer.update(shoulder_pt, elbow_pt, wrist_pt)

            # 5. Nếu vừa hoàn tất cú swing -> Kích hoạt âm thanh/giọng nói cảnh báo
            if analysis["just_finished"]:
                peak_angle = analysis["peak_angle"]
                if analysis["is_error"]:
                    print(f"\n>>> [CANH BAO] GAP TAY TRAI O DINH SWING! Goc: {peak_angle:.1f} do (< {args.min_angle} do)")
                    voice_service.alert_bent_arm(peak_angle)
                else:
                    print(f"\n>>> [XUAT SAC] TU THE CHUAN PGA! Goc: {peak_angle:.1f} do (>= {args.min_angle} do)")
                    voice_service.alert_good_swing(peak_angle)

            # 6. Vẽ giao diện trực quan (HUD + Skeleton)
            if shoulder_pt and elbow_pt and wrist_pt:
                is_arm_error = analysis["is_error"] if analysis["state"] == SwingAnalyzer.STATE_ANALYZED else (analysis["current_angle"] < args.min_angle)
                draw_skeleton_with_alerts(
                    frame, shoulder_pt, elbow_pt, wrist_pt,
                    analysis["current_angle"], is_arm_error
                )

            draw_hud(
                frame, fps,
                analysis["current_angle"],
                analysis["peak_angle"],
                analysis["state"],
                analysis["is_error"]
            )

            # Hiển thị cửa sổ
            cv2.imshow("He thong Tro ly AI Golf - VAR Golf (10.5 inch / 60 FPS)", frame)

            # 7. Bắt sự kiện bàn đạp dậm chân (Spacebar) hoặc phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == config.TRIGGER_KEY_CODE:  # Phím Space (Bàn đạp dậm chân)
                print("\n[FOOT PEDAL] Da dam chan! Bat dau ghi nhan cu swing...")
                analyzer.trigger_swing_start()
            elif key == config.RESET_KEY_CODE:  # Phím R
                print("\n[RESET] Da reset trang thai.")
                analyzer.reset()
            elif key == config.QUIT_KEY_CODE:   # Phím Q
                print("\n[QUIT] Dang thoat chuong trinh...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice_service.shutdown()
        print("[*] Da giai phong camera va tat he thong an toan.")


if __name__ == "__main__":
    main()
