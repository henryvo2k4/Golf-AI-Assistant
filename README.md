# HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)

Ứng dụng Desktop chuyên nghiệp hỗ trợ Golfer tập luyện và phát hiện lỗi kỹ thuật Swing bằng AI.
Giao diện **PySide6 (Qt6)** hiện đại, tiếng Việt có dấu sắc nét, không hiện cửa sổ CMD.

---

## 1. QUY TRÌNH SỬ DỤNG TRÊN ỨNG DỤNG

1. **Mở ứng dụng:** Nhấp đúp file **`Mo_Ung_Dung_Golf_AI.vbs`** (hoặc `run_test.bat`) — Màn hình Camera Live hiển thị trực tiếp.
2. **Bấm nút / Dậm bàn đạp (`SPACE`):** Hệ thống **đếm ngược 5 giây** (`5... 4... 3... 2... 1...`) để người chơi bước vào thảm chuẩn bị tư thế.
3. **Tự động quay 10 giây:** Ghi hình trọn vẹn cú đánh (có nút **Phân tích ngay** nếu đánh xong sớm hơn 10s).
4. **Phân tích toàn diện 5 lỗi Golf chuẩn PGA & Khoanh vùng lỗi trên ảnh:**
   - **Gập khuỷu tay dẫn hướng (Chicken Wing):** Đo góc vai – khuỷu – cổ tay tại đỉnh swing & chạm bóng (chuẩn $\ge 150^\circ$).
   - **Nhấp nhô / Ngẩng đầu sớm (Head Stability):** Đo biên độ dao động dọc đầu so với lúc Setup (chuẩn $\le 7\%$ chiều cao).
   - **Ngược trục cột sống (Reverse Spine Angle):** Đo độ nghiêng thân trên tại đỉnh swing (không đổ ngược về phía mục tiêu).
   - **Trượt / Lắc hông ngang (Pelvic Sway):** Phát hiện hông sau trượt quá trục chân sau trong pha backswing.
   - **Ổn định chân trụ & đầu gối:** Kiểm tra lỗi sụp đầu gối ($< 128^\circ$) hoặc bật thẳng gối quá sớm (Early Extension).
5. **Xem lại Slow-Motion:** Chuyển sang tab **`🎬 Xem lại Slow-Motion`** để tua chậm cú đánh ở tốc độ `0.25x`, `0.5x`, `1.0x` kèm khung xương AI bám theo từng khung hình.

---

## 2. CẤU TRÚC MÃ NGUỒN

```
D:\Henry\Golf\
├── Mo_Ung_Dung_Golf_AI.vbs   # Mở thẳng App Desktop (không hiện CMD đen)
├── run_test.bat              # Launcher dự phòng
├── run_app.py                # Điểm khởi chạy ứng dụng PySide6
├── config.py                 # Ngưỡng chuẩn PGA, cấu hình đếm ngược 5s & quay 10s
├── requirements.txt          # opencv-python, mediapipe, PySide6, pillow, numpy, pytest
├── assets/models/
│   └── pose_landmarker_full.task  # Mô hình AI Google MediaPipe Tasks
├── src/
│   ├── core/
│   │   ├── pose_detector.py  # Wrapper MediaPipe 1.2.2 PoseLandmarker
│   │   └── swing_analyzer.py # Phân tích chuỗi video & chấm điểm 5 lỗi PGA
│   ├── utils/
│   │   ├── geometry.py       # Toán học Vector không gian
│   │   └── drawing.py        # Vẽ vòng tròn khoanh vùng lỗi & hộp text tiếng Việt (Pillow)
│   └── ui/
│       └── main_window.py    # Giao diện Desktop PySide6 (Live, Countdown, Replay, Cards)
└── tests/
    ├── test_geometry.py
    └── test_pipeline.py
```

---

## 3. KIỂM THỬ

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
