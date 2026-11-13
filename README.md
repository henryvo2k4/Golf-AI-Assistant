# HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF STUDIO)

Ứng dụng Desktop chuyên nghiệp hỗ trợ Golfer tập luyện và phát hiện lỗi kỹ thuật Swing bằng AI 2D + 3D chuẩn cơ sinh học **PGA / TPI**.
Giao diện **Dark Mode OLED Sports Analytics Pro (PySide6 / Qt6)** thiết kế theo chuẩn `ui-ux-pro-max`, 100% icon vector SVG sắc nét, tiếng Việt có dấu, không hiện cửa sổ CMD.

---

## 1. QUY TRÌNH SỬ DỤNG TRÊN ỨNG DỤNG

1. **Mở ứng dụng:** Nhấp đúp file **`Mo_Ung_Dung_Golf_AI.vbs`** (hoặc `run_test.bat`) — Màn hình Camera Live hiển thị trực tiếp cùng khung ngắm căn chỉnh.
2. **Bấm nút / Dậm bàn đạp (`SPACE`):** Hệ thống **đếm ngược 5 giây** (`5... 4... 3... 2... 1...`) kèm vòng tiến độ để người chơi bước vào thảm chuẩn bị tư thế.
3. **Tự động quay 10 giây:** Ghi hình trọn vẹn cú đánh (có nút **Dừng sớm & Phân tích ngay** nếu đánh xong sớm hơn 10s).
4. **Phân tích AI 2D + 3D chuẩn PGA/TPI & Khoanh vùng lỗi:**
   - **Tự động phân pha & nhận diện góc quay:** Bộ lọc Gaussian 5 khung hình khử nhiễu, phân tách chính xác 4 pha (`Chuẩn bị`, `Đỉnh Swing`, `Chạm bóng`, `Kết thúc`) và tự nhận diện góc quay **Chính diện (Face-On)** hoặc **Nhìn ngang (Down-the-Line)**.
   - **Gập khuỷu tay dẫn hướng (Chicken Wing):** Đo góc vai – khuỷu – cổ tay trong **không gian 3D thực** tại Đỉnh Swing & Chạm bóng (chuẩn `>= 150°`).
   - **Ổn định đầu hai chiều (Head Stability):** Đo biên độ dao động dọc & ngang của đầu so với lúc Setup (chuẩn `<= 7%` dọc, `<= 8%` ngang chiều cao cơ thể).
   - **Độ xoay vai tại Đỉnh Swing (Shoulder Turn X-Z):** Đo góc xoay vai thực trong mặt phẳng ngang 3D X-Z (chuẩn `>= 65°`).
   - **Trục cột sống & Hông theo góc máy:**
     - *Chính diện (Face-On):* Chấm **Ngược trục cột sống (Reverse Spine Angle)** & **Trượt hông ngang (Pelvic Sway)**.
     - *Nhìn ngang (Down-the-Line):* Chấm **Mất góc gập lưng / Nhổm người (Early Extension / Loss of Posture)**.
   - **Ổn định chân trụ & Đầu gối:** Kiểm tra lỗi sụp gối (`< 128°`) hoặc duỗi cứng đầu gối (`> 176°`) trong không gian 3D.
5. **Xem lại Slow-Motion:** Chuyển sang tab **Xem lại Slow-Motion** để tua chậm cú đánh ở tốc độ `0.25x`, `0.50x`, `1.00x` kèm khung xương AI bám theo từng khung hình.

---

## 2. CẤU TRÚC MÃ NGUỒN

```text
D:\Henry\Golf\
├── Mo_Ung_Dung_Golf_AI.vbs      # Mở thẳng App Desktop (không hiện CMD đen)
├── run_test.bat                 # Launcher dự phòng
├── run_app.py                   # Điểm khởi chạy ứng dụng PySide6
├── config.py                    # Ngưỡng chuẩn 2D+3D PGA/TPI, đếm ngược 5s & quay 10s
├── requirements.txt             # opencv-python, mediapipe, PySide6, pillow, numpy, pytest
├── assets/models/
│   └── pose_landmarker_full.task  # Mô hình AI Google MediaPipe Tasks
├── src/
│   ├── core/
│   │   ├── pose_detector.py     # Trích xuất song song khớp 2D Pixel + 3D World Mét
│   │   └── swing_analyzer.py    # Lọc mượt, phân 4 pha swing & chấm lỗi 2D+3D PGA/TPI
│   ├── utils/
│   │   ├── geometry.py          # Toán học Vector 2D/3D & Gaussian 5-tap smoothing
│   │   └── drawing.py           # Vẽ vòng tròn khoanh vùng lỗi & callout tiếng Việt (Pillow)
│   └── ui/
│       ├── theme.py             # Design tokens Dark OLED & QSS stylesheet
│       ├── icons.py             # Bộ icon vector SVG (QSvgRenderer, không dùng emoji)
│       └── main_window.py       # Giao diện Desktop PySide6 (Live, Countdown, Replay, Cards)
└── tests/
    ├── test_geometry.py
    └── test_pipeline.py
```

---

## 3. KIỂM THỬ

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
