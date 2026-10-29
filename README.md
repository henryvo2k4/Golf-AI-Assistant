# HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)

Dự án Hệ thống Trợ lý AI Phân tích Tư thế Golf thời gian thực hỗ trợ Golfer phát hiện và sửa lỗi kỹ thuật swing ngoài sân tập.

---

## 1. KIẾN TRÚC HỆ THỐNG

- **Mắt thần (Camera):** Webcam EMEET C60E 60FPS Full HD (Sony CMOS sensor).
- **Bộ não xử lý (Mini PC):** HP ProDesk 600 G4 Mini (Intel Core i5-8500T 6 nhân, 8GB RAM, SSD NVMe 256GB).
- **Màn hình hiển thị & Loa:** Màn hình di động 10.5" 3:2 IPS (420 nits, loa kép tích hợp stereo).
- **Kích hoạt (Trigger):** Bàn đạp dậm chân USB công nghiệp (vỏ kim loại, cáp 5m, chuẩn phím Spacebar HID).
- **Nguồn điện:** Pin sạc dự phòng CUKTECH 120W 25.000mAh cấp điện độc lập 4-5 tiếng.

---

## 2. NGUYÊN LÝ HOẠT ĐỘNG CỦA AI

Hệ thống **không cần huấn luyện lại mô hình AI từ đầu**, mà sử dụng kiến trúc AI kết hợp:
1. **Google MediaPipe Pose:** Nhận diện và theo dõi 33 điểm mốc (3D Landmarks) trên cơ thể người chơi với tốc độ >60 FPS trực tiếp trên CPU.
2. **Biomechanical Vector Math:** Tính toán góc uốn của khuỷu tay trái (giữa Vai - Cùi chỏ - Cổ tay) bằng công thức tích vô hướng vector không gian.
3. **State Machine (Top of Swing Detection):** Tự động phát hiện thời điểm gậy vung lên đỉnh cao nhất (khi cổ tay đạt độ cao cực đại).
4. **PGA Rule Check & Audio Feedback:** Nếu góc tay trái ở đỉnh swing $< 150^\circ$, hệ thống lập tức khoanh tròn đỏ cùi chỏ và phát giọng nói tiếng Việt qua loa: *"Lỗi gập tay trái ở đỉnh swing, hãy giữ thẳng tay!"*.

---

## 3. CẤU TRÚC THƯ MỤC

```
D:\Henry\Golf\
├── config.py                 # File cấu hình tham số (ngưỡng góc, phím trigger, màu sắc UI)
├── requirements.txt          # Danh sách thư viện Python cần cài đặt
├── run_app.py                # File chạy chính (Main entry point)
├── run_test.bat              # Script 1-click chạy thử nghiệm trên Windows
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py       # Hàm toán học tính góc Vector giữa 3 điểm khớp
│   │   └── drawing.py        # Vẽ giao diện HUD, thước đo góc, khung xương cảnh báo
│   └── core/
│       ├── __init__.py
│       ├── pose_detector.py  # Wrapper MediaPipe Pose (xử lý hình ảnh 60FPS)
│       ├── swing_analyzer.py # Nhận diện pha đỉnh Swing & kiểm tra góc tay chuẩn PGA
│       └── voice_alert.py    # Luồng phát âm thanh/giọng nói TTS tiếng Việt đa luồng
├── assets/
│   ├── sounds/               # Âm thanh hiệu ứng (beep, ting, error)
│   └── videos/               # Video clip mẫu quay cú swing để test khi chưa cắm webcam thật
└── tests/
    └── test_geometry.py      # Bộ kiểm thử toán học tính góc vector
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY THỬ

### Bước 1: Cài đặt Python 3.10 hoặc 3.11
Cài đặt qua Windows `winget` hoặc tải từ trang chủ `python.org` (Lưu ý tích chọn **Add Python to PATH**).

### Bước 2: Cài đặt các thư viện cần thiết
Mở Terminal / PowerShell tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy chương trình
- Chạy với Webcam máy tính:
  ```bash
  python run_app.py --source 0
  ```
- Chạy với Video clip mẫu giả lập:
  ```bash
  python run_app.py --source assets/videos/sample_swing.mp4
  ```
- Thao tác:
  - Bấm phím **SPACE** (Phím Cách) trên bàn phím để giả lập cú dậm chân của Bàn đạp USB Foot Pedal.
  - Bấm phím **Q** để thoát chương trình.
