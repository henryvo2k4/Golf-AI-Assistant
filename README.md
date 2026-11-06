# HỆ THỐNG TRỢ LÝ AI PHÂN TÍCH TƯ THẾ GOLF (VAR GOLF)

Hệ thống Trợ lý AI Phân tích Tư thế Golf thời gian thực hỗ trợ Golfer phát hiện và sửa lỗi kỹ thuật swing ngoài sân tập.
Chế độ hoạt động: **Hiển thị trực quan 100% trên màn hình (Visual HUD & On-screen Alerts)**.

---

## 1. KIẾN TRÚC PHẦN CỨNG

- **Mắt thần (Camera):** Webcam EMEET C60E 60FPS Full HD (Sony CMOS sensor, ren ốc 1/4" chuẩn tripod).
- **Bộ não xử lý (Mini PC):** HP ProDesk 600 G4 Mini (Intel Core i5-8500T 6 nhân, 8GB RAM, SSD NVMe 256GB).
- **Màn hình hiển thị:** Màn hình di động 10.5" 3:2 IPS (420 nits chống chói ngoài trời).
- **Kích hoạt (Trigger):** Bàn đạp dậm chân USB công nghiệp (vỏ kim loại, cáp 5m, chuẩn phím Spacebar HID).
- **Nguồn điện:** Pin sạc dự phòng CUKTECH 120W 25.000mAh cấp điện độc lập 4-5 tiếng.

---

## 2. NGUYÊN LÝ HOẠT ĐỘNG CỦA AI

1. **Google MediaPipe Pose:** Nhận diện và theo dõi 33 điểm mốc (3D Landmarks) trên cơ thể người chơi với tốc độ >60 FPS trực tiếp trên CPU.
2. **Biomechanical Vector Math:** Tính toán góc uốn của khuỷu tay trái (giữa Vai 11 - Cùi chỏ 13 - Cổ tay 15) bằng công thức tích vô hướng vector không gian.
3. **State Machine (Top of Swing Detection):** Tự động phát hiện thời điểm gậy vung lên đỉnh cao nhất (khi cổ tay đạt độ cao cực đại trong khung hình).
4. **Phản hồi trực quan 100% trên màn hình (Visual Alert):**
   - **Tư thế Đạt chuẩn PGA ($\ge 150^\circ$):** Đường khung xương tay màu xanh lá, hiện Banner lớn màu xanh: *"XUAT SAC: TAY TRAI THANG DAT CHUAN PGA"*.
   - **Tư thế Lỗi gập tay ($< 150^\circ$):** Khoanh 2 vòng tròn đỏ nhấp nháy tại cùi chỏ, hiện Banner đỏ cảnh báo: *"LOI: GAP TAY TRAI O DINH SWING (Goc chi dat XX do)"*.

---

## 3. CẤU TRÚC DỰ ÁN

```
D:\Henry\Golf\
├── .venv/                    # Môi trường ảo Python 3.11 độc lập
├── config.py                 # File cấu hình tham số (ngưỡng 150°, phím trigger, màu sắc UI)
├── requirements.txt          # Danh sách thư viện Python (opencv, mediapipe, numpy, pytest)
├── run_app.py                # File chạy chính (Main entry point)
├── run_test.bat              # Script 1-click chạy thử nghiệm trên Windows
├── README.md                 # Hướng dẫn chi tiết
│
├── src/
│   ├── utils/
│   │   ├── geometry.py       # Giải thuật Vector tính góc uốn khớp 2D/3D
│   │   └── drawing.py        # Vẽ Dashboard HUD, thước đo góc & Banner cảnh báo
│   └── core/
│       ├── pose_detector.py  # Wrapper MediaPipe Pose (xử lý hình ảnh 60FPS)
│       └── swing_analyzer.py # State Machine nhận diện đỉnh Swing & chấm điểm
│
└── tests/
    └── test_geometry.py      # Bộ kiểm thử toán học Vector (Unit Tests)
```

---

## 4. HƯỚNG DẪN CHẠY THỬ NGHIỆM

### Cách 1: Chạy 1-Click bằng file BAT (Khuyên dùng)
Nhấp đúp chuột vào file **`run_test.bat`**:
- Nhấn phím `1` để khởi động ứng dụng với Webcam.
- Nhấn phím `2` để chạy kiểm tra toán học Vector.

### Cách 2: Chạy bằng dòng lệnh Terminal
```bash
# Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# Chạy với Webcam
python run_app.py --source 0
```

### Thao tác điều khiển trong ứng dụng:
- Bấm phím **SPACE** (hoặc dậm bàn đạp chân): Bắt đầu ghi nhận cú swing.
- Bấm phím **R**: Reset về trạng thái ban đầu để đánh cú mới.
- Bấm phím **Q**: Thoát chương trình.
