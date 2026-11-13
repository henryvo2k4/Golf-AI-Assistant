# VAR GOLF STUDIO - AI GOLF SWING BIOMECHANICS KIOSK

Borderless fullscreen desktop application for automated Golf Swing posture diagnostics using **2D + 3D PGA / TPI Biomechanics** and Google MediaPipe PoseLandmarker.
Designed for **mouse-free / touch-free 1-button (`SPACE` / Foot Pedal) golf bay displays** with a clean **Modern White Bento UI**.

---

## 1. HANDS-FREE WORKFLOW

1. **Launch:** Open **`Mo_Ung_Dung_Golf_AI.vbs`** — launches directly into borderless fullscreen with zero console windows.
2. **Start (`SPACE` / Foot Pedal):** Triggers a **5-second countdown** (`5... 4... 3... 2... 1...`) with a radial HUD so the player can step into Address posture.
3. **Auto Capture (10s):** Records the complete swing (press `SPACE` anytime to stop early and analyze immediately).
4. **Horizontal Auto-Sliding Diagnostic Carousel:**
   - Evaluates **Bent Lead Elbow (3D)**, **Head Stability (2D Vertical & Lateral)**, **Shoulder Turn (3D X-Z Plane)**, **Face-On Reverse Spine Angle & Lateral Pelvic Sway** (or **Down-the-Line Early Extension / Posture Loss**), and **Lead Knee Stability (3D)**.
   - Displays each finding as a full-screen slide pairing the **joint-circled snapshot** with large **PGA/TPI coaching text**, holding **4 seconds per slide** before gliding horizontally to the next issue automatically.
   - Press **`SPACE`** anytime on the report screen to jump straight back to Live Camera for the next swing.

---

## 2. KEYBOARD SHORTCUTS

- `SPACE`: Start Countdown -> Stop Recording Early -> Next Swing
- `Left` / `Right`: Step diagnostic slides manually (optional)
- `H`: Toggle `Right-Handed` / `Left-Handed` on Live screen
- `M`: Toggle camera mirror mode
- `F11`: Toggle fullscreen / windowed mode
- `Ctrl+Q`: Exit application

---

## 3. TESTING

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
