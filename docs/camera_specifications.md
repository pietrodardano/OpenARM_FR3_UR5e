# HBV HD CAMERA (Workcell RGB Camera)

The only perception sensor of the demo: an RGB camera (no depth) fixed on the torso
mount, looking forward and 20° down at the table ([physical_workcell.md](physical_workcell.md)).
The vision branch runs GroundingDINO + SAM2 (+ optional VLM) on its frames and projects
mask centroids onto the known table plane ([demo_task.md](demo_task.md#3-frames-and-the-vision-contract)).

## 1. Hardware Identification & System Profile

Extracted directly via macOS hardware subsystem (`system_profiler SPCameraDataType` & `AVFoundation` backend):

| Parameter | Value |
|---|---|
| **Device Name** | `HBV HD CAMERA` |
| **Model ID** | `UVC Camera VendorID_2760 ProductID_838` |
| **USB VID / PID** | `0x0ac8` (Z-Star Microelectronics Corp.) / `0x0346` (HBVCAM module) |
| **Unique ID** | `0x1100000ac80346` |
| **Protocol** | Standard USB Video Class (UVC 1.1 / 1.5) & USB Audio Class (UAC) |
| **Interface** | USB 2.0 High-Speed (Plug-and-play, no proprietary driver needed) |
| **Built-in Audio** | Integrated USB microphone (`HBV HD CAMERA` audio channel) |

---

## 2. Video Capture Specifications

- **Native Resolution:** 1280 × 720 (720p HD) and 1920 × 1080 (1080p Full HD)
- **Framerate:** Up to 30 FPS progressive
- **Supported Native Pixel Formats:**
  - `uyvy422` (preferred raw format for lowest latency on macOS AVFoundation)
  - `yuyv422`
  - `nv12`
  - `0rgb` / `bgr0`
- **Field of View (FOV):** Wide-angle lens (approx. 90°–120° DFOV typical for HBVCAM modules); the scene model uses a provisional 70° vertical FOV until intrinsic calibration
- **Mounting Note:** If the module is mounted upside down on the torso bracket, apply a 180° rotation in software (`vision.rotate_180`) before calibration and detection.

---

## 3. Platform Device Indices

### macOS (AVFoundation)
- **Device Name:** `"HBV HD CAMERA"`
- **Device Index:** Typically `1` when MacBook built-in camera is index `0`:
  - `[0]`: MacBook Pro Camera
  - `[1]`: **HBV HD CAMERA**

### Linux (V4L2)
- Automatically exposed as `/dev/video*` via standard kernel module `uvcvideo`.
- Audio exposed via `snd-usb-audio`.

---

## 4. Software Access Examples

### Python (OpenCV)
```python
import cv2

# Initialize HBV HD Camera (index 1 on macOS with built-in webcam)
cap = cv2.VideoCapture(1)

# Configure stream format
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30.0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # If the camera is mounted upside down on its bracket:
    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    cv2.imshow("HBV HD Camera - Workcell View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### FFmpeg / Command Line Stream
```bash
# Capture single test frame on macOS
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -pix_fmt uyvy422 -i "HBV HD CAMERA" -vframes 1 -y snapshot.jpg

# Low-latency UDP stream to a vision process on another machine:
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -pix_fmt uyvy422 -i "HBV HD CAMERA" \
       -vf "hflip,vflip" -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts udp://127.0.0.1:5555
```
