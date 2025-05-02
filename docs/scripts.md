# Script reference 📜

| Script | Location | Run on | Purpose |
|--------|----------|--------|---------|
| `list_realsense_devices.py` | repo root | Jetson **or** PC | Print model, serial & firmware of each connected camera. Handy sanity-check. |
| `stream_realsense.py` | repo root | either | Display RGB + depth locally without the whole network pipeline. Supports IR & IMU. |
| `jetson_realsense_stream.py` | repo root | **Jetson NX** | Capture RealSense and push **two RTP/H.264 streams** (RGB & depth) to a client IP. |
| `receive_realsense_gst.py` | repo root | **PC/Laptop** | Pure-GStreamer receiver that works even if your OpenCV build lacks GStreamer. Displays colour + depth. |
| `receive_realsense_stream.py` | repo root | **PC/Laptop** | Alternative receiver that relies on `cv2.VideoCapture` (requires OpenCV compiled with `WITH_GSTREAMER=ON`). Provides access to *raw* 16-bit depth. |

Parameters accepted by the main scripts:

## `jetson_realsense_stream.py`

```
usage: jetson_realsense_stream.py [-h] --client-ip IP [--width W] [--height H] [--fps FPS]

optional arguments:
  -h, --help            show this help message and exit
  --client-ip IP        IPv4 address of the receiver (required)
  --width W             frame width  [default: 640]
  --height H            frame height [default: 480]
  --fps FPS             frames per second [default: 30]
```

## `receive_realsense_gst.py`

No parameters – the ports & formats are hard-coded to match the sender.

If you need to adjust them look for these constants at the top of the file:

```python
RGB_PORT = 5600
DEPTH_PORT = 5602
WIDTH = 640
HEIGHT = 480
FPS = 30
```
