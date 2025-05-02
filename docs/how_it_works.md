# How does it work? 🤔

The setup consists of two Python scripts and **GStreamer** glue code.

```
┌────────┐    USB3     ┌──────────────┐     RTP/UDP      ┌──────────────┐
│ D435i  │──────────▶ │ Jetson NX /  │ ───────────────▶ │  Laptop /    │
└────────┘            │  jetson_re.. │  5600  5602      │receive_..gst │
                      └──────────────┘                   └──────────────┘
                      (H.264 / RGB)     (H.264 / depth-colormap)
```

## 1. `jetson_realsense_stream.py`

Runs **on the Jetson NX** inside the robot.

1. Uses **`pyrealsense2`** to open the camera with the resolution & FPS you specify.
2. Captures two streams every frame:
   • `rs.stream.color` → BGR 8-bit image.  
   • `rs.stream.depth` → 16-bit millimetre depth image.
3. The depth frame is optionally run through a **temporal filter** to reduce noise and is then colour-mapped to an 8-bit BGR image using OpenCV (Plasma LUT).  This makes it easy to encode with standard video codecs.
4. Two **`GstAppSrc`** elements act as entrypoints into a GStreamer pipeline:

   ```text
   RGB  : AppSrc → videoconvert → nvvidconv → nvv4l2h264enc → rtph264pay → udpsink 5600
   Depth: AppSrc → videoconvert → nvvidconv → nvv4l2h264enc → rtph264pay → udpsink 5602
   ```

   *The Jetson’s hardware H.264 encoder (`nvv4l2h264enc`) keeps CPU usage low.*

5. Result: **Two RTP streams** leave the Jetson over the network:
   • **5600/UDP** – H.264 encoded RGB
   • **5602/UDP** – H.264 encoded colourised depth

## 2. `receive_realsense_gst.py`

Runs **on your PC**.

1. Builds two **GStreamer pipelines** that start with `udpsrc` and end with an **`GstAppSink`**:

   ```text
   udpsrc → rtph264depay → avdec_h264 → videoconvert → BGR → AppSink
   ```

2. Buffers pulled from the sinks are converted to NumPy arrays, stacked side-by-side and displayed with `cv2.imshow()`.

3. If you need the *raw* 16-bit depth instead of a colour-mapped preview see the alternative script [`receive_realsense_stream.py`](../receive_realsense_stream.py) that uses OpenCV’s GStreamer backend and RFC-4175 raw video.

## 3. Network considerations

• A wired Gigabit connection easily copes with the combined 6–8 Mbit/s bitrate.  
• On Wi-Fi lower latencies can be achieved by keeping both devices on the same AP and 5 GHz band.  
• The receiver trusts frames to arrive in order; avoid WAN/VPN links that may re-order or drop UDP traffic.

## 4. Customising resolution / FPS / bitrate

*Jetson →* Supply `--width`, `--height`, `--fps` to change the camera configuration.  
*Encoder settings →* Adjust the `bitrate=` property of the two `nvv4l2h264enc` instances inside `jetson_realsense_stream.py` if you need higher quality.
