"""receive_realsense_gst.py  –  **GStreamer-only** client

This variant does **not** rely on OpenCV’s `cv2.VideoCapture`, therefore it
works even when your OpenCV build lacks GStreamer support (which is the case
for the default Ubuntu `python3-opencv` and the PyPI wheels).

It uses PyGObject (gst-python) to pull buffers directly from `appsink`,
converts them to NumPy arrays, colourises the 16-bit depth, and shows the
combined image with OpenCV’s highgui (OpenCV itself is only used for display
and colour-mapping – both of which still work without GStreamer).

Run on the **laptop / workstation** that should receive the streams emitted by
`jetson_realsense_stream.py`.

Dependencies
------------
sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 \
                 gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad gstreamer1.0-libav

python3 -m pip install --upgrade numpy opencv-python  # highgui only
"""

from __future__ import annotations

import sys
import time

import numpy as np
import cv2

# GStreamer
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp


RGB_PORT = 5600
DEPTH_PORT = 5602
WIDTH = 640
HEIGHT = 480
FPS = 30


def build_rgb_sink() -> tuple[GstApp.AppSink, Gst.Pipeline]:
    pipeline = Gst.parse_launch(
        f"udpsrc port={RGB_PORT} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def build_depth_sink() -> tuple[GstApp.AppSink, Gst.Pipeline]:
    # Depth is colour-mapped and H.264 encoded (pt=97) on the Jetson.
    pipeline = Gst.parse_launch(
        f"udpsrc port={DEPTH_PORT} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def colourise_depth(depth16: np.ndarray) -> np.ndarray:
    depth_clip = np.clip(depth16, 0, 6000)
    depth8 = cv2.convertScaleAbs(depth_clip, alpha=255.0 / 6000)
    return cv2.applyColorMap(depth8, cv2.COLORMAP_PLASMA)


def main() -> None:
    Gst.init(None)

    rgb_sink, rgb_pipeline = build_rgb_sink()
    depth_sink, depth_pipeline = build_depth_sink()

    for p in (rgb_pipeline, depth_pipeline):
        p.set_state(Gst.State.PLAYING)

    last = time.perf_counter()

    try:
        while True:
            sample_rgb = rgb_sink.emit("try-pull-sample", Gst.SECOND // FPS)
            sample_d = depth_sink.emit("try-pull-sample", Gst.SECOND // FPS)

            if not sample_rgb or not sample_d:
                # No frame yet – avoid busy loop
                time.sleep(0.005)
                continue

            buf_rgb = sample_rgb.get_buffer()
            buf_d = sample_d.get_buffer()

            # Extract RGB image
            rgb = np.frombuffer(buf_rgb.extract_dup(0, buf_rgb.get_size()), dtype=np.uint8)
            rgb = rgb.reshape((HEIGHT, WIDTH, 3))

            depth_bgr = np.frombuffer(buf_d.extract_dup(0, buf_d.get_size()), dtype=np.uint8)
            depth_bgr = depth_bgr.reshape((HEIGHT, WIDTH, 3))

            combo = cv2.hconcat([rgb, depth_bgr])

            # FPS overlay
            now = time.perf_counter()
            fps = 1.0 / (now - last)
            last = now
            cv2.putText(combo, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("RGB + Depth", combo)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    finally:
        for p in (rgb_pipeline, depth_pipeline):
            p.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)
