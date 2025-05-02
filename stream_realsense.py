"""
stream_realsense.py
====================

A small convenience wrapper around Intel RealSense SDK (librealsense) that
demonstrates how to:

1. Detect a connected RealSense device.
2. Stream depth + colour (RGB) frames at the same resolution / FPS.
3. Optionally stream the two infrared (IR) channels and the IMU (gyroscope & accelerometer) if the
   selected model supports them (e.g. D435i).
4. Display the images live using OpenCV.
5. Exit cleanly when the user presses the **ESC** or **q** key.

This file is 100 % self-contained – the only runtime dependencies are:

* pyrealsense2  (``pip install pyrealsense2``)
* opencv-python  (``pip install opencv-python``)

No additional helper libraries or ROS runtimes are required.

There is *no* hardware connected inside the execution environment that runs
this script during CI, therefore the *main* clause is guarded so that the
file can be imported without throwing an exception if no camera is present.
When you actually run the script on a machine with a RealSense camera
connected, an OpenCV window will pop up and display the live feed.

Author: OpenAI Codex-CLI helper
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:  # pragma: no cover – only happens if dependency missing
    raise SystemExit(
        "pyrealsense2 is not installed. Install it with 'pip install pyrealsense2'"
    ) from exc


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def colourise_depth(depth_frame: rs.depth_frame) -> cv2.Mat:
    """Converts a depth frame (16-bit, in millimetres) into an 8-bit BGR image.

    The function normalises the depth range to 0-255 and applies the OpenCV
    *JET* colour map so that closer objects appear red and farther objects
    blue.
    """

    depth_image = cv2.cvtColor(
        cv2.convertScaleAbs(depth_frame.get_data(), alpha=0.03), cv2.COLOR_GRAY2BGR
    )
    depth_coloured = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return depth_coloured


def get_first_device(context: rs.context) -> Optional[rs.device]:
    """Return the first RealSense device if any, otherwise *None*."""

    devices = context.query_devices()
    if len(devices) == 0:
        return None
    return devices[0]


# ---------------------------------------------------------------------------
# Main streaming routine
# ---------------------------------------------------------------------------


def run(
    rgb_width: int = 640,
    rgb_height: int = 480,
    fps: int = 30,
    enable_infra: bool = False,
    enable_imu: bool = False,
):
    """Open a pipeline, start streaming, and display frames."""

    ctx = rs.context()
    device = get_first_device(ctx)

    if device is None:
        raise RuntimeError("No RealSense device found. Plug in a camera and try again.")

    print("Found device:", device.get_info(rs.camera_info.name))
    print("  Serial number:", device.get_info(rs.camera_info.serial_number))
    print("  Firmware ver.:", device.get_info(rs.camera_info.firmware_version))

    # Configure pipeline streams
    pipeline = rs.pipeline(ctx)
    config = rs.config()

    # If you have multiple cameras, you may specify the serial number here:
    # config.enable_device(<serial>)

    # Depth and colour should have matching resolution + fps when we plan to
    # perform alignment.
    config.enable_stream(rs.stream.depth, rgb_width, rgb_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)

    if enable_infra:
        # Left and right infrared
        config.enable_stream(rs.stream.infrared, 1, rgb_width, rgb_height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, rgb_width, rgb_height, rs.format.y8, fps)

    if enable_imu:
        # D435i exposes gyro at 400 Hz and accel at 250 Hz (but we can ask for
        # any value <= the max).
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)

    # Apply some recommended depth-postprocessing options to improve quality.
    spatial_filter = rs.spatial_filter()  # edge-preserving smoothing
    temporal_filter = rs.temporal_filter()  # reduces depth noise over time

    align_to = rs.stream.color  # align depth to colour coordinate system
    align = rs.align(align_to)

    # Start streaming
    print("Starting pipeline …")
    profile = pipeline.start(config)

    print("Camera intrinsics (colour stream):")
    colour_intr: rs.video_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colour_intr.get_intrinsics()
    print(f"  Resolution    : {intr.width} × {intr.height}")
    print(f"  Focal length  : fx={intr.fx:.1f}  fy={intr.fy:.1f}")
    print(f"  Principal pt. : cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    # Main loop ----------------------------------------------------------------
    last_time = time.perf_counter()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Align depth to colour so that pixel (u,v) matches
            aligned_frames = align.process(frames)

            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            colour_frame: rs.video_frame = aligned_frames.get_color_frame()

            if not depth_frame or not colour_frame:
                # Should rarely happen, but continue gracefully.
                continue

            # Post-process depth
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)

            # Convert RealSense frames to numpy arrays
            colour_image = colour_frame.get_data()  # returns a numpy.ndarray in BGR order
            depth_coloured = colourise_depth(depth_frame)

            # Combine side-by-side for display (make sure both are same height)
            combined = cv2.hconcat([colour_image, depth_coloured])

            # FPS counter
            now = time.perf_counter()
            fps_calc = 1.0 / (now - last_time)
            last_time = now
            cv2.putText(
                combined,
                f"FPS: {fps_calc:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RealSense RGB + Depth", combined)

            if enable_infra:
                ir_left = aligned_frames.get_infrared_frame(1)
                ir_right = aligned_frames.get_infrared_frame(2)
                if ir_left and ir_right:
                    ir_left_img = ir_left.get_data()
                    ir_right_img = ir_right.get_data()
                    cv2.imshow("IR-left", ir_left_img)
                    cv2.imshow("IR-right", ir_right_img)

            if enable_imu:
                gyro: rs.motion_frame = frames.first_or_default(rs.stream.gyro)
                accel: rs.motion_frame = frames.first_or_default(rs.stream.accel)
                if gyro and accel:
                    g_data = gyro.as_motion_frame().get_motion_data()
                    a_data = accel.as_motion_frame().get_motion_data()
                    print(
                        f"Gyro [rad/s]: x={g_data.x:+.3f} y={g_data.y:+.3f} z={g_data.z:+.3f} | "
                        f"Accel [m/s²]: x={a_data.x:+.3f} y={a_data.y:+.3f} z={a_data.z:+.3f}",
                        end="\r",
                    )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # ESC or q to quit
                break
    finally:
        print("\nStopping pipeline, closing windows …")
        pipeline.stop()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# "python stream_realsense.py" entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple RealSense viewer (colour + depth) written in Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--width", type=int, default=640, help="Width of the RGB/depth stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the RGB/depth stream")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--infra", action="store_true", help="Also display the two IR streams")
    parser.add_argument("--imu", action="store_true", help="Print IMU (gyro + accel) readings")

    args = parser.parse_args()

    try:
        run(
            rgb_width=args.width,
            rgb_height=args.height,
            fps=args.fps,
            enable_infra=args.infra,
            enable_imu=args.imu,
        )
    except RuntimeError as err:
        sys.exit(str(err))
