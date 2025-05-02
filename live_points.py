"""Live point-cloud viewer for a Livox MID-360 (or any Livox unit).

This script is intentionally *simpler* than :pymod:`live_slam`.  It just shows
the raw point-cloud streaming off the sensor – no motion estimation, no SLAM.

Running this first is a great way to verify that

1.  the Livox SDK shared library is found, and
2.  datagrams really make it from the LiDAR to your PC.

Prerequisites
-------------
Exactly the same as for :pymod:`live_slam`:

* a working Livox-SDK **or** Livox-SDK2 build installed system-wide so that the
  shared library (``liblivox_sdk.so`` / ``liblivox_lidar_sdk.so``) is on the
  library search path;
* Python packages from ``requirements.txt`` (namely *numpy* and *open3d*).

Usage ::

    python live_points.py

Point-cloud frames are visualised with *Open3D* in real-time.  Hit *Esc* or
press *Ctrl-C* in the terminal to quit.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Mount orientation: 'normal' or 'upside_down'.  Most G1 units have the MID-360
# mounted upside-down.  Therefore the *default* is now 'upside_down'.  If your
# sensor is right-side-up set the env-var `LIVOX_MOUNT=normal`.
# ---------------------------------------------------------------------------

import os

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("LIVOX_MOUNT must be 'normal' or 'upside_down'")

import numpy as np  # after env check – avoids unused import earlier
import open3d as o3d

# ---------------------------------------------------------------------------
# Dynamic import of the right SDK wrapper (SDK2 preferred, SDK1 fallback)
# ---------------------------------------------------------------------------

try:
    from livox2_python import Livox2 as _Livox

    _SDK2 = True
except Exception as _e:  # pragma: no cover – SDK2 not present / not built
    print("[INFO] livox2_python unavailable (", _e, ") – falling back to SDK1.")
    from livox_python import Livox as _Livox

    _SDK2 = False


# ---------------------------------------------------------------------------
# Minimal single-thread visualiser (same as live_slam.py)
# ---------------------------------------------------------------------------


class _Viewer:
    """Open3D visualiser whose *tick* method we drive from the main thread."""

    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox – live point-cloud", width=1280, height=720)

        # We keep a *ring-buffer* of the most recent N frames and visualise
        # the *union*.  This removes the distracting “blinking” that occurs
        # when each new sparse MID-360 scan entirely replaces the previous
        # one.
        self._frames: list[np.ndarray] = []
        self._max_frames = 15  # ≈0.75 s @ 20 Hz – tune to taste

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        # Static coordinate frame at the LiDAR origin so you always know where
        # “the robot” (sensor) is and which way its local XYZ axes point.
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        if MOUNT == "upside_down":
            R180 = np.diag([1.0, -1.0, -1.0, 1.0])
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)

        self._first = True

    # Called from any SDK (background) thread
    def push(self, xyz: np.ndarray):
        self._frames.append(xyz)
        # drop oldest frame when buffer full
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    # Called from the *main* thread
    def tick(self) -> bool:
        if self._frames:
            merged = np.concatenate(self._frames, axis=0)
            self._pcd.points = o3d.utility.Vector3dVector(merged)
            self._vis.update_geometry(self._pcd)
            if self._first:
                self._vis.reset_view_point(True)  # fit camera once
                self._first = False
            self._latest = None

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self):
        self._vis.destroy_window()


# ---------------------------------------------------------------------------
# LiDAR wrapper subclass that forwards every frame to the viewer
# ---------------------------------------------------------------------------


class LiveViewer(_Livox):
    """Thin proxy between SDK callback and the :class:`_Viewer`."""

    def __init__(self):
        # SDK2 requires a JSON config path; SDK1 does not.  Try the new API
        # first and gracefully fall back if the signature does not match.
        if _SDK2:
            super().__init__("mid360_config.json", host_ip="192.168.123.222")  # type: ignore[arg-type]
        else:
            super().__init__()  # SDK1 has no required arguments

        self._view = _Viewer()

    # ------------------------------------------------------------------
    # Callback from SDK base-class – runs in *background* thread(s)
    # ------------------------------------------------------------------

    def handle_points(self, xyz: np.ndarray):  # noqa: D401 (imperative mood)
        # Apply mount orientation correction if needed
        if MOUNT == "upside_down":
            # 180° rotation around the X axis: (x, y, z) -> (x, -y, -z)
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # Down-sample extremely dense frames for smoother rendering – 150 k pts/s
        # is plenty for a preview.
        if xyz.shape[0] > 100_000:
            step = xyz.shape[0] // 100_000
            xyz = xyz[:: step]

        self._view.push(xyz)

    # ------------------------------------------------------------------

    def shutdown(self):
        super().shutdown()
        self._view.close()


# ---------------------------------------------------------------------------
# Main entry-point – standard Ctrl-C handling
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover – manual demo script
    lidar = LiveViewer()

    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and lidar._view.tick():  # type: ignore[attr-defined]
            time.sleep(0.01)
    finally:
        lidar.shutdown()


if __name__ == "__main__":
    main()
