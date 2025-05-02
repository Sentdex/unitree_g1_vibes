"""Real-time LiDAR-SLAM demo for a Livox MID-360.

Prerequisites
-------------
1. Build & install the Livox-SDK shared library (see ``livox_python.py``).
2. ``pip install -r requirements.txt`` where the file lists
   ``numpy``, ``open3d==0.16.0`` (or newer), and ``kiss-icp``.

Run ::

    python live_slam.py

You should see a live, growing point-cloud map in an Open3D window.
"""

# Preset-controlled real-time SLAM demo for the Livox MID-360.
# Choose INDOOR vs OUTDOOR at the top – parameters below are easy to tweak.

from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any
import os

# ---------------------------------------------------------------------------
# Mount orientation correction – default assumes the MID-360 is mounted
# upside-down on the robot.  Override with the env-var:
#     LIVOX_MOUNT=normal  python live_slam.py
# ---------------------------------------------------------------------------

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("LIVOX_MOUNT must be 'normal' or 'upside_down'")

_R_MOUNT = None
if MOUNT == "upside_down":
    _R_MOUNT = np.diag([1.0, -1.0, -1.0, 1.0])

# ---------------------------------------------------------------------------
# KISS-ICP import logic – cope with package layout changes.
# ---------------------------------------------------------------------------

KissICP = None  # type: ignore

_IMPORT_ERRORS = []
try:  # v1.2+ exposes class under "kiss_icp.pipeline"
    from kiss_icp.pipeline import KissICP  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERRORS.append(e)

if KissICP is None:
    try:  # legacy (<1.0) path
        from kiss_icp.pybind import KissICP  # type: ignore
    except Exception as e:  # pragma: no cover
        _IMPORT_ERRORS.append(e)

if KissICP is None:
    _msgs = " | ".join(str(e) for e in _IMPORT_ERRORS)
    raise SystemExit(
        "Could not import KISS-ICP (tried kiss_icp.pipeline & kiss_icp.pybind).\n"
        "Package is missing or broken.  Install/upgrade with:\n"
        "    pip install --upgrade 'kiss-icp'\n\nDetails: "
        + _msgs
    )

# Try SDK2 first (push-mode).  Fallback to legacy SDK if not present.
try:
    from livox2_python import Livox2 as _Livox
except Exception as e:
    print("[INFO] livox2_python unavailable (", e, ") – falling back to SDK1.")
    from livox_python import Livox as _Livox

# ---------------------------------------------------------------------------
# User-selectable presets (INDOOR / OUTDOOR)
# ---------------------------------------------------------------------------

# Pick the desired preset here or export the environment variable `LIVOX_PRESET`.
PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        # Livox2 pseudo-frame aggregation
        "frame_time": 0.35,      # seconds
        "frame_packets": 200,

        # Map & viz
        "voxel_size": 0.4,       # m
        "max_range": 30.0,       # m
        "downsample_limit": 5_000_000,  # keep up to N pts in viewer

        # ICP tuning
        "min_motion": 0.03,      # m
        "conv_criterion": 5e-5,
        "max_iters": 800,
    },
    "outdoor": {
        "frame_time": 0.20,
        "frame_packets": 120,
        "voxel_size": 1.0,
        "max_range": 120.0,
        "downsample_limit": 3_000_000,
        "min_motion": 0.10,
        "conv_criterion": 1e-4,
        "max_iters": 500,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"Unknown PRESET '{PRESET}'. Choose one of {_PRESETS.keys()}.")

# Short alias to the active dictionary so later code is concise
_P = _PRESETS[PRESET]


# ---------------------------------------------------------------------------
# Visualisation utilities
# ---------------------------------------------------------------------------


class _Viewer:
    """Open3D visualiser that shows both the map *and* the current pose."""

    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None

        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None

        self._first = True

    # ------------------------------------------------------------------
    # Thread-safe queues (very small – only last item matters)
    # ------------------------------------------------------------------

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        """Called from background thread with new map + pose."""

        self._latest_pts = xyz
        self._latest_pose = pose

    # ------------------------------------------------------------------
    # Called from the *main/UI* thread
    # ------------------------------------------------------------------

    def tick(self) -> bool:
        updated = False

        if self._latest_pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(self._latest_pts)
            self._vis.update_geometry(self._pcd)
            self._latest_pts = None
            updated = True

        if self._latest_pose is not None:
            self._update_pose_vis(self._latest_pose)
            self._latest_pose = None
            updated = True

        if self._first and updated:
            self._vis.reset_view_point(True)  # auto-fit once we have data
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    # ------------------------------------------------------------------
    def _update_pose_vis(self, pose: np.ndarray):
        # Remove old geometry (if any)
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)

        # Derive a reasonable size from current map extent so the frame is
        # always visible regardless of room size.
        size = 0.5
        if len(self._pcd.points) > 0:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03  # 3 % of diagonal
            size = max(0.2, min(size, 2.0))  # clamp to [0.2 m, 2 m]

        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)
        self._vis.update_geometry(self._cam_frame)

    # ------------------------------------------------------------------
    def close(self):
        self._vis.destroy_window()


# ---------------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------------


class LiveSLAMDemo(_Livox):
    def __init__(self):
        # For SDK2 we need a config path; for SDK1 not.
        # ------------------------------------------------------------------
        # Construct underlying Livox driver with preset aggregation settings
        # ------------------------------------------------------------------

        _sdk_kwargs = {}

        # Livox-SDK **2** wrapper supports frame_time/packets arguments
        if _Livox.__name__ == "Livox2":  # type: ignore[attr-defined]
            _sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])

        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.222", **_sdk_kwargs)  # type: ignore[arg-type]
        except TypeError:
            # legacy SDK1 signature (no args or fewer kwargs)
            super().__init__()

        # Use preset tuned for Livox FOV / scan pattern.
        # Build a default configuration for KISS-ICP (API ≥ 1.2)
        try:
            from kiss_icp.config import load_config  # type: ignore

            cfg = load_config(config_file=None, max_range=_P["max_range"])
        except Exception as e:  # pragma: no cover – extremely old wheels
            print("[KISS-ICP] Could not create config via load_config:", e)
            raise SystemExit(
                "Your installed kiss-icp wheel is too old – please upgrade: `pip install -U kiss-icp`. "
            ) from e

        # Apply preset specific tuning – works with both old & new cfg layouts
        try:
            cfg.mapping.voxel_size = _P["voxel_size"]
            cfg.mapping.max_points_per_voxel = 30
        except AttributeError:
            pass

        cfg.adaptive_threshold.min_motion_th = _P["min_motion"]
        cfg.registration.convergence_criterion = _P["conv_criterion"]
        cfg.registration.max_num_iterations = _P["max_iters"]

        self._slam = KissICP(cfg)
        self._viewer = _Viewer()

        # Down-sample threshold for visualisation
        self._vis_max_points = _P["downsample_limit"]

    # ------------------------------------------------------------------
    # Overridden callback – receives each raw frame
    # ------------------------------------------------------------------

    def handle_points(self, xyz: np.ndarray):  # noqa: D401
        """Process one LiDAR frame.

        Besides forwarding the frame to KISS-ICP we apply a *very* small
        pre-filter that discards returns coming from the robot itself – more
        specifically reflections from the G-1’s head that sit roughly at the
        same height as the MID-360.  Those points are extremely close to the
        sensor and can deteriorate both the SLAM solution and any downstream
        occupancy grid.  We therefore remove all returns that

        1. lie within a narrow vertical band around the LiDAR plane, **and**
        2. are closer than ≈ 3 inches (8 cm) in the horizontal plane.

        The numbers are intentionally conservative so legitimate nearby
        obstacles (for instance an actual wall that is <10 cm away) are still
        kept.  If necessary they can be tuned via the two environment
        variables shown below.

        Environment variables
        --------------------
        LIDAR_SELF_FILTER_RADIUS   – horizontal exclusion radius in metres
                                     (default: 0.08 ≈ 3 in)
        LIDAR_SELF_FILTER_Z        – half height of the vertical dead-band in
                                     metres (default: 0.05 ≈ 2 in)
        """

        import os
        import numpy as _np

        # ------------------------------------------------------------------
        # 1.  Remove reflections from the robot itself (head / mounting)
        # ------------------------------------------------------------------

        # Defaults intentionally generous – covers ≈12-inch radius around the
        # sensor and ±24 cm in height which should safely encompass the G-1’s
        # head even when the robot moves or tilts significantly.
        try:
            r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.30))
            dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.24))
        except ValueError:
            r_xy, dz = 0.08, 0.05  # fallback to sane defaults

        if xyz.size > 0:
            # Horizontal distance from sensor centreline (x-y plane)
            dist_xy = _np.linalg.norm(xyz[:, :2], axis=1)
            close = dist_xy < r_xy

            # Vertical proximity to the LiDAR plane (z ≈ 0 in sensor coords)
            near_plane = _np.abs(xyz[:, 2]) < dz

            mask = ~(close & near_plane)

            # Only allocate new array if we actually filtered anything to
            # keep the common path (no filtering necessary) fast.
            if mask.sum() != xyz.shape[0]:
                xyz = xyz[mask]

        # KISS-ICP ≥1.2 expects (points, timestamps).  We don't have per-point
        # timestamps readily available, so pass a zeros array of shape (N,).
        try:
            # Older kiss-icp (<1.3): expects only points
            self._slam.register_frame(xyz)
        except TypeError:
            # Newer kiss-icp (>=1.3): expects (points, twist) where twist is a
            # per-point 6-vector [ω, v].  Pass a zero-twist to indicate "no
            # motion" so that Sophus::SO3::exp() does not assert.
            import numpy as _np

            # Provide synthetic per-point timestamps spanning one scan period
            period = 1.0 / 20.0  # Mid-360 default 20 Hz
            ts = _np.linspace(0.0, period, num=xyz.shape[0], dtype=_np.float64)
            self._slam.register_frame(xyz, ts)
        try:
            cloud = self._slam.get_map()
        except AttributeError:
            # Newer kiss-icp exposes VoxelHashMap via .local_map
            cloud = self._slam.local_map.point_cloud()
        # Apply mount orientation correction for visualisation & ICP pose.
        if _R_MOUNT is not None:
            cloud = cloud * np.array([1.0, -1.0, -1.0], dtype=cloud.dtype)

        if cloud.shape[0] > self._vis_max_points:
            step = int(cloud.shape[0] / self._vis_max_points) + 1
            cloud = cloud[::step]

        # Current pose (4×4 matrix) – copy to avoid threading issues
        pose = self._slam.last_pose.copy()  # type: ignore[attr-defined]
        if _R_MOUNT is not None:
            pose = _R_MOUNT @ pose

        self._viewer.push(cloud, pose)

    # ------------------------------------------------------------------

    def shutdown(self):
        super().shutdown()
        self._viewer.close()


def main():  # pragma: no cover
    demo = LiveSLAMDemo()

    # Allow Ctrl-C
    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()
