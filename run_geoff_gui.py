"""run_geoff_gui.py – single-window PySide6 GUI

Layout
======
┌─────────────────────────── MainWindow ─────────────────────────────┐
│ ┌─────────────┐  ┌──────────────────────────────────────────────┐ │
│ │   RGB       │  │        3-D SLAM (pyqtgraph.GLViewWidget)     │ │
│ │   640×480   │  │  – rotate / zoom / click‐to-pick planned –   │ │
│ └─────────────┘  │                                              │ │
│ ┌─────────────┐  │                                              │ │
│ │  Depth      │  │                                              │ │
│ │  640×480    │  │                                              │ │
│ └─────────────┘  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘

Keyboard tele-op and the RealSense / Livox workers run unchanged in
background threads (imported from *run_geoff_stack*).  The SLAM point-cloud
is rendered in a **GLViewWidget** so it stays interactive while living
inside the Qt layout.

Requirements
------------
    pip install pyside6 pyqtgraph~=0.13

(pyqtgraph uses *qtpy* and therefore works with PySide6 automatically.)
"""

# noqa: D301
# pylint: disable=attribute-defined-outside-init

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Tuple

# Qt imports must be available at *class* definition time because we now
# derive GeoffWindow from QtCore.QObject so that it can act as a global
# event-filter.

try:
    from PySide6 import QtCore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – missing optional dep.
    raise SystemExit(
        "PySide6 is required for run_geoff_gui.py – install with:\n"
        "    pip install pyside6 pyqtgraph"
    ) from exc

# ------------------------------------------------------------------------
# Re-use the RealSense receiver & tele-op threads from run_geoff_stack
# ------------------------------------------------------------------------

# NOTE: we still import the RealSense receiver and shared-state helpers
#       from *run_geoff_stack* but **do not** start the keyboard thread
#       any more.  Instead we handle key presses directly via Qt so the
#       listener lives in the main GUI thread and works reliably on all
#       platforms / display servers.

from run_geoff_stack import (  # type: ignore
    _rx_realsense,
    _state,
    _state_lock,
)

# ---------------------------------------------------------------------------
# Battery monitor – subscribes to LowState and publishes %SOC in _state.
# ---------------------------------------------------------------------------


def _rx_battery(stop: "threading.Event", iface: str):  # noqa: D401
    """Background worker that keeps the latest battery % in shared _state."""

    try:
        import time
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize  # type: ignore

        # Helper to write SOC into shared state ----------------------------
        def _publish(soc_val: int | None = None, voltage: float | None = None):
            with _state_lock:
                if soc_val is not None:
                    _state["soc"] = soc_val
                if voltage is not None:
                    _state["voltage"] = voltage

        def _attempt_sub(name: str, msg_type, cb):
            try:
                sub = ChannelSubscriber(name, msg_type)
                sub.Init(cb, 50)
                return True
            except Exception:
                return False

        # -----------------------------------------------------------
        # 1) Unitree Go/G1 – LowState
        # -----------------------------------------------------------
        ok = False
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_  # type: ignore

            def _cb_go(msg: LowState_):  # type: ignore[valid-type]
                soc_val = getattr(getattr(msg, 'bms_state', None), 'soc', None)
                if soc_val is not None and soc_val > 0:
                    _publish(int(soc_val))
                else:
                    _publish(voltage=float(msg.power_v))

            ok = _attempt_sub("rt/lowstate", LowState_, _cb_go)
        except Exception:
            ok = False

        # -----------------------------------------------------------
        # 2) Humanoid HG – BmsState topic
        # -----------------------------------------------------------
        if not ok:
            try:
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_  # type: ignore

                def _cb_hg(msg: BmsState_):  # type: ignore[valid-type]
                    _publish(int(msg.soc))

                ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)
            except Exception:
                ok = False

        # -----------------------------------------------------------
        # If both failed, maybe factory not initialised – do that and retry
        # -----------------------------------------------------------
        if not ok:
            try:
                ChannelFactoryInitialize(0, iface)
            except Exception:
                pass  # could still fail if already init failed earlier

            if not ok:
                # retry both subscriptions once more ------------------------------------------------
                ok = _attempt_sub("rt/lowstate", LowState_, _cb_go) if 'LowState_' in locals() else False
                if not ok and 'BmsState_' in locals():
                    ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)

        if not ok:
            raise RuntimeError("Could not subscribe to any battery SOC topic")

        # Idle – callbacks already handle updates
        while not stop.is_set():
            time.sleep(0.5)

    except Exception as exc:  # pylint: disable=broad-except
        import sys

        print("[run_geoff_gui] Battery monitor disabled:", exc, file=sys.stderr)


# ------------------------------------------------------------------------
# Provide a *push-only* viewer for live_slam that just stores the newest map
# in a shared variable.  The Qt thread will visualise it with pyqtgraph.
# ------------------------------------------------------------------------


_slam_latest: Tuple[Any, Any] | None = None  # (xyz ndarray, pose ndarray)
_slam_lock = threading.Lock()


def _patch_live_slam_for_pyqt() -> None:  # noqa: D401
    """Monkey-patch live_slam._Viewer so it no longer opens a GLFW window."""

    import numpy as np  # pylint: disable=import-error

    class _QtViewer:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self._latest_pts: np.ndarray | None = None
            self._latest_pose: np.ndarray | None = None

        # -------- called from SLAM thread --------------------------------
        def push(self, xyz: np.ndarray, pose: np.ndarray):
            global _slam_latest  # noqa: PLW0603
            with _slam_lock:
                _slam_latest = (xyz, pose)

        # -------- tick() signature kept for compatibility ---------------
        def tick(self) -> bool:  # noqa: D401
            # Nothing to do – return True so SLAM main-loop stays alive.
            return True

        def close(self):
            pass

    import live_slam as _ls  # type: ignore

    _ls._Viewer = _QtViewer  # type: ignore[attr-defined]


# ------------------------------------------------------------------------
# SLAM worker – start after patching
# ------------------------------------------------------------------------


def _run_slam(store_evt: threading.Event):  # pragma: no cover – needs HW
    try:
        _patch_live_slam_for_pyqt()

        import live_slam as _ls  # type: ignore

        demo = _ls.LiveSLAMDemo()  # type: ignore[attr-defined]

        while not store_evt.is_set():
            time.sleep(0.05)  # viewer is headless – no need to tick fast

        demo.shutdown()
    except Exception as exc:  # pylint: disable=broad-except
        print("[run_geoff_gui] SLAM thread disabled:", exc, file=sys.stderr)


# ------------------------------------------------------------------------
# Qt GUI ------------------------------------------------------------------
# ------------------------------------------------------------------------


class GeoffWindow(QtCore.QObject):  # type: ignore[misc]  # pylint: disable=too-few-public-methods
    def __init__(self, iface: str, ground_clear_in: float):
        super().__init__()

        from PySide6 import QtWidgets, QtGui  # type: ignore

        # Clearance (in metres) above detected ground before a point is treated
        # as an obstacle.
        self._clear_m = ground_clear_in * 0.0254  # inch → metres
        import pyqtgraph.opengl as gl  # type: ignore

        self.app = QtWidgets.QApplication(sys.argv)

        # ---------------- main widgets ----------------------------------
        self.rgb_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.depth_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        self.rgb_lbl.setMinimumSize(640, 320)
        self.depth_lbl.setMinimumSize(640, 320)

        # 2-D occupancy preview  -------------------------------------------------
        # Replace the static QLabel with a pyqtgraph ImageItem inside an
        # interactive ViewBox so users can freely zoom / pan the bird-eye map.

        import pyqtgraph as pg  # type: ignore

        self.map_view = pg.GraphicsLayoutWidget()  # acts like a regular QWidget
        self.map_view.setMinimumSize(640, 320)

        # Use a dedicated ViewBox so we can lock the aspect-ratio while still
        # allowing mouse interaction (wheel = zoom, drag = pan).
        self._map_vb = self.map_view.addViewBox(lockAspect=True, enableMouse=True)
        self._map_vb.setMenuEnabled(False)
        self._map_vb.invertY(True)  # match conventional image coordinates

        # The ImageItem will be updated every frame with the freshly rendered
        # occupancy canvas produced by _update_2d_map().
        self._map_img = pg.ImageItem()
        self._map_vb.addItem(self._map_img)

        # ------------------------------------------------------------------
        #  Route-planning state
        # ------------------------------------------------------------------

        # Latest binary occupancy (True = obstacle) in image coordinates.
        self._occ_map: "np.ndarray | None" = None  # type: ignore[name-defined]

        # Metadata (min_x, min_y, scale) that maps between world ↔ image px.
        self._map_meta: tuple[float, float, float] | None = None

        # Last planned path as list of pixel-positions (x, y) – image coords.
        self._path_px: list[tuple[int, int]] | None = None

        # Forward mouse clicks on the scene graph to our handler so users can
        # pick a navigation target directly on the 2-D map.  The signal is
        # emitted for *all* clicks inside the GraphicsView, therefore we
        # convert into ViewBox coordinates and ignore positions outside the
        # valid 0 … 479 range.
        self.map_view.scene().sigMouseClicked.connect(self._on_map_click)

        # GL viewer for point-cloud
        self.gl_view = gl.GLViewWidget()
        # Start a bit further back so the full map fits in view.
        self.gl_view.opts["distance"] = 30
        self.gl_view.setCameraPosition(distance=30, elevation=20, azimuth=45)
        # Ensure the GL pane starts with a reasonable width so users don’t
        # have to manually resize the splitter on every launch.
        self.gl_view.setMinimumWidth(640)

        # scatter item – updated incrementally
        self._scatter = gl.GLScatterPlotItem()
        self.gl_view.addItem(self._scatter)

        # list that currently holds the 3 coloured axis lines representing
        # the robot pose.  We remove & rebuild them whenever a new pose comes
        # in from the SLAM thread.
        self._pose_items: list[gl.GLLinePlotItem] = []

        # -------- layout -----------------------------------------------
        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(left)
        v.addWidget(self.rgb_lbl)
        v.addWidget(self.depth_lbl)
        v.addWidget(self.map_view)
        splitter.addWidget(left)
        splitter.addWidget(self.gl_view)
        splitter.setStretchFactor(1, 2)
        # Initialise splitter sizes (left, right)
        splitter.setSizes([640, 640])

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Geoff-Stack")
        self.win.setCentralWidget(splitter)

        # Give the main window an initial size that comfortably shows both
        # the RGB/Depth stack (640 px) and the 3-D view (another 640 px).
        self.win.resize(1600, 760)

        self.status = QtWidgets.QLabel()
        self.win.statusBar().addWidget(self.status)

        # ---------------- timers ---------------------------------------
        self._refresh = QtCore.QTimer()
        self._refresh.setInterval(30)  # ms
        self._refresh.timeout.connect(self._on_tick)
        self._refresh.start()


        # ------------------------------------------------------------------
        #  Tele-operation state (Qt native handling) ------------------------
        # ------------------------------------------------------------------

        self._stop_evt = threading.Event()

        # pressed key set keeps Qt.Key enums / lower-case chars
        self._pressed: set[object] = set()

        # current target velocities that will be sent to the robot
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

        # Track current balance mode (0 – static stand, 1 – continuous gait).
        # We initialise it to -1 so the first call always sets an explicit
        # mode once we know whether the user is commanding motion.
        self._bal_mode: int = -1

        # try to boot the Unitree G-1 so we can actually drive – failure is
        # caught so the GUI still runs on machines that only want to watch
        # the streams.
        try:
            from hanger_boot_sequence import hanger_boot_sequence  # type: ignore

            self._bot = hanger_boot_sequence(iface=iface)
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Tele-op disabled:", exc, file=sys.stderr)
            self._bot = None

        # timer that updates velocities & sends Move at 10 Hz
        self._drive_timer = QtCore.QTimer()
        self._drive_timer.setInterval(100)  # ms  (10 Hz)
        self._drive_timer.timeout.connect(self._on_drive_tick)
        self._drive_timer.start()

        # Install as global event filter so we receive key events no matter
        # which child widget currently has focus.
        self.app.installEventFilter(self)  # type: ignore[arg-type]

# ---------------- background workers -------------------------------------
# RealSense receiver & SLAM still run in their own background threads.  The
# tele-op logic is now handled *inside* the Qt event loop so we no longer
# need the separate `_keyboard_thread`.

        self._threads = [
            threading.Thread(target=_rx_realsense, args=(self._stop_evt,), daemon=True),
            threading.Thread(target=_run_slam, args=(self._stop_evt,), daemon=True),
            threading.Thread(target=_rx_battery, args=(self._stop_evt, iface), daemon=True),
        ]
        for t in self._threads:
            t.start()

        # Graceful quit
        self.app.aboutToQuit.connect(self._on_quit)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def _numpy_to_qpix(self, bgr):
        import numpy as np  # local
        from PySide6 import QtGui  # type: ignore

        if bgr is None or bgr.dtype != np.uint8:
            return None
        h, w, _ = bgr.shape
        qimg = QtGui.QImage(bgr.data.tobytes(), w, h, 3 * w, QtGui.QImage.Format_BGR888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    # ------------------------------------------------------------------
    def _on_tick(self):
        import numpy as np  # type: ignore

        with _state_lock:
            rgbd = _state.get("rgbd")
            vx, vy, om = _state.get("vel", (0.0, 0.0, 0.0))
            soc = _state.get("soc")

        if rgbd is not None and rgbd.shape == (480, 1280, 3):
            rgb, depth = rgbd[:, :640], rgbd[:, 640:]
            px1, px2 = self._numpy_to_qpix(rgb), self._numpy_to_qpix(depth)
            if px1:
                self.rgb_lbl.setPixmap(px1)
            if px2:
                self.depth_lbl.setPixmap(px2)

        status_txt = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {om:+.2f}"
        if soc is not None:
            status_txt += f"   battery {soc:3d}%"
        else:
            with _state_lock:
                volt = _state.get("voltage")
            if volt is not None:
                status_txt += f"   V {volt:5.1f}"
        self.status.setText(status_txt)

        # ----------- point-cloud update --------------------------------
        with _slam_lock:
            data = _slam_latest
        if data is None:
            return

        xyz, pose = data
        if xyz.shape[0] == 0:
            return

        # Down-sample for UI speed
        if xyz.shape[0] > 200_000:
            xyz = xyz[:: int(xyz.shape[0] / 200_000) + 1]

        # -------- continuous gradient with emphasised landmarks ----------
        # 1) height in feet (relative to current minimum so ground = 0)
        z_ft = xyz[:, 2] * 3.28084
        z_rel = z_ft - z_ft.min()

        # 2) normalise into 0-1 over a slightly wider 0–9 ft band so reds
        #    appear only on very high ceilings; tweak _SPAN_FT to taste.
        _SPAN_FT = 9.0
        v = np.clip(z_rel / _SPAN_FT, 0.0, 1.0)

        # 3) gamma (<1) – higher value => softer gradient
        _GAMMA = 0.35
        v_gamma = v ** _GAMMA

        # 4) map to colour – use the perceptually uniform "turbo" colormap
        #    shipped with pyqtgraph (falls back to simple HSV if not found).
        try:
            import pyqtgraph as pg  # type: ignore

            cmap = pg.colormap.get("turbo")  # type: ignore[attr-defined]
            colors = cmap.map(v_gamma, mode="float")  # returns Nx4 float
        except Exception:  # pragma: no cover – minimal fallback
            # Fallback to HSV rainbow like before
            h = 0.66 * (1.0 - v_gamma)
            s = np.ones_like(h)
            val = np.ones_like(h)

            i = np.floor(h * 6).astype(int)
            f = h * 6 - i
            p = val * (1 - s)
            q = val * (1 - f * s)
            t = val * (1 - (1 - f) * s)

            r = np.choose(i % 6, [val, q, p, p, t, val])
            g = np.choose(i % 6, [t, val, val, q, p, p])
            b = np.choose(i % 6, [p, p, t, val, val, q])
            colors = np.stack([r, g, b, np.ones_like(r)], axis=1)

        self._scatter.setData(pos=xyz, size=1.0, color=colors)

        # ---------------- 2-D occupancy map -----------------------------
        self._update_2d_map(xyz, pose)

        # ---------------- pose visualisation ---------------------------
        if pose is not None and pose.shape == (4, 4):
            self._update_pose_axes(pose, xyz)

    # ------------------------------------------------------------------
    # Qt native keyboard handling --------------------------------------
    # ------------------------------------------------------------------

    # helper constants identical to keyboard_controller.py
    _LIN_STEP = 0.05
    _ANG_STEP = 0.2

    # Speed caps (m/s) depending on whether the user keeps <Shift> pressed.
    _SPEED_LIMIT_NORMAL = 0.6
    _SPEED_LIMIT_FAST = 1.2

    def _current_speed_limit(self) -> float:  # noqa: D401 – simple getter
        """Return the velocity clamp to use for the current key state."""

        return (
            self._SPEED_LIMIT_FAST
            if self._is_pressed("shift")
            else self._SPEED_LIMIT_NORMAL
        )

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        """Clamp *val* into ±*limit* range."""

        return max(-limit, min(limit, val))

    # Qt calls this for *all* events once we installed the object as filter
    def eventFilter(self, _obj, ev):  # type: ignore[override]
        from PySide6 import QtCore  # local import to avoid stub issues

        if ev.type() == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return False  # let the default handler run

            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                self._pressed.add(name)
                return True  # handled

        elif ev.type() == QtCore.QEvent.KeyRelease:
            if ev.isAutoRepeat():
                return False

            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                self._pressed.discard(name)
                return True

        return False  # other events continue normal processing

    # ------------------------------------------------------------------
    @staticmethod
    def _qt_key_name(key: int, text: str | None) -> str | None:
        """Map Qt key code → our canonical names (w,a,s,space,…)."""
        from PySide6 import QtCore  # local

        mapping = {
            QtCore.Qt.Key_Space: "space",
            QtCore.Qt.Key_Escape: "esc",
            QtCore.Qt.Key_Z: "z",
            QtCore.Qt.Key_Shift: "shift",
        }

        if key in mapping:
            return mapping[key]

        if text:
            ch = text.lower()
            if ch in ("w", "a", "s", "d", "q", "e"):
                return ch
        return None

    # ------------------------------------------------------------------
    def _is_pressed(self, name: str) -> bool:
        return name in self._pressed

    # ------------------------------------------------------------------
    def _on_drive_tick(self):  # noqa: D401
        # Update target velocities based on current pressed keys.

        lim = self._current_speed_limit()

        if self._is_pressed("w") and not self._is_pressed("s"):
            self._vx = self._clamp(self._vx + self._LIN_STEP, lim)
        elif self._is_pressed("s") and not self._is_pressed("w"):
            self._vx = self._clamp(self._vx - self._LIN_STEP, lim)
        else:
            self._vx = 0.0

        if self._is_pressed("q") and not self._is_pressed("e"):
            self._vy = self._clamp(self._vy + self._LIN_STEP, lim)
        elif self._is_pressed("e") and not self._is_pressed("q"):
            self._vy = self._clamp(self._vy - self._LIN_STEP, lim)
        else:
            self._vy = 0.0

        if self._is_pressed("a") and not self._is_pressed("d"):
            self._omega = self._clamp(self._omega + self._ANG_STEP, lim)
        elif self._is_pressed("d") and not self._is_pressed("a"):
            self._omega = self._clamp(self._omega - self._ANG_STEP, lim)
        else:
            self._omega = 0.0

        # Space bar forces full stop
        if self._is_pressed("space"):
            self._vx = self._vy = self._omega = 0.0

        # Exit keys ----------------------------------------------------
        if self._is_pressed("z"):
            if self._bot is not None:
                try:
                    self._bot.Damp()
                except Exception:
                    pass
            self.app.quit()
            return

        if self._is_pressed("esc"):
            if self._bot is not None:
                try:
                    self._bot.StopMove()
                    self._bot.ZeroTorque()
                except Exception:
                    pass
            self.app.quit()
            return

        # Send command every tick (10 Hz)
        if self._bot is not None:
            try:
                self._bot.Move(self._vx, self._vy, self._omega, continous_move=True)  # type: ignore[arg-type]

                # Keep the robot in static balance mode when no motion is
                # commanded and switch to continuous gait when the operator
                # requests movement.  This avoids the "walking in place"
                # behaviour sometimes observed when the controller remains in
                # mode-1 even though target velocity is zero.
                desired_mode = 0 if (self._vx == self._vy == self._omega == 0.0) else 1
                if desired_mode != self._bal_mode:
                    try:
                        self._bot.SetBalanceMode(desired_mode)
                        self._bal_mode = desired_mode
                    except Exception:
                        pass
            except Exception as exc:
                print("[run_geoff_gui] Move failed:", exc, file=sys.stderr)
                self._bot = None  # disable further attempts

        # publish for HUD ------------------------------------------------
        with _state_lock:
            _state["vel"] = (self._vx, self._vy, self._omega)

    # ------------------------------------------------------------------
    #  Map click → route planning
    # ------------------------------------------------------------------

    def _on_map_click(self, ev):  # noqa: D401
        """Handle mouse clicks on the 2-D occupancy map.

        The GraphicsScene forwards all mouse events; we convert the scene
        position into *view* coordinates (matching our image pixels after the
        ViewBox transform) and start a path-planning run from the current
        robot location to the clicked goal if both are inside the map.
        """

        import numpy as np  # local import

        if self._occ_map is None or self._map_meta is None:
            return  # map not ready yet

        # Convert click → image pixel
        pos = ev.scenePos()
        # Map into ViewBox coordinates (float)
        view_pt = self._map_vb.mapSceneToView(pos)  # type: ignore[attr-defined]
        gx, gy = int(view_pt.x()), int(view_pt.y())

        if not (0 <= gx < 480 and 0 <= gy < 480):
            return  # outside canvas

        # Locate current robot pixel (rx, ry) from last stored pose meta.
        rob_px = getattr(self, "_robot_px", None)
        if rob_px is None:
            return  # cannot plan without robot position

        rx, ry = rob_px

        # Trigger only on *double* left-click so regular single clicks are
        # reserved for panning / zooming inside the ViewBox and do not start
        # an expensive A* search every time the user merely selects or drags
        # the map.

        if not getattr(ev, "double", lambda: False)():  # pyqtgraph helper
            return

        # Plan path (returns list of (x,y) incl. start+goal) ---------------
        path = self._plan_path(rx, ry, gx, gy, self._occ_map)

        if path is not None and len(path) > 1:
            self._path_px = path
        else:
            print("[run_geoff_gui] No path found to clicked target.")

        # Trigger immediate refresh so user sees result without waiting for
        # the next timer tick – safe because we are inside Qt thread.
        self._on_tick()

    # ------------------------------------------------------------------
    @staticmethod
    def _plan_path(sx: int, sy: int, gx: int, gy: int, occ: "np.ndarray") -> list[tuple[int, int]] | None:  # type: ignore[name-defined]
        """A* search on 2-D occupancy grid favouring wide clearance.

        occ – boolean array, True for obstacle, shape (H, W).
        Coordinates in *image* convention (x right, y down).
        Returns list [(x0,y0), …, (xn,yn)] or None if unreachable.
        """

        import heapq
        import math
        import numpy as np  # local import
        import cv2  # type: ignore

        h, w = occ.shape

        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return None

        if occ[sy, sx] or occ[gy, gx]:
            return None  # start or goal blocked

        # Pre-compute distance transform (pixels → nearest obstacle)
        free_uint8 = (~occ).astype(np.uint8)  # 1 = free
        dist = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) or 1.0

        # Weight that biases the search towards the map centre away from
        # obstacles.  Larger => stronger preference for clearance.
        _BIAS = 3.0

        def cell_cost(x: int, y: int) -> float:
            d_norm = dist[y, x] / max_dist  # 0 … 1
            # Lower cost when d_norm high (far from obstacle)
            return 1.0 + _BIAS * (1.0 - d_norm)

        # A* search ----------------------------------------------------
        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, (sx, sy)))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = { (sx, sy): 0.0 }

        def heuristic(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)

        while open_set:
            _, current = heapq.heappop(open_set)
            cx, cy = current

            if current == (gx, gy):
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # explore neighbours (8-connected)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if occ[ny, nx]:
                        continue

                    step = math.hypot(dx, dy) * cell_cost(nx, ny)
                    tentative = g_score[current] + step

                    if tentative < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative
                        f = tentative + heuristic(nx, ny)
                        heapq.heappush(open_set, (f, (nx, ny)))

        return None  # unreachable

    # ------------------------------------------------------------------
    def _on_quit(self):  # noqa: D401
        self._stop_evt.set()
        self._drive_timer.stop()
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Pose axes helper --------------------------------------------------
    # ------------------------------------------------------------------

    def _update_pose_axes(self, pose: "np.ndarray", pts: "np.ndarray") -> None:  # type: ignore[name-defined]
        """Render a small RGB coordinate frame at the robot pose."""

        import numpy as np  # local
        import pyqtgraph.opengl as gl  # local reuse

        # Remove any previous frame first
        for item in self._pose_items:
            self.gl_view.removeItem(item)
        self._pose_items.clear()

        # Derive a reasonable axis length from the map size
        size = 0.5
        if pts.shape[0] > 0:
            span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            size = max(0.2, min(span * 0.03, 2.0))  # 3 % diag, clamp

        origin = pose[:3, 3]
        rot = pose[:3, :3]

        axes = {
            (1.0, 0.0, 0.0, 1.0): rot @ np.array([size, 0, 0]),  # X red
            (0.0, 1.0, 0.0, 1.0): rot @ np.array([0, size, 0]),  # Y green
            (0.0, 0.0, 1.0, 1.0): rot @ np.array([0, 0, size]),  # Z blue
        }

        for color, vec in axes.items():
            pts_arr = np.vstack([origin, origin + vec])
            item = gl.GLLinePlotItem(pos=pts_arr, color=color, width=2, antialias=True)
            self.gl_view.addItem(item)
            self._pose_items.append(item)

    # ------------------------------------------------------------------
    # 2-D occupancy helper ---------------------------------------------
    # ------------------------------------------------------------------

    def _update_2d_map(self, xyz: "np.ndarray", pose: "np.ndarray" | None) -> None:  # type: ignore[name-defined]
        """Derive simple bird-eye occupancy map ignoring ground."""

        import numpy as np  # local
        import cv2  # type: ignore

        if xyz.shape[0] == 0:
            return  # nothing yet

        # Define overall bounds from full cloud to ensure robot always inside
        min_x, max_x = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        min_y, max_y = float(xyz[:, 1].min()), float(xyz[:, 1].max())

        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span  # margin 5 px

        # Store mapping so click-handlers can convert between pixel ↔ world
        # Note: we intentionally map *world y* → horizontal pixel and *world x*
        # → vertical so that "forward" (positive x) appears **upwards** in the
        # occupancy view which matches the intuitive mapping for a top-down
        # map (north/up = forward).
        self._map_meta = (min_x, min_y, scale)

        # Helper closure -------------------------------------------------
        def world_to_px(xw: "np.ndarray", yw: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:  # type: ignore[name-defined]
            """Vectorised conversion world (x, y) → image (px, py)."""

            # Horizontal: +y to the *right* – adjust here if your physical
            # coordinate frame differs.  We apply *no* inversion so positive
            # world-Y appears on the right.  Vertical axis is still flipped
            # so forward (+X) is up.
            px = ((yw - min_y) * scale + 5).astype(np.int32)
            py = ((xw - min_x) * scale + 5).astype(np.int32)
            py = 479 - py  # flip so +x (forward) is up in the image
            return px, py

        canvas = np.full((480, 480, 3), 30, dtype=np.uint8)

        # ------------------------------------------------------------------
        #  Robust ground estimation
        # ------------------------------------------------------------------
        # Using simply *min(z)* is very sensitive to single noisy spikes or
        # the occasional reflection that is slightly closer than the real
        # floor.  That jitter results in the dynamic threshold lifting just
        # enough so genuine floor points punch through and are then shown as
        # obstacles.

        # 1) Robust *instantaneous* estimate – take the 5-th percentile so a
        #    few spurious low readings cannot drag the ground estimate down.
        ground_z_inst = float(np.percentile(xyz[:, 2], 5.0))

        # 2) Exponential smoothing over time – the robot tilts slightly while
        #    walking so the perceived floor distance varies a bit.  Keep a
        #    slowly adapting global value so momentary bumps do not flip
        #    points above / below the clearance threshold every frame.

        _ALPHA = 0.05  # smoothing factor 0 → off, 1 → no smoothing
        if not hasattr(self, "_ground_z_smooth"):
            # First frame – start directly with the instantaneous value.
            self._ground_z_smooth = ground_z_inst  # type: ignore[attr-defined]
        else:
            self._ground_z_smooth = (
                (1.0 - _ALPHA) * self._ground_z_smooth + _ALPHA * ground_z_inst  # type: ignore[attr-defined]
            )

        ground_z = float(self._ground_z_smooth)  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        #  Self-sensor suppression – ignore returns that are almost level with
        #  the LiDAR plane *and* very close to the robot’s centre (mostly the
        #  G-1’s own head / mounting bracket).  The exact same logic already
        #  runs in live_slam.handle_points() for the SLAM front-end but we
        #  repeat it here to also clean up any residual points that might
        #  have slipped through in earlier scans that are still present in
        #  the aggregated local map.
        # ------------------------------------------------------------------

        import os as _os

        try:
            _R_XY = float(_os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.30))
            _DZ = float(_os.environ.get("LIDAR_SELF_FILTER_Z", 0.24))
        except ValueError:
            _R_XY, _DZ = 0.08, 0.05

        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]

            diff = xyz - rob_pos  # broadcast subtraction
            dist_xy = np.linalg.norm(diff[:, :2], axis=1)
            close = dist_xy < _R_XY
            near_plane = np.abs(diff[:, 2]) < _DZ
            keep_mask = ~(close & near_plane)

            if keep_mask.sum() != xyz.shape[0]:
                xyz = xyz[keep_mask]

        # Any point higher than (ground + clearance) is flagged as an obstacle.
        thresh = ground_z + self._clear_m

        # Obstacles above clearance
        pts = xyz[xyz[:, 2] > thresh]

        # Binary occupancy buffer (True = obstacle)
        occ = np.zeros((480, 480), dtype=bool)

        if pts.shape[0] > 0:
            x_obs, y_obs = pts[:, 0], pts[:, 1]
            px_obs, py_obs = world_to_px(x_obs, y_obs)
            valid = (px_obs >= 0) & (px_obs < 480) & (py_obs >= 0) & (py_obs < 480)
            px_obs, py_obs = px_obs[valid], py_obs[valid]

            # Update occupancy grid
            occ[py_obs, px_obs] = True

            # Draw obstacles into canvas for visualisation
            canvas[py_obs, px_obs] = (255, 255, 255)

        cv2.rectangle(canvas, (0, 0), (479, 479), (255, 255, 255), 1)

        # ---------------- robot arrow ---------------------------------
        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            rx, ry = world_to_px(np.array([rob_pos[0]]), np.array([rob_pos[1]]))
            rx, ry = int(rx[0]), int(ry[0])

            # Persist robot pixel so planner knows where to start
            self._robot_px = (rx, ry)

            # Guarantee that the robot’s own cell is considered *free* for
            # planning purposes even if the distance-based obstacle mask
            # (above-ground thresh) flagged it as occupied due to lidar /
            # re-projection noise.  We also clear the 8-neighbourhood so the
            # planner is never trapped at the very first move.

            rr0, rr1 = max(0, ry - 1), min(480, ry + 2)
            rc0, rc1 = max(0, rx - 1), min(480, rx + 2)
            occ[rr0:rr1, rc0:rc1] = False

            # heading angle (yaw) from rotation matrix – robot x-axis
            # Derive endpoint by converting a point straight ahead (robot +
            # 0.25 m along local +x) into pixel coordinates. This avoids any
            # manual trigonometry that would break once we alter the map
            # projection.

            fwd_m = 0.25  # 25 cm arrow length in world space
            # Forward vector in world coords is first column of rotation
            fwd_vec = pose[:3, 0] * fwd_m
            tip_world = rob_pos + fwd_vec
            tx, ty = world_to_px(np.array([tip_world[0]]), np.array([tip_world[1]]))
            tx, ty = int(tx[0]), int(ty[0])

            cv2.arrowedLine(canvas, (rx, ry), (tx, ty), (0, 255, 0), 2, tipLength=0.8)

        # ------------------------------------------------------------------
        #  Overlay planned path (if any)
        # ------------------------------------------------------------------

        if self._path_px and len(self._path_px) > 1:
            cv2.polylines(
                canvas,
                [np.array(self._path_px, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 255),
                thickness=2,
            )

            # Highlight the goal with a solid red dot so it remains visible
            # regardless of zoom level.  The last element in the list is
            # always the clicked goal pixel.
            gx, gy = self._path_px[-1]
            cv2.circle(canvas, (gx, gy), 4, (0, 0, 255), -1)

        # Store occupancy grid for the planner (we copy to avoid aliasing)
        self._occ_map = occ.copy()

        # Update interactive image – pyqtgraph expects the image with the
        # first axis being *y*.  The generated canvas already follows that
        # convention so we can pass it verbatim.

        try:
            self._map_img.setImage(canvas, levels=(0, 255))  # type: ignore[arg-type]
        except Exception:
            # Fallback to non-interactive QLabel if pyqtgraph failed to
            # initialise for some reason (e.g. missing OpenGL on headless
            # test runners).  We keep the old code path as graceful degrade.
            px = self._numpy_to_qpix(canvas)
            if px and hasattr(self, "rgb_lbl"):  # ensure GUI built
                from PySide6 import QtWidgets as _QtW  # local import

                if not hasattr(self, "_legacy_lbl"):
                    self._legacy_lbl = _QtW.QLabel(alignment=QtCore.Qt.AlignCenter)  # type: ignore[attr-defined]
                    self._legacy_lbl.setMinimumSize(640, 320)
                    self._map_vb.hide()
                    # Replace the map_view in the layout – safe because this
                    # code executes only once in the rare fallback path.
                    self.map_view.setParent(None)
                    self.rgb_lbl.parentWidget().layout().addWidget(self._legacy_lbl)

                self._legacy_lbl.setPixmap(px)

    # ------------------------------------------------------------------
    def run(self):  # noqa: D401
        self.win.show()
        sys.exit(self.app.exec())


# ------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="enp68s0f1", help="NIC connected to the Unitree G-1")
    parser.add_argument(
        "--clear",
        type=float,
        default=18.0,
        help="Clearance (in inches) above detected floor before a point is tagged as an obstacle",
    )
    args = parser.parse_args()

    GeoffWindow(args.iface, args.clear).run()


if __name__ == "__main__":
    main()
