#!/usr/bin/env python3.10
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

# ---------------------------------------------------------------------------
#  Logging – capture all console output to a rotating file so each run starts
#  with a fresh log while still preserving a small history of previous
#  sessions (``run_geoff_gui.log.1`` …).  The active log lives in the user's
#  home directory under ``~/.geoff_stack`` so that multiple clones of the
#  repo share a single central log location.
# ---------------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging() -> logging.Logger:  # noqa: D401 – small helper
    """Initialise root logger and return the GUI-specific child logger."""

    # Always start with a *new* logfile so we only capture output from the
    # current run.  When the file grows beyond *maxBytes* it is rotated into
    # ``run_geoff_gui.log.1`` (and older versions are discarded according to
    # *backupCount*).

    # Place the logfile alongside this script so it lives *inside* the project
    # directory and is therefore easy to find when the code base is moved to
    # another machine.  A dedicated sub-directory would keep things tidy, but
    # writing directly next to the script avoids any surprises regarding
    # non-existent folders on read-only setups.

    # Log directory inside the repository so everything stays self-contained.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / "run_geoff_gui.log"

    # File handler – overwrite on every launch, rotate after ~2 MB.
    fh = RotatingFileHandler(
        log_path, mode="w", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )

    # Console handler – keep printing to the *original* stderr so the user
    # still sees messages when launching from a terminal.
    _orig_stderr = sys.stderr
    ch = logging.StreamHandler(_orig_stderr)

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    # ------------------------------------------------------------------
    #  Silence the noisy NumPy ctypeslib PEP3118 warning -----------------
    # ------------------------------------------------------------------
    # Unitree’s Livox C-wrapper returns ctypes structures whose internal
    # *PEP 3118* buffer format strings occasionally mis-advertise the true
    # itemsize.  NumPy therefore emits a RuntimeWarning **every single**
    # time we turn such a struct into an ndarray.  Our stderr→logger bridge
    # upgrades that warning to *ERROR* level which scares users even though
    # the message is harmless.
    #
    # Filter it out globally so the log stays clean while leaving other
    # warnings untouched.
    # ------------------------------------------------------------------
    import warnings  # local late import – only needed here

    warnings.filterwarnings(
        "ignore",
        message=r"A builtin ctypes object gave a PEP3118 format string that does not match its itemsize.*",
        category=RuntimeWarning,
        module=r"numpy\.ctypeslib",
    )

    # Redirect *all* writes to sys.stdout / sys.stderr so that stray prints
    # from third-party libs end up in the log as well (while still appearing
    # in the console via the StreamHandler above).

    class _StreamToLogger:  # pylint: disable=too-few-public-methods
        def __init__(self, level: int):
            self._level = level
            self._logger = logging.getLogger("geoff_gui")

        def write(self, msg: str):  # noqa: D401 – stream interface
            msg = msg.rstrip()
            if msg:
                self._logger.log(self._level, msg)

        def flush(self):  # noqa: D401 – stream interface
            pass

    sys.stdout = _StreamToLogger(logging.INFO)  # type: ignore[assignment]
    sys.stderr = _StreamToLogger(logging.ERROR)  # type: ignore[assignment]

    # Child logger for our own messages – inherits handlers from root.
    return logging.getLogger("geoff_gui")


# Initialise immediately so anything that executes during import is captured
log = _setup_logging()

# Qt imports must be available at *class* definition time because we now
# derive GeoffWindow from QtCore.QObject so that it can act as a global
# event-filter.

try:
    from PySide6 import QtCore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – missing optional dep.
    log.error("PySide6 is required – install via 'pip install pyside6 pyqtgraph'")
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

    # ------------------------------------------------------------------
    #  Safety patch – make LiveSLAMDemo robust against occasional KISS-ICP
    #  initialisation hiccups.  We wrap its handle_points() so that *any*
    #  exception inside the original implementation (for instance due to an
    #  uninitialised pose on the very first scan) is caught and we still
    #  forward the raw xyz to the viewer.  This guarantees that the Qt GUI
    #  always receives something to show and therefore never stays blank.
    # ------------------------------------------------------------------

    try:
        _orig_hp = _ls.LiveSLAMDemo.handle_points  # type: ignore[attr-defined]

        def _safe_hp(self, xyz):  # type: ignore[no-self-use]
            try:
                _orig_hp(self, xyz)  # type: ignore[misc]
            except Exception as exc:  # pylint: disable=broad-except
                # Push raw points without a valid pose.  The GL scatter still
                # renders; the pose axes just remain absent until KISS-ICP
                # recovers.
                try:
                    self._viewer.push(xyz, None)
                except Exception:
                    pass
                print("[run_geoff_gui] KISS-ICP first frame failed:", exc)

        _ls.LiveSLAMDemo.handle_points = _safe_hp  # type: ignore[assignment]
    except Exception:
        pass


# ------------------------------------------------------------------------
# SLAM worker – start after patching
# ------------------------------------------------------------------------


def _run_slam(stop_evt: threading.Event):  # pragma: no cover – needs HW
    """Background worker that runs the Livox SLAM pipeline.

    We let the *driver* block inside its own ``spin()`` method so the SDK
    threads can push point-cloud frames uninterrupted.  As soon as the Qt
    application requests shutdown (``stop_evt`` is set) we gracefully tear
    everything down.
    """

    try:
        _patch_live_slam_for_pyqt()

        import live_slam as _ls  # type: ignore

        demo = _ls.LiveSLAMDemo()  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # Run the SDK loop – either via the provided .spin() helper (SDK2) or
        # a simple sleep-loop fallback when the wrapper doesn’t expose it.
        # ------------------------------------------------------------------

        spin_fn = getattr(demo, "spin", None)

        # Launch the SDK spin-loop (if present) in yet another daemon thread
        # so we can still monitor *stop_evt* and call ``shutdown`` ourselves.
        if callable(spin_fn):
            t_spin = threading.Thread(target=spin_fn, daemon=True)
            t_spin.start()

        try:
            while not stop_evt.is_set():
                # Even though our custom _QtViewer does not *need* its tick()
                # method called (it simply returns True), the original
                # LiveSLAMDemo main-loop expected to be able to perform some
                # periodic house-keeping inside that function.  Calling it
                # here re-establishes full behavioural parity with the
                # upstream script and – crucially – ensures that any side
                # effects future versions add will still run.

                try:
                    demo._viewer.tick()  # type: ignore[attr-defined]
                except Exception:
                    pass

                time.sleep(0.05)
        finally:
            try:
                demo.shutdown()
            except Exception:
                pass

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_geoff_gui] SLAM thread disabled:", exc, file=sys.stderr)


# ------------------------------------------------------------------------
# Qt GUI ------------------------------------------------------------------
# ------------------------------------------------------------------------


class GeoffWindow(QtCore.QObject):  # type: ignore[misc]  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        iface: str,
        ground_clear_in: float,
        *,
        hand: str = "left",
        grip_force: float | None = None,
    ):
        """Create main GUI window.

        Parameters
        ----------
        iface
            Network interface connected to the robot.
        ground_clear_in
            Clearance (in **inches**) above the detected ground plane before a
            point is considered an obstacle (forwarded to the SLAM obstacle
            filter).
        hand
            Which Dex3 hand is physically attached to the robot.  Defaults to
            ``"left"`` to preserve the original behaviour.
        grip_force
            Optional feed-forward torque (approx. **N·m**) applied during the
            continuous *g*rab mode.
        """

        super().__init__()

        from PySide6 import QtWidgets, QtGui  # type: ignore

        # Clearance (in metres) above detected ground before a point is treated
        # as an obstacle.
        self._clear_m = ground_clear_in * 0.0254  # inch → metres

        # Store CLI grip force so the lower section in the constructor can
        # read it even before the specific hand control attributes are
        # initialised.
        self._cli_grip_force = grip_force if grip_force is not None else 0.3
        import pyqtgraph.opengl as gl  # type: ignore

        self.app = QtWidgets.QApplication(sys.argv)

        # ------------------------------------------------------------------
        #  Make Ctrl-C (SIGINT) close the application immediately.
        #  -----------------------------------------------------------------
        import signal

        try:
            signal.signal(signal.SIGINT, lambda *_: self.app.quit())
        except Exception:
            pass

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

        # ------------------------------------------------------------------
        #  User controls – Damp Arms / Center Waist -------------------------
        # ------------------------------------------------------------------

        self._btn_damp = QtWidgets.QPushButton("Damp arms & center waist")
        self._btn_damp.setToolTip("Switch upper body to passive mode and set waist to 0 rad")
        self._btn_damp.clicked.connect(self._on_damp_pressed)  # type: ignore[arg-type]
        self.win.statusBar().addPermanentWidget(self._btn_damp)

        # ------------------------------------------------------------------
        #  Arm selector – Left / Right --------------------------------------
        # ------------------------------------------------------------------

        self._arm_selector = QtWidgets.QComboBox()
        self._arm_selector.addItems(["Left arm", "Right arm"])
        self._arm_selector.setCurrentIndex(0)  # default → left arm
        self._arm_selector.setToolTip("Select which arm to control and run inference on")

        # Keep a readily accessible textual flag so other helpers can query
        # the active side without touching the UI element.
        self._active_arm: str = "left"

        def _on_sel_changed(idx: int):
            self._active_arm = "left" if idx == 0 else "right"
            try:
                self._configure_arm_variables()
            except Exception as exc:
                print("[run_geoff_gui] Reconfigure arm failed:", exc, file=sys.stderr)

        self._arm_selector.currentIndexChanged.connect(_on_sel_changed)  # type: ignore[arg-type]
        self.win.statusBar().addPermanentWidget(self._arm_selector)

        # ------------------------------------------------------------------
        #  Visual feedback for pressed keys ---------------------------------
        # ------------------------------------------------------------------

        # A small semi‐transparent overlay in the top-left corner lists all
        # currently pressed control keys in an easily visible manner so that
        # users receive instant confirmation that their keyboard input has
        # been recognised by the application.

        # Make the overlay a child of the *central widget* so it is guaranteed
        # to paint *above* the normal contents (splitter, GL view, …) even on
        # platforms where a direct child of QMainWindow would otherwise be
        # obscured by the central widget.

        # Overlay lives *inside* the GL viewer so it stays attached to that
        # pane even when users resize the window or fiddle with the splitter.

        self._key_overlay = QtWidgets.QWidget(self.gl_view)
        self._key_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._key_overlay.move(10, 10)

        # Container style provides common translucent background.
        self._key_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150);"
            "border-radius: 6px;"
        )

        _lay = QtWidgets.QVBoxLayout(self._key_overlay)
        _lay.setContentsMargins(8, 6, 8, 6)
        _lay.setSpacing(0)

        # Header – stays fully opaque at all times.
        self._header_lbl = QtWidgets.QLabel("Key inputs:")
        self._header_lbl.setStyleSheet(
            "color: #ffff00; font: 12pt 'Consolas', 'Monaco', 'Courier New', monospace;"
        )
        _lay.addWidget(self._header_lbl)

        # Dynamic key list label.
        self._keys_lbl = QtWidgets.QLabel("–")
        self._keys_lbl.setStyleSheet(
            "color: #ffff00; font: bold 24pt 'Consolas', 'Monaco', 'Courier New', monospace;"
        )
        _lay.addWidget(self._keys_lbl)

        self._key_overlay.adjustSize()
        self._key_overlay.show()

        # Opacity effect & fade animation applied only to keys label.
        self._keys_opacity = QtWidgets.QGraphicsOpacityEffect(self._keys_lbl)
        self._keys_lbl.setGraphicsEffect(self._keys_opacity)

        self._fade_anim = QtCore.QPropertyAnimation(self._keys_opacity, b"opacity", self)
        self._fade_anim.setDuration(600)  # ms

        def _on_fade_finished():
            # Reset for next cycle.
            self._keys_opacity.setOpacity(1.0)
            self._keys_lbl.setText("–")
            self._key_overlay.adjustSize()

        self._fade_anim.finished.connect(_on_fade_finished)  # type: ignore[arg-type]

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

        # ------------------------------------------------------------------
        #  Right arm startup sequence ---------------------------------------
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        #  Arm control (left *or* right)
        # ------------------------------------------------------------------
        #
        # The Unitree SDK exposes a dedicated real-time DDS topic (``rt/arm_sdk``)
        # that accepts LowCmd messages for *both* arms.  Historically the GUI
        # drove only the **right** arm (joint IDs 22 … 28).  With the recent
        # addition of a model for the **left** arm (IDs 15 … 21) we now make
        # the controlled side configurable so that users can switch directly
        # inside the application.
        #
        # A small combo-box in the status-bar selects the active arm at run-time
        # (defaults to *Left arm*).  All subsequent logic – pose sequence,
        # smooth ramp generator, ML-inference, damp-button – references the
        # generic *self._arm_joint_idx* list so that no code path is limited
        # to a hard-coded set of joint indices any longer.

        self._arm_pub = None  # type: ignore[assignment]
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher  # type: ignore
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_  # type: ignore
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # type: ignore
            from unitree_sdk2py.utils.crc import CRC  # type: ignore

            # ---------------- joint indices -----------------------------
            _WAIST_YAW_IDX = 12

            # --------------------------------------------------------------
            #  Per-arm joint index definitions
            # --------------------------------------------------------------

            _LEFT_IDX = {idx: 0 for idx in range(15, 22)}
            _RIGHT_IDX = {idx: 0 for idx in range(22, 29)}

            _ARM_IDX = _LEFT_IDX if self._active_arm == "left" else _RIGHT_IDX

            # Store indices for helpers (e.g. Damp button)
            self._arm_joint_idx: list[int] = list(_ARM_IDX.keys())
            self._waist_idx: int = _WAIST_YAW_IDX

            _NOT_USED_IDX = 29  # enables arm_sdk when q = 1 (per Unitree docs)

            self._crc = CRC()

            # Persistent full message that we update each cycle.
            self._arm_cmd = unitree_hg_msg_dds__LowCmd_()  # type: ignore[attr-defined]
            self._arm_cmd.motor_cmd[_NOT_USED_IDX].q = 1  # type: ignore[index]

            # ------------------------------------------------------------------
            #  Target pose sequence (rad) --------------------------------------
            # ------------------------------------------------------------------

            # Helper: list of (joint_idx, target_q) for each pose.
            if self._active_arm == "right":
                # Two-step start-up sequence (existing behaviour)
                self._pose_seq: list[list[tuple[int, float]]] = [
                    [
                        (_WAIST_YAW_IDX, 0.0),
                        (22, -0.023),  # shoulder pitch
                        (23, -0.225),  # shoulder roll
                        (24, +0.502),  # shoulder yaw
                        (25, +1.317),  # elbow
                        (26, +0.185),  # wrist pitch
                        (27, +0.125),  # wrist roll
                        (28, -0.182),  # wrist yaw
                    ],
                    [
                        (_WAIST_YAW_IDX, 0.0),
                        (22, +0.087),
                        (23, -0.271),
                        (24, +0.323),
                        (25, +0.691),
                        (26, +0.240),
                        (27, -0.771),
                        (28, -0.176),
                    ],
                ]
            else:
                # Single-step initial pose for the left arm (provided values)
                self._pose_seq = [
                    [
                        (_WAIST_YAW_IDX, 0.0),
                        (15, +0.211),  # shoulder pitch
                        (16, +0.181),  # shoulder roll
                        (17, -0.284),  # shoulder yaw
                        (18, +0.672),  # elbow
                        (19, -0.379),  # wrist roll
                        (20, -0.852),  # wrist pitch
                        (21, -0.019),  # wrist yaw
                    ]
                ]

            # ------------------------------------------------------------------
            # Per-joint command state
            # ------------------------------------------------------------------

            # Internal dictionary {idx: commanded_q} to build a smooth ramp.
            self._cmd_q: dict[int, float] = {idx: 0.0 for idx in _ARM_IDX}
            self._cmd_q[_WAIST_YAW_IDX] = 0.0

            # ------------------------------------------------------------------
            #  Live feedback – subscribe to LowState so we know the *current*
            #  joint angles.  Initialising the command dictionary with the
            #  real angles avoids the sudden “snap” that occurs when stiff
            #  position control (high kp) is enabled with targets far away
            #  from the present configuration.
            # ------------------------------------------------------------------

            self._joint_cur: dict[int, float] = {}


            self._ls_sub = None  # type: ignore[attr-defined]

            # Deferred LowState subscription so the GUI never blocks during
            # start-up.  The thread attempts to open the subscriber once and
            # silently gives up if the middleware is unavailable.

            def _init_ls_sub():
                """Subscribe to *rt/lowstate* using whichever Unitree IDL is
                available on the current robot (HG – humanoid, GO – quadruped).

                Older firmware revisions published exactly the same LowState
                message under different namespaces.  We therefore try both
                variants and silently keep going with the first one that
                succeeds so that the GUI continues to work on all platforms
                without user intervention.
                """

                from unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore

                # List of candidate IDL paths to try in order of preference.
                _candidates = [
                    "unitree_sdk2py.idl.unitree_hg.msg.dds_.LowState_",
                    "unitree_sdk2py.idl.unitree_go.msg.dds_.LowState_",
                ]

                for dotted in _candidates:
                    try:
                        mod_path, cls_name = dotted.rsplit(".", 1)
                        mod = __import__(mod_path, fromlist=[cls_name])
                        LowState_ = getattr(mod, cls_name)  # type: ignore[valid-type]

                        def _ls_cb(msg):  # type: ignore[valid-type]
                            # Grab only the joints we actively command so
                            # that the feedback dictionary always carries the
                            # latest measured angle for the selected arm.
                            for j_idx in (*self._arm_joint_idx, _WAIST_YAW_IDX):
                                try:
                                    self._joint_cur[j_idx] = msg.motor_state[j_idx].q  # type: ignore[index]
                                except Exception:
                                    pass

                        sub = ChannelSubscriber("rt/lowstate", LowState_)

                        # Try to init with a small timeout – if the DDS stack
                        # is unavailable this call can otherwise block for
                        # seconds.
                        sub.Init(_ls_cb, 200)

                        # Keep reference so GC doesn’t kill it.
                        self._ls_sub = sub  # type: ignore[attr-defined]
                        return  # success – no need to try further variants
                    except Exception:
                        continue  # try next namespace

                # If all attempts fail simply leave feedback disabled – the
                # rest of the GUI (and the arm sequence) will still work,
                # albeit without the snap-free initialisation based on real
                # joint angles.

            threading.Thread(target=_init_ls_sub, daemon=True).start()

            # Flag so _on_arm_tick can lazily copy the first feedback sample.
            self._initialised_from_state = False

            # Sequence progress trackers.
            self._seq_idx = 0  # which pose are we moving towards (0 / 1)
            self._SEQ_EPS = 0.01  # rad tolerance to consider a joint "reached"
            self._STEP = 0.02      # rad per 20 ms tick (~1.1°) – gives a slow glide

            # DDS publisher
            self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)  # type: ignore[arg-type]
            self._arm_pub.Init()

            # Timer at 50 Hz that applies the ramp.
            self._arm_timer = QtCore.QTimer()
            self._arm_timer.setInterval(20)  # ms
            self._arm_timer.timeout.connect(self._on_arm_tick)
            self._arm_timer.start()

            # --------------------------------------------------------------
            #  Ready *both* arms on start-up -------------------------------
            # --------------------------------------------------------------
            # Historically the GUI only initialised the **selected** arm
            # (default → right) while the other side stayed completely
            # passive.  We now move *both* arms into a comfortable standby
            # posture so that operators can immediately take control of
            # either one without the need for a separate warm-up routine.
            #
            # The active arm follows its regular smooth ramp handled by
            # *_on_arm_tick*.  For the **other** arm we publish a *one-off*
            # LowCmd that sets its joints to the final target pose with
            # moderate stiffness.  Subsequent timer ticks keep these values
            # unchanged because *_on_arm_tick* only touches joints listed in
            # *self._cmd_q* (i.e. the currently selected side).  This means
            # the non-active arm holds position but does not consume any
            # additional bandwidth or CPU time.

            def _apply_pose_once(pose: list[tuple[int, float]]):
                for j_idx, q_val in pose:
                    mc = self._arm_cmd.motor_cmd[j_idx]  # type: ignore[index]
                    mc.q = q_val
                    mc.dq = 0.0
                    mc.tau = 0.0
                    mc.kp = 60.0
                    mc.kd = 1.5

            # Build ready pose for the *other* arm (opposite of _active_arm).
            if self._active_arm == "left":
                _ready_other = [
                    (22, +0.087),  # shoulder pitch
                    (23, -0.271),  # shoulder roll
                    (24, +0.323),  # shoulder yaw
                    (25, +0.691),  # elbow
                    (26, +0.240),  # wrist pitch
                    (27, -0.771),  # wrist roll
                    (28, -0.176),  # wrist yaw
                ]
            else:  # active → right, therefore ready the *left* arm
                _ready_other = [
                    (15, +0.211),  # shoulder pitch
                    (16, +0.181),  # shoulder roll
                    (17, -0.284),  # shoulder yaw
                    (18, +0.672),  # elbow
                    (19, -0.379),  # wrist roll
                    (20, -0.852),  # wrist pitch
                    (21, -0.019),  # wrist yaw
                ]

            _apply_pose_once(_ready_other)

            # Re-calculate CRC and transmit immediately so the robot starts
            # holding the pose before the first 50 Hz timer tick occurs.
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)  # type: ignore[attr-defined]
            try:
                self._arm_pub.Write(self._arm_cmd)  # type: ignore[arg-type]
            except Exception:
                pass

        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Arm control disabled:", exc, file=sys.stderr)

        # ------------------------------------------------------------------
        #  Dex3 right-hand control -----------------------------------------
        # ------------------------------------------------------------------

        self._dex3 = None  # type: ignore[assignment]
        try:
            from unitree_sdk2py.dex3 import Dex3Client  # type: ignore

            # ------------------------------------------------------------------
            #  Attempt to connect to the requested hand.  If the explicit NIC
            #  name fails (common when the local interface differs from the
            #  hard-coded default) fall back to the SDK’s built-in *auto*
            #  detection by retrying with ``interface=None``.  This mirrors the
            #  resilient approach used by *handdev_gui.py* and greatly reduces
            #  the chance of ending up with *self._dex3 = None* which would
            #  silently disable the entire grip logic.
            # ------------------------------------------------------------------

            try:
                self._dex3 = Dex3Client(hand=hand, interface=iface)
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"[run_geoff_gui] Dex3 connection failed on interface '{iface}':",
                    exc,
                    file=sys.stderr,
                )

                try:
                    self._dex3 = Dex3Client(hand=hand, interface=None)
                    print("[run_geoff_gui] Dex3 connected via auto-detected NIC.")
                except Exception as exc2:  # pylint: disable=broad-except
                    print("[run_geoff_gui] Dex3 auto-detect failed:", exc2, file=sys.stderr)
                    self._dex3 = None

            # ------------------------------------------------------------------
            #  Load optional hand pose CSV (open / middle / closed)
            # ------------------------------------------------------------------

            self._hand_poses: dict[str, list[float]] = {}
            try:
                import csv

                csv_path = Path("data/hand_states.csv")
                if csv_path.exists():
                    with csv_path.open("r", newline="") as fp:
                        rdr = csv.DictReader(fp)
                        for row in rdr:
                            label = row.get("label")
                            if label:
                                try:
                                    vals = [float(row[f"joint{i}"]) for i in range(7)]
                                    self._hand_poses[label.lower()] = vals
                                except Exception:
                                    pass
            except Exception as exc:  # pylint: disable=broad-except
                print("[run_geoff_gui] Could not load hand_states.csv:", exc, file=sys.stderr)

            # ------------------------------------------------------------------
            #  Hand motion ramp helpers ----------------------------------------
            # ------------------------------------------------------------------

            self._hand_cmd_q: list[float] = [0.0] * 7
            self._hand_pose_seq: list[list[float]] = []
            self._hand_seq_idx: int = 0
            self._HAND_STEP = 0.1  # rad per tick (≈5.7°)

            self._hand_timer = QtCore.QTimer()
            self._hand_timer.setInterval(20)  # ms – 50 Hz
            self._hand_timer.timeout.connect(self._on_hand_tick)
            self._hand_timer.start()

            # Simplified open/close key poses (requested by user) ----------
            # These override any CSV content when the p / o keys are used.
            self._simple_open_pose = [
                -0.15717165172100067,
                -0.41322529315948486,
                0.02846403606235981,
                0.17782948911190033,
                -0.025226416066288948,
                0.17983606457710266,
                -0.027690349146723747,
            ]

            self._simple_closed_pose = [
                0.07452802360057831,
                0.9478388428688049,
                1.766921877861023,
                -1.4442411661148071,
                -1.4384468793869019,
                -1.5298594236373901,
                -1.4153316020965576,
            ]

            # Current high-level target pose (always 7 elements).
            self._hand_target: list[float] = list(self._simple_open_pose)
            # Adaptive & continuous grasp flags
            # Modes:
            #   idle       – no movement, hold current position
            #   closing    – legacy scripted close via _hand_pose_seq
            #   holding    – adaptive grasp finished, maintain hold
            #   opening    – scripted open sequence active
            #   adaptive   – pressure-based incremental closing (triggered by key *p*)
            #   grabbing   – NEW: continuously attempt to close with constant torque
            self._hand_mode: str = "idle"

            # ------------------------------------------------------------------
            #  Continuous grasp configuration  ----------------------------------
            # ------------------------------------------------------------------

            # Primary joints to close first when a grab is initiated.  These
            # typically correspond to the distal joints of the index, middle
            # and thumb so the fingers wrap around an object before the more
            # proximal joints add additional force.
            self._GRAB_PRIMARY_IDX: list[int] = [1, 4, 6]

            # Feed-forward torque [N·m] applied to *each* closing joint while
            # _hand_mode == 'grabbing'.  Sign is corrected automatically for
            # every individual joint depending on its closing direction.  A
            # modest default keeps the forces reasonable but the value can be
            # changed at runtime via the new ``--grip-force`` CLI flag.
            self._GRAB_TAU: float = getattr(self, "_cli_grip_force", 0.3)

            # Continuous tightening parameters ---------------------------
            # _SEQ_EPS already defined earlier (0.01 rad) and the per-tick
            # _HAND_STEP ramp moves each joint smoothly toward *closed*.

            # Stage flag so we only start moving the secondary joints once the
            # primaries reached their closed target.
            self._grab_stage: int = 0  # 0 → primary first, 1 → all joints

            # Pre-compute joint-wise *closing* direction so we know the correct
            # torque signs.  If required poses are unavailable fall back to +1.
            self._hand_open_pose = self._hand_poses.get("open", [0.0] * 7)
            self._hand_closed_pose = self._hand_poses.get("closed", [0.0] * 7)
            self._close_dir = [
                1.0 if (c - o) >= 0 else -1.0 for o, c in zip(self._hand_open_pose, self._hand_closed_pose)
            ]

            # Adaptive grasp tuning -----------------------------------------
            self._PRESS_TARGET = 0.4  # desired minimum pressure (N?)
            self._PRESS_HYST = 0.05   # hysteresis

            # Logger helper for detailed grip debug
            self._log_hand = logging.getLogger("geoff_gui.hand")
            self._PRESS_THR = 0.5  # contact threshold (approx – tune!)
            self._PRESS_MIN_COUNT = 3  # how many tips must exceed threshold
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Dex3 hand control disabled:", exc, file=sys.stderr)
            self._dex3 = None

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

        # Finalise arm-specific helpers for the initial *left* default.
        try:
            self._configure_arm_variables()
        except Exception as exc:
            print("[run_geoff_gui] Initial arm configuration failed:", exc, file=sys.stderr)

    # ------------------------------------------------------------------
    #  Dynamic arm configuration helper ---------------------------------
    # ------------------------------------------------------------------

    def _configure_arm_variables(self):  # noqa: D401
        """(Re)initialise per-arm state after the user toggled the selector.

        All places that need arm-specific lists query *self._arm_joint_idx*
        so updating that attribute together with the start‐up pose sequence
        is sufficient.  If the configuration is called at run-time we also
        reset the current sequence index so the new arm smoothly moves to
        its initial posture.
        """

        _WAIST_YAW_IDX = 12

        self._arm_joint_idx = list(range(15, 22)) if self._active_arm == "left" else list(range(22, 29))

        # Ensure the command dictionary carries an entry for every controlled
        # joint so that later ramp updates work without KeyError.
        if not hasattr(self, "_cmd_q"):
            self._cmd_q = {}
        for idx in self._arm_joint_idx:
            self._cmd_q.setdefault(idx, 0.0)
        self._cmd_q.setdefault(_WAIST_YAW_IDX, 0.0)

        # Remove any stale joint entries from the previously selected arm so
        # that subsequent LowCmd messages touch *only* the currently active
        # side (plus waist).
        for idx in list(self._cmd_q):
            if idx not in self._arm_joint_idx and idx != _WAIST_YAW_IDX:
                self._cmd_q.pop(idx, None)

        # Build the appropriate start-up pose sequence.
        if self._active_arm == "right":
            # Same two-step sequence as before.
            self._pose_seq = [
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (22, -0.023),
                    (23, -0.225),
                    (24, +0.502),
                    (25, +1.317),
                    (26, +0.185),
                    (27, +0.125),
                    (28, -0.182),
                ],
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (22, +0.087),
                    (23, -0.271),
                    (24, +0.323),
                    (25, +0.691),
                    (26, +0.240),
                    (27, -0.771),
                    (28, -0.176),
                ],
            ]
        else:
            self._pose_seq = [
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (15, +0.211),
                    (16, +0.181),
                    (17, -0.284),
                    (18, +0.672),
                    (19, -0.379),
                    (20, -0.852),
                    (21, -0.019),
                ]
            ]

        # Reset progress so new arm starts moving immediately.
        self._seq_idx = 0

        # Force fresh LowState-based initialisation to avoid a snap when
        # switching arms on a running robot.
        self._initialised_from_state = False

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

        # Always refresh the key overlay so users receive instant feedback.
        self._update_key_overlay()

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

    # ------------------------------------------------------------------
    #  Key-overlay helpers ----------------------------------------------
    # ------------------------------------------------------------------

    _DISPLAY_NAMES = {
        "space": "␣",
        "shift": "⇧",
        "esc": "⎋",
        "up_arrow": "↑",
        "down_arrow": "↓",
        "left_arrow": "←",
        "right_arrow": "→",
    }

    def _display_name(self, key: str) -> str:  # noqa: D401 – tiny helper
        """Return a short user-friendly representation for *key*."""

        if key in self._DISPLAY_NAMES:
            return self._DISPLAY_NAMES[key]
        # Single character like 'w', 'a', … – show upper-case for clarity.
        if len(key) == 1:
            return key.upper()
        return key

    def _update_key_overlay(self) -> None:  # noqa: D401 – helper
        """Refresh the on-screen list of currently pressed keys."""

        # Header is always present so the overlay never disappears – helps
        # newcomers discover the available controls even before they press a
        # key for the first time.

        if self._pressed:
            # If keys are currently pressed make sure any running fade is
            # aborted and the overlay is fully opaque again.
            if self._fade_anim.state() == QtCore.QAbstractAnimation.Running:
                self._fade_anim.stop()
                self._keys_opacity.setOpacity(1.0)

            keys_txt = "  ".join(self._display_name(k) for k in sorted(self._pressed))
            self._keys_lbl.setText(keys_txt)
            self._key_overlay.adjustSize()
            self._key_overlay.move(10, 10)
            self._key_overlay.raise_()

        else:
            # No keys pressed → start fade if not already running and current
            # text is not the placeholder dash.
            if (
                self._fade_anim.state() != QtCore.QAbstractAnimation.Running
                and self._keys_lbl.text() != "–"
            ):
                self._fade_anim.stop()
                self._fade_anim.setStartValue(1.0)
                self._fade_anim.setEndValue(0.0)
                self._fade_anim.start()

    # Qt calls this for *all* events once we installed the object as filter
    def eventFilter(self, _obj, ev):  # type: ignore[override]
        from PySide6 import QtCore  # local import to avoid stub issues

        if ev.type() == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return False  # let the default handler run

            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                    # Store key state so Drive control continues to work for
                    # the original tele-op keys (w/a/s/d/…).  For the newly
                    # added arrow + f/b keys we additionally trigger a single
                    # shot inference-based arm movement.

                    self._pressed.add(name)

                    # ----------------------------------------------------
                    # Arm movement inference trigger
                    # ----------------------------------------------------
                    self._maybe_arm_inference(name)

                    # ----------------------------------------------------
                    # Hand (Dex3) open/close trigger
                    # ----------------------------------------------------
                    self._maybe_hand_control(name)

                    return True  # handled

        elif ev.type() == QtCore.QEvent.KeyRelease:
            if ev.isAutoRepeat():
                return False

            key = ev.key()
            name = self._qt_key_name(key, ev.text())

            if name is None:
                return False

            # Remove key from pressed set for *all* keys so GUI overlay stays
            # accurate.  (The previous special-case for p/o is no longer
            # needed after simplification of the hand-control logic.)

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

            # Arrow keys – mapped to dedicated names so we can trigger the
            # learned arm‐movement inference later on.
            QtCore.Qt.Key_Up: "up_arrow",
            QtCore.Qt.Key_Down: "down_arrow",
            QtCore.Qt.Key_Left: "left_arrow",
            QtCore.Qt.Key_Right: "right_arrow",
        }

        if key in mapping:
            return mapping[key]

        if text:
            ch = text.lower()
            if ch in ("w", "a", "s", "d", "q", "e", "u", "j", "f", "b", "p", "o"):
                return ch

            # Hand control keys (right Dex3)
            if ch in ("g", "h"):
                return ch
        return None

    # ------------------------------------------------------------------
    def _is_pressed(self, name: str) -> bool:
        return name in self._pressed

    # ------------------------------------------------------------------
    #  Inference-guided arm motion --------------------------------------
    # ------------------------------------------------------------------

    def _maybe_arm_inference(self, key_name: str) -> None:  # noqa: D401
        """Trigger a single inference step when *key_name* corresponds to one
        of the configured arrow / f / b commands.  The current measured
        active-arm joint angles act as *start* pose while the pressed key
        defines the high-level *direction* fed into the MLP regressor.  The
        predicted *end* joint targets are then reached smoothly by reusing
        the existing ramp logic inside *_on_arm_tick*.
        """

        # Map our canonical key names → direction string accepted by the ML
        dir_map = {
            "up_arrow": "up",
            "down_arrow": "down",
            "left_arrow": "left",
            "right_arrow": "right",
            "f": "forward",
            "b": "back",
        }

        direction = dir_map.get(key_name)
        if direction is None:
            return  # unrelated key

        # ----------------------------------------------------------------
        #  Pre-conditions – we need both SDK publisher and at least one
        #  feedback sample so the current joint positions are known.
        # ----------------------------------------------------------------
        if self._arm_pub is None:
            # Arm control not available (SDK missing or failed earlier).
            return

        # If no feedback yet we cannot tailor the *start* pose to the exact
        # current configuration.  Instead of giving up completely we fall
        # back to the last commanded angles so that inference still works –
        # this merely re-introduces the small initial snap the feedback-based
        # initialisation was meant to avoid but greatly improves usability
        # on robots/PCs where the LowState topic is unavailable.

        try:
            # Lazy import so GUI start-up does not fail if dependencies for
            # the ML model are missing.  They are lightweight (joblib, numpy,
            # pandas, scikit-learn) so import time is negligible.
            from data.inference_arm import predict_end_positions, load_bundle  # type: ignore

            # Cache the loaded bundle per arm so repeated inferences are instant.
            if not hasattr(self, "_arm_bundle_cache"):
                self._arm_bundle_cache = {}

            if self._active_arm not in self._arm_bundle_cache:
                try:
                    from pathlib import Path

                    bundle_path = Path(f"data/artifacts/{self._active_arm}-arm/arm_mlp.joblib")
                    self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                except TypeError as exc:
                    # ------------------------------------------------------------------
                    # Backwards-compatibility shim: models trained with newer
                    # scikit-learn versions pickle NumPy RandomState objects
                    # using a two-arg helper that older NumPy releases reject.
                    # Monkey-patch the ctor so *joblib.load()* succeeds.
                    # ------------------------------------------------------------------
                    if "__randomstate_ctor" in str(exc):
                        try:
                            import numpy.random._pickle as _np_pickle  # type: ignore

                            def _rs_ctor(*_args, **_kwargs):  # noqa: D401 – internal
                                import numpy as _np

                                # Return a default RandomState – the exact
                                # seed does not matter for *inference*.
                                return _np.random.RandomState()

                            _np_pickle.__randomstate_ctor = _rs_ctor  # type: ignore[attr-defined]

                            # Retry once more.
                            self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                        except Exception:
                            # Still fails – propagate to generic handler below.
                            raise
                    else:
                        raise
                except Exception:
                    # Fallback to default path inside helper if explicit path
                    # fails (relative cwd etc.).  Any error here will be
                    # handled by the outer *except*.
                    self._arm_bundle_cache[self._active_arm] = load_bundle()

            bundle = self._arm_bundle_cache[self._active_arm]

            # Current *start* joint angles (arm specific).  Prefer the
            # measured LowState sample; if that is not yet available, fall
            # back to the last commanded values so that the regressor still
            # receives a plausible pose vector.

            start_joints = [
                self._joint_cur.get(j_idx, self._cmd_q.get(j_idx, 0.0))
                for j_idx in sorted(self._arm_joint_idx)
            ]

            preds = predict_end_positions(
                direction,
                start_joints,
                arm=self._active_arm,
                bundle=bundle,
            )

            # Build new one-step pose so *_on_arm_tick* ramps toward the
            # freshly predicted targets at the already tuned speed.
            target_pose = [(self._waist_idx, self._cmd_q.get(self._waist_idx, 0.0))]
            target_pose += list(zip(sorted(self._arm_joint_idx), preds))

            self._pose_seq = [target_pose]
            self._seq_idx = 0

        except Exception as exc:  # pylint: disable=broad-except
            # Any failure should not crash the GUI – we merely report it.
            import sys

            print("[run_geoff_gui] Arm inference failed:", exc, file=sys.stderr)

    # ------------------------------------------------------------------
    #  Dex3 hand control -----------------------------------------------
    # ------------------------------------------------------------------

    def _maybe_hand_control(self, key_name: str) -> None:  # noqa: D401
        """Issue open/close commands for the Dex3 hand.

        Keyboard mapping:
            g – close (grip)
            h – open (release)
        """

        if self._dex3 is None:
            return  # Hand control not available

        # -------------------------------------------------------------
        # Simplified direct open (o) / close (p) ----------------------
        # -------------------------------------------------------------

        if key_name in ("p", "o"):
            # Fetch current measured joint angles – fall back to last cmd.
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]  # type: ignore[index]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)

            if key_name == "p":  # CLOSE
                target = self._simple_closed_pose
                self._hand_mode = "closing"
                self._hand_target = list(target)
            else:  # 'o' OPEN
                target = self._simple_open_pose
                self._hand_mode = "opening"
                self._hand_target = list(target)

            # Clear any old sequence control – we now depend solely on
            # _hand_target.  The timer loop will drive *self._hand_cmd_q*
            # towards that target every tick.

            self._hand_pose_seq.clear()
            self._hand_seq_idx = 0
            return

        # -------------------------------------------------------------
        #  Legacy / advanced modes (g/h adaptive, grabbing etc.) -------
        # -------------------------------------------------------------

        # Ensure we have captured poses available for legacy handling.
        if not getattr(self, "_hand_poses", None):
            return

        # ------------------------------------------------------------------
        # 1) Pressure-adaptive grasp (legacy – keys p / o) --------------------
        # ------------------------------------------------------------------
        if key_name in ("p", "o"):
            if self._hand_mode in ("closing", "holding") and key_name == "p":
                return  # already closing/holding
            if self._hand_mode == "opening" and key_name == "o":
                return  # already opening

            if key_name == "p":
                self._log_hand.info("Adaptive grasp triggered")
                # prepare sequence towards closed but enable adaptive flag
                target_label = "closed"
                middle = self._hand_poses.get("middle")
                target = self._hand_poses.get(target_label)
                if middle is None or target is None:
                    return

                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]  # type: ignore[index]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)

                self._hand_pose_seq = [cur]  # start from current
                self._hand_seq_idx = 0
                self._hand_mode = "adaptive"
            else:  # 'o' release fully to open pose
                print("[Dex3] adaptive open", file=sys.stderr)
                target = self._hand_poses.get("open")
                if target is None:
                    return
                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)

                self._hand_pose_seq = [cur, target]
                self._hand_seq_idx = 1
                self._hand_mode = "opening"
            return

        # ------------------------------------------------------------------
        # 2) Continuous grab / release (NEW – keys g / h) --------------------
        # ------------------------------------------------------------------

        if key_name == "g":
            # Start a new continuous grab if not already active.
            if self._hand_mode != "grabbing":
                self._hand_mode = "grabbing"
                # Continuous mode – all joints move towards the stored
                # *closed* posture right away.  No stage gating so each joint
                # continues its own approach even if others are blocked by an
                # object.
                # Clear any previously active scripted pose sequence so the
                # grab logic can take full control of the joint targets.
                self._hand_pose_seq = []
                self._hand_seq_idx = 0
                self._log_hand.info("Continuous grab initiated (torque=%.2f N·m)", self._GRAB_TAU)
            return

        if key_name == "h":
            # Abort any ongoing grab and execute the normal open sequence.
            target_label = "open"
            middle = self._hand_poses.get("middle")
            target = self._hand_poses.get(target_label)

            if target is None or middle is None:
                print("[run_geoff_gui] Missing hand pose for 'open' or 'middle'.")
                return

            # Fetch current measured joint angles – fall back to last command.
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]  # type: ignore[index]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)

            # Build new pose sequence: current -> middle -> open
            self._hand_pose_seq = [cur, middle, target]
            self._hand_seq_idx = 1
            self._hand_mode = "opening"
            self._log_hand.info("Hand opening sequence queued.")
            return

        # Ignore all other keys
        return

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
    #  Arm control helper ----------------------------------------------
    # ------------------------------------------------------------------

    def _on_arm_tick(self) -> None:  # noqa: D401
        """Periodic publisher that drives the selected arm through the
        predefined start-up poses.

        A gentle per-joint ramp (``self._STEP`` rad every 20 ms) is applied so
        the movement appears smooth without any abrupt jerks.
        """

        if self._arm_pub is None:
            return  # SDK unavailable or failed earlier

        # Prefer waiting for one *LowState* feedback sample so the commanded
        # trajectory can start exactly at the **current** joint positions –
        # this prevents an abrupt snap when the high-stiffness position
        # controller engages.  However, on some deployments the ``rt/lowstate``
        # topic is unavailable which previously meant the entire arm routine
        # stayed **disabled** forever.  To keep the GUI functional in such
        # scenarios we now fall back to the pre-initialised *zero* pose after
        # a brief grace period instead of exiting early.

        # Allow up to 1 s for the first feedback packet.  After that we assume
        # that the topic is missing and continue with the default angles so
        # at least the learnt motions and manual damp-button still work.

        if not self._joint_cur and not getattr(self, "_no_fb_deadline", None):
            # Remember the moment we first noticed the missing feedback and
            # define a deadline after which we stop waiting for rt/lowstate
            # and run the arm sequence anyway.

            self._no_fb_deadline = time.time() + 1.0  # type: ignore[attr-defined]

        if not self._joint_cur and time.time() < self._no_fb_deadline:  # type: ignore[attr-defined]
            return  # keep waiting for LowState a little longer

        # ------------------------------------------------------------------
        #  Determine current *target* pose of the active sequence step
        # ------------------------------------------------------------------

        if self._seq_idx >= len(self._pose_seq):
            target_pose = self._pose_seq[-1]  # hold final pose
        else:
            target_pose = self._pose_seq[self._seq_idx]

        # ------------------------------------------------------------------
        #  Progress commanded joint angles towards the target pose
        # ------------------------------------------------------------------

        all_reached = True
        # --------------------------------------------------------------
        #  One-shot initialisation from *measured* joint positions (if we
        #  already received at least one LowState sample).  Doing this here
        #  – before the ramp logic – ensures we start the sequence from the
        #  actual configuration and therefore avoid the sudden snap caused
        #  by commanding 0 rad at high stiffness.
        # --------------------------------------------------------------

        if not self._initialised_from_state and self._joint_cur:
            for j_idx, q_val in self._joint_cur.items():
                self._cmd_q[j_idx] = q_val
            self._initialised_from_state = True

        # ----------------------------------------------------------------
        #  Now progress every commanded joint toward the current target
        # ----------------------------------------------------------------
        for idx, tgt in target_pose:
            cur = self._cmd_q.get(idx, 0.0)
            diff = tgt - cur
            if abs(diff) <= self._SEQ_EPS:
                self._cmd_q[idx] = tgt
            else:
                step = self._STEP if diff > 0 else -self._STEP
                if abs(step) > abs(diff):
                    step = diff  # no overshoot
                self._cmd_q[idx] = cur + step
                all_reached = False

        # When everything is within tolerance advance to next pose.
        if all_reached and self._seq_idx < len(self._pose_seq):
            self._seq_idx += 1

        # ------------------------------------------------------------------
        #  Build and transmit LowCmd message
        # ------------------------------------------------------------------

        try:
            # Apply commanded q/kp/kd for each joint we touch.  Other joints
            # keep their default kp=kd=0 → firmware treats them as passive
            # (Damp).
            for idx, q in self._cmd_q.items():
                mc = self._arm_cmd.motor_cmd[idx]  # type: ignore[index]
                mc.q = q
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 60.0
                mc.kd = 1.5

            # Recompute CRC before publish (required by firmware).
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)  # type: ignore[attr-defined]

            self._arm_pub.Write(self._arm_cmd)  # type: ignore[arg-type]
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Arm publish failed:", exc, file=sys.stderr)

    # ------------------------------------------------------------------
    #  Dex3 hand control helper ----------------------------------------
    # ------------------------------------------------------------------

    def _on_hand_tick(self):  # noqa: D401
        """Smoothly progress the Dex3 hand through the currently active
        pose sequence (built by *_maybe_hand_control*).  Uses the same
        per-joint ramp approach as the arm helper so motion appears gradual
        without sudden jerks.
        """

        if self._dex3 is None:
            return

        # No work when completely idle (no target set and no sequence).
        if self._hand_mode not in ("closing", "opening", "grabbing") and not self._hand_pose_seq:
            return

        # ------------------------------------------------------------------
        #  Simplified open/close handling – move towards *self._hand_target*
        # ------------------------------------------------------------------

        if self._hand_mode in ("closing", "opening"):
            target = list(self._hand_target)

        # ------------------------------------------------------------------
        #  Continuous grab mode overrides the standard pose sequence logic.
        # ------------------------------------------------------------------

        if self._hand_mode == "grabbing":
            # Continuous tighten – drive *all* joints toward the recorded
            # closed angles, each joint at its own pace depending on current
            # difference.  There is no global stage that could block further
            # movement of other fingers.

            closed_pose = self._hand_closed_pose

            # Build target incrementally so *each* joint progresses towards
            # its closed angle independently – even if another finger is
            # already blocked by an object.  We therefore compute the next
            # step explicitly instead of relying on the static *target =
            # closed_pose* approach used earlier.

            target = list(self._hand_cmd_q)
            for j in range(7):
                cur = self._hand_cmd_q[j]
                tgt = closed_pose[j]
                diff = tgt - cur

                # Move one step in the closing direction if we have not yet
                # reached the goal (within epsilon).  This guarantees that we
                # keep "trying" to close even after temporary obstacles have
                # moved out of the way.
                if abs(diff) > self._SEQ_EPS:
                    step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                    # Avoid overshoot.
                    if abs(step) > abs(diff):
                        step = diff
                    target[j] = cur + step
                else:
                    target[j] = tgt

        else:
            # Determine current target pose from the scripted sequence.
            if self._hand_seq_idx >= len(self._hand_pose_seq):
                target = self._hand_pose_seq[-1] if self._hand_pose_seq else list(self._hand_cmd_q)
            else:
                target = self._hand_pose_seq[self._hand_seq_idx]

        # --------------------------------------------------------------
        #  Adaptive grasp – override target progression based on pressure
        # --------------------------------------------------------------
        if self._hand_mode == "adaptive":
            # Read pressures
            state = self._dex3.read_state(timeout=0.0)
            pressures = []
            if state is not None:
                try:
                    for ps in state.press_sensor_state:
                        pressures.extend(list(ps.pressure))
                except Exception:
                    pass

            self._log_hand.debug("press=%s cmd=%s", pressures[:12], [round(q,2) for q in self._hand_cmd_q])

            # Decide adjustment: if average pressure below target close all joints a bit
            avg_press = (sum(pressures)/len(pressures)) if pressures else 0.0

            if avg_press < self._PRESS_TARGET:
                # close joints towards closed pose
                target = self._hand_poses.get("closed", self._hand_cmd_q)
            else:
                target = self._hand_cmd_q  # hold

            try:
                state = self._dex3.read_state(timeout=0.0)
                if state is not None:
                    # Pressures array may have variable length; treat last 4
                    # elements as fingertip pads (thumb, index, middle, ring).
                    # Unitree docs mention 9 sensors; use max index available.
                    pressures = []
                    for ps in state.press_sensor_state:  # type: ignore[attr-defined]
                        try:
                            pressures.extend(list(ps.pressure))  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    # Pick a subset representing tips (adjust indices if needed)
                    tip_idx = [2, 5, 8, 11] if len(pressures) >= 12 else list(range(len(pressures)))
                    cnt = sum(1 for i in tip_idx if i < len(pressures) and pressures[i] >= self._PRESS_THR)

                    if cnt >= self._PRESS_MIN_COUNT:
                        # Sufficient contact – stop at current posture.
                        # Hold current posture and switch to holding mode
                        self._hand_pose_seq = [list(self._hand_cmd_q)]
                        self._hand_seq_idx = 1
                        self._hand_mode = "holding"
                        target = list(self._hand_cmd_q)
            except Exception as exc:
                print("[run_geoff_gui] adaptive grasp sensor read failed:", exc, file=sys.stderr)

        # Ramp each joint value ---------------------------------------
        all_reached = True
        for i, tgt in enumerate(target):
            cur = self._hand_cmd_q[i]
            diff = tgt - cur
            if abs(diff) <= self._HAND_STEP:
                self._hand_cmd_q[i] = tgt
            else:
                step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                self._hand_cmd_q[i] = cur + step
                all_reached = False

        # Advance to next pose if current reached.
        if self._hand_mode == "grabbing":
            # Continue running indefinitely so the controller keeps applying
            # torque and tracks any release/slippage once the object is
            # removed.
            all_reached = False
        else:
            if all_reached and self._hand_seq_idx < len(self._hand_pose_seq):
                self._hand_seq_idx += 1

        # When opening finished switch back to idle.
        if self._hand_mode == "opening" and all_reached and self._hand_seq_idx >= len(self._hand_pose_seq):
            self._hand_mode = "idle"

        # ------------------------------------------------------------------
        #  Publish command ---------------------------------------------------
        # ------------------------------------------------------------------
        try:
            cmd = self._dex3._make_zero_cmd()  # type: ignore[attr-defined]
            mins, maxs = self._dex3._limits()  # type: ignore[attr-defined]

            # Use gentle gains.
            # Increased gains so all joints, especially the high-load base
            # joints, receive enough authority to reach their target angles.
            kp = 8.0  # stronger proportional gain for authoritative position hold
            kd = 1.5

            for i, q in enumerate(self._hand_cmd_q):
                mode = self._dex3._pack_mode(i, status=0x01, timeout=False)  # type: ignore[attr-defined]
                mc = cmd.motor_cmd[i]  # type: ignore[index]
                mc.mode = mode
                mc.kp = kp
                mc.kd = kd

                # Feed-forward torque for continuous grab mode so the fingers
                # keep applying a closing force even after the desired joint
                # angles have been reached.
                if self._hand_mode in ("grabbing", "closing", "holding"):
                    mc.tau = max(0.3, self._GRAB_TAU) * self._close_dir[i]
                else:
                    mc.tau = 0.0

                # Clamp within URDF limits to avoid faults.
                mc.q = max(min(q, maxs[i]), mins[i])

            ok = self._dex3._publish(cmd)  # type: ignore[attr-defined]

            if not ok:
                # First time detect – warn user once.
                if not getattr(self, "_dex3_no_match_warned", False):
                    print(
                        "[Dex3] WARNING – no subscriber matched for rt/dex3 cmd; retrying…",
                        file=sys.stderr,
                    )
                    self._dex3_no_match_warned = True  # type: ignore[attr-defined]

                # Perform a couple of quick retries so the first few timer
                # ticks after start still deliver at least one successful
                # publish once the DDS topic matches (mirrors grip/open
                # behaviour which loops ~20×).
                for _ in range(3):
                    if self._dex3._publish(cmd):  # type: ignore[attr-defined]
                        break
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Dex3 publish failed:", exc, file=sys.stderr)
            self._arm_pub = None

    # ------------------------------------------------------------------
    #  GUI button – Damp upper body & center waist ----------------------
    # ------------------------------------------------------------------

    def _on_damp_pressed(self):  # noqa: D401
        """Make both arms passive (kp=kd=0) while keeping the legs in
        balanced stand, then set the waist yaw joint to 0 rad so the torso
        faces forward again.  Executes once when the user clicks the button.
        """

        if self._arm_pub is None:
            print("[run_geoff_gui] Damp request – SDK unavailable")
            return

        # Stop the automatic arm timer so it no longer overwrites our
        # one-off passive command.
        try:
            if hasattr(self, "_arm_timer"):
                self._arm_timer.stop()
        except Exception:
            pass

        try:
            # 1) Set kp=kd=0 for all right-arm joints so they go limp.
            # Make *both* arms limp so users can safely interact regardless of
            # which side is currently active.
            for idx in (*range(15, 22), *range(22, 29)):
                mc = self._arm_cmd.motor_cmd[idx]  # type: ignore[index]
                # Keep current angle to avoid sudden jump when stiffness drops.
                mc.q = self._cmd_q.get(idx, 0.0)
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0

            # 2) Center waist at 0 rad with normal stiffness so the torso
            #    stays facing forward.
            waist_idx = getattr(self, "_waist_idx", 12)
            mc_w = self._arm_cmd.motor_cmd[waist_idx]  # type: ignore[index]
            mc_w.q = 0.0
            mc_w.dq = 0.0
            mc_w.tau = 0.0
            mc_w.kp = 60.0
            mc_w.kd = 1.5

            # Update internal cmd_q so later calls (if the user re-enables
            # the arm sequence) start from the centred posture.
            self._cmd_q[waist_idx] = 0.0

            # Recompute CRC and publish once.
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)  # type: ignore[attr-defined]
            self._arm_pub.Write(self._arm_cmd)  # type: ignore[arg-type]

            print("[run_geoff_gui] Arms damped, waist centred.")

        except Exception as exc:  # pylint: disable=broad-except
            print("[run_geoff_gui] Damp request failed:", exc, file=sys.stderr)

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
        if hasattr(self, "_arm_timer"):
            self._arm_timer.stop()
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

        # -------------------------------------------------------------
        # Apply the same LiDAR mount correction that live_slam uses so
        # the visual pose axes appear level when the head is tilted.
        # -------------------------------------------------------------
        try:
            from live_slam import _R_MOUNT as _LS_R_MOUNT  # type: ignore

            if _LS_R_MOUNT is not None:
                rot = rot @ _LS_R_MOUNT[:3, :3]
        except Exception:  # pragma: no cover – live_slam missing in tests
            pass

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
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="left",
        help="Which arm to control on startup (default: left)",
    )

    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="left",
        help="Which Dex3 hand is connected (default: left).  Pass --hand right "
        "when a right-hand unit is attached.",
    )

    parser.add_argument(
        "--grip-force",
        type=float,
        dest="grip_force",
        metavar="N·m",
        default=0.3,
        help="Feed-forward torque (approx. N·m) applied during continuous grabs (default: 0.3)",
    )
    args = parser.parse_args()

    window = GeoffWindow(
        args.iface,
        args.clear,
        hand=args.hand,
        grip_force=args.grip_force,
    )
    window._active_arm = args.arm  # type: ignore[attr-defined]
    # Re-initialise arm variables with the command-line choice.
    window._arm_selector.setCurrentIndex(0 if args.arm == "left" else 1)

    try:
        window._configure_arm_variables()
    except Exception as exc:  # pragma: no cover – config failures should not crash
        print("[run_geoff_gui] Initial arm switch failed:", exc, file=sys.stderr)

    window.run()


if __name__ == "__main__":
    main()
