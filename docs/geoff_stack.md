# Geoff Robot – Full Software Stack 🦾

Welcome!  This page gives you the **big-picture** view of everything that
comes together when you launch `run_geoff_gui.py`.

Use it as an interactive map – expand the call-outs, click the links, zoom the
diagrams – and you should walk away with a clear mental model of *how the
pieces talk to each other*.

---

## 1. Architectural bird-eye view

```mermaid
flowchart LR
    subgraph Sensors & HW
        RS["RealSense<br/>D435i"]
        Lx["Livox<br/>Mid-360 LiDAR"]
        IMU[Unitree G1<br/>IMU]
    end

    subgraph Jetson NX  
        style Jetson NX fill:#e8f0ff,stroke:none
        RS  -- USB3 -->  JETRX[jetson_realsense_stream.py]
        Lx  -- Ethernet/UDP --> LIVOX[livox2_python.py]
        IMU -- Serial -->  IMU_IN[unitree_sdk2_python]

        JETRX -- RTP/H.264 --> Net((LAN))
        LIVOX -- Custom UDP --> Net
        IMU_IN -- Protobuf --> Net
    end

    subgraph Laptop / Ground-station
        style Laptop / Ground-station fill:#e8f0ff,stroke:none
        Net --> RX[_rx_realsense (run_geoff_stack)]
        Net --> LDL[lidar_reader (g1_lidar.*)]
        Net --> FSM[fsm_controller]
        RX & LDL & FSM --> GUI[run_geoff_gui.py<br/>PySide6 GUI]
        GUI --> USER([Keyboard / Game-pad])
    end

    style Net fill:#fff,stroke:#999,stroke-dasharray: 5 5
```

Hover over a component in the *live* documentation site to see a brief
tooltip of its responsibilities.  The diagram is generated straight from the
Mermaid code block – **feel free to hack it** if you add new sensors or
replace scripts.

---

## 2. Threading model inside `run_geoff_gui.py`

The GUI is only the tip of the iceberg – under the hood several **background
threads** keep data flowing while Qt happily renders at 60 FPS.

| Thread | Target | Purpose |
| ------ | ------ | ------- |
| `Qt` *(main)* | PySide6 | Event-loop, painting, keyboard interception |
| `SLAM` | `live_slam.LiveSLAMDemo` | Runs OpenVSLAM and pushes a point-cloud into a shared variable every ~50 ms |
| `RealSense RX` | `_rx_realsense` | Receives the 2× RTP streams, converts them to NumPy and updates `SharedState.rgb/depth` |
| `LiDAR RX` | `g1_lidar.live_view` | Streams Livox packets & keeps the latest sweep in memory |

All communication is **lock-free** where possible (double buffering), protected
by a lightweight `threading.Lock` only when absolutely required.

```text
GUI thread ─┬─ poll() SharedState every paintEvent()
            ├─ draws RGB & Depth QLabel
            └─ plots SLAM point-cloud in a GLViewWidget

SLAM thread ──> pushes (xyz, pose) tuples ─┐
RS RX thread ──> writes new BGR frames     ├─ SharedState
LiDAR thread ──> writes new sweeps        ┘
```

---

## 3. “Show me the code!” – expandable snippets

=== "Starting the GUI"

    ```python title="run_geoff_gui.py (excerpt)"
    from run_geoff_stack import _rx_realsense, _state, _state_lock

    # …
    rs_thr = threading.Thread(target=_rx_realsense, daemon=True)
    rs_thr.start()

    slam_thr = threading.Thread(target=_run_slam, args=(shutdown_evt,), daemon=True)
    slam_thr.start()
    ```

=== "RealSense receiver"

    ```python title="run_geoff_stack.py (simplified)"
    def _rx_realsense():
        # Build 2× GStreamer pipelines → pull buffers → numpy arrays
        while not stop_evt.is_set():
            rgb, depth = _pull_next()
            with _state_lock:
                _state.rgb = rgb
                _state.depth = depth
    ```

Code tabs use **Material for MkDocs’** `===` tab syntax which renders as
switchable buttons in the browser.

---

## 4. Deep-dives & external docs

* Need details on the Livox packet format?  Head over to the original
  `livox_docs.html` – it is bundled with the site.
* Want to tweak the RealSense encoder?  See **Quick start → Scripts
  reference**.
* Curious about the finite-state machine that keeps Geoff upright?  Open
  `fsm_cheatsheet.html` for a state diagram and control-flow tables.

All three HTML references are copied into the built site so you can open them
offline without an internet connection.

---

## 5. Build the docs locally

```bash
# 1. Install the deps once
pip install mkdocs-material pygments pymdown-extensions

# 2. Serve with live reload
mkdocs serve

# 3. Or build a static site in the "site/" folder
mkdocs build
```

> **Tip** – MkDocs’ `--dirtyreload` flag speeds up rebuilds when you only
> change Markdown/diagrams.

Enjoy exploring 🕹️
