<!--
  Interactive stack explorer. The content below is lifted from the original
  stand-alone HTML file but turned into a Markdown page so that it is wrapped
  by the Material theme.  That way it inherits the site palette, header and
  (crucially) the dark-mode toggle.
-->

# Interactive Geoff-stack explorer

<style>
  /* Let the page blend in with the Material palette. */
  html body {
    height: 100%;
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
      Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
    background: var(--md-default-bg-color);
    color: var(--md-default-fg-color);
  }

  #wrapper {
    display: flex;
    height: calc(100vh - 4.5rem); /* subtract header height */
    width: 100vw;                /* span full viewport width */
    margin-left: calc(50% - 50vw); /* escape theme max-width centre */
  }

  #mynetwork {
    flex: 0 0 60%;
    border-right: 1px solid var(--md-default-fg-color--lightest, #ccc);
    height: 100%;
  }

  #info {
    flex: 1 1 40%;
    padding: 1.2rem 2rem;
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
  }

  h2 {
    margin-top: 0;
  }

  pre {
    background: var(--md-code-bg-color, #2e3440);
    padding: 0.8rem 1rem;
    border-radius: 4px;
    overflow-x: auto;
  }

  /* Adaptive table styling that respects the current theme */
  table.mini {
    font-size: 0.8rem;
    border-collapse: collapse;
    width: 100%;
    margin-top: 0.3rem;
  }
  table.mini th,
  table.mini td {
    padding: 4px 6px;
    border: 1px solid var(--md-default-fg-color--lightest, #444);
  }
  table.mini th {
    background: var(--md-table-head-background-color, var(--md-accent-fg-color));
    color: var(--md-table-head-color, var(--md-accent-bg-color));
  }
  table.mini tr:nth-child(even) td {
    background: var(--md-table-row-even-background-color, rgba(255,255,255,0.05));
  }

  code {
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier,
      monospace;
    font-size: 0.9rem;
  }

  .fade {
    opacity: 0.3;
  }
</style>

<!-- Vis-Network (stand-alone build); loaded from CDN. If you are offline
     replace the src/href with local copies inside docs/assets/. -->
<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js" defer></script>
<link href="https://unpkg.com/vis-network/styles/vis-network.min.css" rel="stylesheet" type="text/css">

<div id="wrapper">
  <div id="mynetwork"></div>
  <div id="info">
    <h2>Welcome 👋</h2>
    <p>
      Click a node to learn what part it plays inside
      <strong>run_geoff_gui.py</strong>. Double-click a node to follow the
      "Source" link when available.
    </p>
    <p style="font-size:0.85rem;opacity:0.7">
      Tip – drag the nodes around, zoom with the mouse-wheel.
    </p>
  </div>
</div>

<script type="text/javascript">
  window.addEventListener('DOMContentLoaded', () => {
    // Abort gracefully if vis.js failed to load
    if (typeof vis === 'undefined') {
      const inf = document.getElementById('info');
      inf.innerHTML =
        '<h2>vis-network missing</h2><p>The interactive graph needs ' +
        'the <code>vis-network</code> library. Connect to the internet or ' +
        'place a local copy under <code>docs/assets/</code> and update ' +
        'the <code>&lt;script&gt;</code> tag.</p>';
      return;
    }

    // ----------- Graph data --------------------------------------------
    const nodes = new vis.DataSet([
      { id: 'gui', label: 'run_geoff_gui.py', shape: 'box', color: '#1976d2' },

      { id: 'rs_rx', label: 'RealSense Receiver\n_rx_realsense', group: 'worker' },
      { id: 'slam', label: 'live_slam', group: 'worker' },
      { id: 'lidar', label: 'Livox LiDAR\nlive_view', group: 'worker' },
      { id: 'teleop', label: 'Keyboard\ntele-op', group: 'worker' },

      { id: 'rgb', label: 'RGB stream', group: 'stream' },
      { id: 'depth', label: 'Depth stream', group: 'stream' },

      { id: 'lidar_sensor', label: 'LiDAR sensor', group: 'sensor' },
      { id: 'rs_cam', label: 'RealSense D435i', group: 'sensor' },
      { id: 'jetson', label: 'Jetson NX\n(on Unitree G1)', group: 'compute' },

      {
        id: 'user',
        label: 'You',
        shape: 'star',
        color: '#6a1b9a',
        font: { color: '#ffffff', strokeWidth: 0 }
      }
    ]);

    const edges = new vis.DataSet([
      // Sensors into Jetson
      { from: 'rs_cam', to: 'jetson', arrows: 'to' },
      { from: 'lidar_sensor', to: 'jetson', arrows: 'to' },

      // Jetson streams out to laptop workers
      { from: 'jetson', to: 'rs_rx', arrows: 'to' },
      { from: 'jetson', to: 'lidar', arrows: 'to' },

      // Tele-op commands back to Jetson
      { from: 'teleop', to: 'jetson', arrows: 'to' },

      // Workers to GUI / user
      { from: 'rs_rx', to: 'rgb' },
      { from: 'rs_rx', to: 'depth' },
      { from: 'rgb', to: 'gui' },
      { from: 'depth', to: 'gui' },

      { from: 'lidar', to: 'gui' },
      { from: 'slam', to: 'gui' },
      { from: 'teleop', to: 'gui' },

      { from: 'user', to: 'teleop', arrows: 'to' },
      { from: 'gui', to: 'user', arrows: 'to' }
    ]);

    // ----------- Node explanations ------------------------------------
    const docs = {
      gui: {
        title: 'run_geoff_gui.py',
        html: `
<p>The main PySide6 window. It embeds:</p>
<ul>
  <li>Two <code>QLabel</code>s for live RGB & depth.</li>
  <li>A <code>GLViewWidget</code> for the SLAM point-cloud.</li>
  <li>An interactive bird-eye occupancy map.</li>
</ul>
<p>It also installs a global Qt <em>event-filter</em> so key-presses steer Geoff
without the separate <code>keyboard_controller.py</code> thread.</p>
<p><a href="https://github.com/your-repo/run_geoff_gui.py" target="_blank">Source ↗</a></p>`
      },
      rs_rx: {
        title: '_rx_realsense (worker thread)',
        html: `
<p><code>jetson_realsense_stream.py</code> (on-board Jetson) pushes two RTP/H.264 streams which this worker decodes with GStreamer.</p>

<h4 style="margin:0.8rem 0 0.3rem;font-size:0.9rem">Network ports</h4>
<table class="mini">
  <tr><th>Port</th><th>Payload</th></tr>
  <tr><td>5600/udp</td><td>RTP payload 96 · H.264-encoded RGB</td></tr>
  <tr><td>5602/udp</td><td>RTP payload 97 · H.264-encoded colourised depth</td></tr>
</table>

<h4 style="margin:0.8rem 0 0.3rem;font-size:0.9rem">Sender&nbsp;CLI</h4>
<pre style="background:var(--md-code-bg-color);padding:6px 8px;font-size:0.8rem">
python3 jetson_realsense_stream.py --client-ip 192.168.123.222 \
      --width 640 --height 480 --fps 30
</pre>

<p style="font-size:0.85rem;margin-top:0.6rem">Receiver writes the decoded NumPy frames into a shared struct read by the GUI every paint-event.</p>`
      },
      rgb: {
        title: 'RGB stream',
        html:
          '<p>8-bit BGR @ 30 FPS, 640×480. Encoded with the Jetson\'s ' +
          'hardware H.264 encoder (<code>nvv4l2h264enc</code>).</p>'
      },
      depth: {
        title: 'Depth stream',
        html:
          '<p>16-bit depth → colourised to 8-bit BGR so it can be ' +
          'H.264-encoded as well.</p>'
      },
      lidar: {
        title: 'live_view (LiDAR worker)',
      html: `
<p><strong>Livox MID-360</strong> UDP packets → NumPy XYZ points.</p>

<h3 style="margin:0.8rem 0 0.4rem">Key files</h3>
<ul style="font-size:0.9rem;line-height:1.4">
  <li><code>livox2_python.py</code> &amp; <code>livox_python.py</code> – ctypes wrappers around the vendor SDKs.</li>
  <li><code>live_points.py</code> – quick Open3D viewer to sanity-check the stream.</li>
  <li><code>live_slam.py</code> – feeds frames into KISS-ICP.</li>
</ul>

<h3 style="margin:1rem 0 0.4rem">Mini tech-stack</h3>
<table class="mini">
  <tr><th>Layer</th><th>Role</th></tr>
  <tr><td>Livox-SDK&nbsp;2</td><td>Discovers sensor, converts UDP → Cartesian</td></tr>
  <tr><td>ctypes wrapper</td><td>Exposes each packet as NumPy array</td></tr>
  <tr><td>NumPy</td><td>Zero-copy lingua-franca for Python</td></tr>
  <tr><td>KISS-ICP</td><td>Odometry & voxel-hash map</td></tr>
  <tr><td>Open3D</td><td>Fast point-cloud visualisation</td></tr>
</table>`
      },
      slam: {
        title: 'live_slam.py',
      html: `
<p><strong>KISS-ICP</strong> runs in a background thread turning raw frames into odometry &amp; a growing map.</p>
<h4 style="margin:0.7rem 0 0.3rem;font-size:0.9rem">Data-flow</h4>
<pre style="background:var(--md-code-bg-color);padding:6px 8px;font-size:0.8rem">
LiDAR frame → KISS-ICP → { pose 4×4, local voxel map } → Qt GUI
</pre>
<h4 style="margin:0.8rem 0 0.3rem;font-size:0.9rem">Highlights</h4>
<ul style="font-size:0.9rem;line-height:1.4">
  <li>Monkeys patched KISS-ICP viewer to avoid opening its own GLFW window.</li>
  <li>Pose is published to the tele-op module for closed-loop control.</li>
  <li>Thread-safe ring buffer ensures GUI never blocks SLAM.</li>
</ul>`
      },
      teleop: {
        title: 'Keyboard tele-op',
      html: `
<p>Captures <kbd>W&nbsp;A&nbsp;S&nbsp;D</kbd>/<kbd>↑ ← ↓ →</kbd> and publishes velocity commands via ZeroMQ.</p>

<h4 style="margin:0.8rem 0 0.3rem;font-size:0.9rem">FSM cheat-sheet</h4>
<table class="mini">
  <tr><th>FSM&nbsp;ID</th><th>State</th><th>Use</th></tr>
  <tr><td>0</td><td>Zero-Torque</td><td>Motors off – safe to handle legs</td></tr>
  <tr><td>1</td><td>Damp</td><td>Viscous damping – default power-on</td></tr>
  <tr><td>4</td><td>Stand-up</td><td>Firmware stand – good after Damp</td></tr>
  <tr><td>200</td><td><strong>Start</strong></td><td>Balance controller & gait planner</td></tr>
</table>
<h4 style="margin:1rem 0 0.3rem;font-size:0.9rem">Bring-up sequence</h4>
<ol style="font-size:0.85rem;padding-left:1.2rem;line-height:1.4">
  <li><code>Damp</code> – joints soft.</li>
  <li><code>Stand-up</code> (FSM&nbsp;4) – firmware extends legs part-way.</li>
  <li>Raise <code>SetStandHeight</code> until <code>mode</code> flips 2→0.</li>
  <li><code>BalanceStand(0)</code>.</li>
  <li>Re-send final <code>SetStandHeight</code>.</li>
  <li><code>Start</code> (FSM&nbsp;200) – engage walking controller.</li>
</ol>`
      },
      rs_cam: {
        title: 'Intel RealSense D435i',
        html:
          '<p>Stereo RGB-D camera delivering depth & colour over USB3. ' +
          'Configured to 640×480 @ 30 FPS for low latency.</p>'
      },
      lidar_sensor: {
        title: 'Livox MID-360',
        html:
          '<p>360° spinning LiDAR providing 100 k pts/s. UDP multicast ' +
          'packets are parsed by the Python SDK wrapper.</p>'
      },
      user: {
        title: 'That’s you!',
        html:
          '<p>Use the keyboard/game-pad to drive; watch the combined sensor ' +
          'data update in real-time.</p>'
      },
      jetson: {
        title: 'Jetson NX (inside the Unitree G1)',
        html:
          '<p><strong>NVIDIA Jetson Xavier NX</strong> that sits on the robot ' +
          'itself.  Runs two main Python services:</p>' +
          '<ul><li><code>jetson_realsense_stream.py</code> – captures RGB & depth, ' +
          'encodes them with the hardware H.264 engine and multicasts RTP.</li>' +
          '<li><code>livox2_python.py</code> – ingests Livox UDP packets, does ' +
          'basic filtering then forwards them to the ground-station.</li></ul>' +
          '<p>Receives velocity commands from the tele-op thread and passes ' +
          'them to the Unitree low-level controller.</p>'
      }
    };

    // ----------- Vis options -----------------------------------------
    const options = {
      physics: {
        stabilization: { iterations: 250 },
        barnesHut: {
          gravitationalConstant: -2000,
          springLength: 120,
          springConstant: 0.04
        }
      },
      interaction: { hover: true },
      groups: {
        worker: { color: { background: '#ffb74d' }, shape: 'ellipse' },
        stream: { color: { background: '#4db6ac' }, shape: 'ellipse' },
        sensor: { color: { background: '#90a4ae' }, shape: 'database' },
        compute: { color: { background: '#c5e1a5' }, shape: 'box' }
      }
    };

    const container = document.getElementById('mynetwork');
    const network = new vis.Network(container, { nodes, edges }, options);

    // ----------- UI behaviour -----------------------------------------
    const info = document.getElementById('info');

    function show(id) {
      const doc = docs[id];
      if (!doc) return;
      info.innerHTML = `<h2>${doc.title}</h2>${doc.html}`;
    }

    network.on('click', (params) => {
      if (params.nodes.length === 1) {
        show(params.nodes[0]);
      }
    });

    network.on('doubleClick', (params) => {
      const id = params.nodes[0];
      const doc = docs[id];
      if (doc && doc.link) window.open(doc.link, '_blank');
    });

    // Auto-select GUI on load
    network.once('stabilized', () => show('gui'));
  });
</script>
