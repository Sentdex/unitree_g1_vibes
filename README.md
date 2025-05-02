# Livox MID-360 – Pure-Python Point-Cloud & SLAM

This mini-repo gives you a 100 % Python workflow for the **Livox MID-360** LiDAR
(or any other Livox unit) – no ROS required.

* ``livox_python.py``  – ctypes wrapper around the official **Livox-SDK**
* ``live_slam.py``    – real-time LiDAR-SLAM demo (KISS-ICP + Open3D)
* ``live_points.py``   – bare-bones *live* point-cloud viewer (no SLAM)

## Quick start

```bash
# 1. Build the vendor SDK once (Linux example)
git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install    # installs liblivox_sdk.so → /usr/local/lib
sudo ldconfig

# 2. Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # open3d, kiss-icp, numpy …

# 3. Run!
python live_slam.py
```

If everything is on the same subnet (LiDAR: **192.168.123.120**,
PC: **192.168.123.222**) you’ll see a real-time growing point-cloud map in an
Open3D window.

## Custom processing

Override (or monkey-patch) the method

```python
def Livox.handle_points(self, xyz: np.ndarray):
    ...
```

to log frames to disk, feed a neural net, etc.

Enjoy – and open an issue / PR if you hit any problems!
