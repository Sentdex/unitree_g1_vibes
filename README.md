# WIP

Main file here is run_geoff_gui.py, everything ties into that.

I'll clean this up later. 

maybe.

# RGB / Depth camera stuff:

Crontab on the jetson board:
@reboot /usr/bin/python3 /home/unitree/jetson_realsense_stream.py --client-ip 192.168.123.222 --width 640 --height 480 --fps 30

https://github.com/Sentdex/unitree_g1_vibes/blob/main/jetson_realsense_stream.py

Receiver side: 
https://github.com/Sentdex/unitree_g1_vibes/blob/main/receive_realsense_gst.py



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

