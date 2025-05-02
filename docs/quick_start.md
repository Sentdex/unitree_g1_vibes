# Quick Start ⚡️

If you only care about **seeing colour & depth images as quickly as possible** follow the steps below.  For *why* things are done this way see the other chapters.

## 1  Wiring & network

• Connect the Intel RealSense **D435i** camera to the **Jetson NX** USB-C port.  
• Make sure your laptop/desktop and the Jetson are on the **same subnet** (e.g. `192.168.123.*`). 

> By default the scripts send/receive on UDP **ports 5600 (RGB) and 5602 (Depth)**.  Ensure these are not blocked by a firewall.

## 2  Install dependencies  
### Jetson

```bash
sudo apt update \
  && sudo apt install -y python3-gi gstreamer1.0-tools \
     gstreamer1.0-plugins-{good,bad} gstreamer1.0-libav

python3 -m pip install --user --upgrade pyrealsense2 numpy opencv-python
```

### Laptop / Work-station

```bash
sudo apt update \
  && sudo apt install -y python3-gi gir1.2-gst-plugins-base-1.0 \
     gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad gstreamer1.0-libav

python3 -m pip install --user --upgrade numpy opencv-python
```

## 3  Start the sender on the Jetson

Replace `<PC-IP>` with the IP address of the machine that should *receive* the streams:

```bash
python3 jetson_realsense_stream.py --client-ip <PC-IP> --width 640 --height 480 --fps 30
```

You should see log output similar to:

```
Using RealSense serial 901424060166  @ 640×480 30 fps
GStreamer pipeline → 192.168.123.222 (RGB:H264→5600 | Depth:H264→5602)
```

## 4  Start the receiver on the laptop

```bash
python3 receive_realsense_gst.py
```

An OpenCV window opens that shows **RGB** on the left and the **colour-mapped depth** on the right together with a live FPS overlay.

Press **q** or **ESC** to quit.

🎉 **Done!**   You are now streaming depth and colour in real-time at ~30 FPS over the network.
