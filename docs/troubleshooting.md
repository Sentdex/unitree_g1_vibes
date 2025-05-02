# Troubleshooting 🐛

Having problems?  Here are the most common issues & fixes.

| Symptom | Possible reason | Fix |
|---------|-----------------|------|
| `RuntimeError: No device connected` on the Jetson | USB connection isn’t detected | Re-seat the USB-C plug, check `lsusb -d 8086:0b3a` lists the camera.  Use a *short* USB-C cable. |
| Black window on the receiver | Firewall blocks UDP 5600/5602 | Allow incoming UDP or temporarily disable firewall. |
| Huge latency (>500 ms) | Wi-Fi congestion or power-saving | Use 5 GHz Wi-Fi or Ethernet. Disable `power_save=1` on the Jetson Wi-Fi. |
| Message `WARNING: from element udpsrc0: GStreamer warning: can't allocate buffers` | packets lost – bandwidth too low | Lower resolution/FPS (`--width 424 --height 240 --fps 15`) or increase encoder bitrate. |
| `cv2.imshow` crashes with *qt5 missing* | OpenCV not compiled with HighGUI | Install the official wheels (`pip install opencv-python`) or build OpenCV with GUI support. |

## Debugging tips

• Run the sender with `GST_DEBUG=2` environment variable to see the pipeline status.  
• You can inspect the incoming RTP packets with Wireshark (`udp.port == 5600 or udp.port == 5602`).  
• To record a short clip on the receiver side:  

```bash
gst-launch-1.0 -e udpsrc port=5600 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
  rtph264depay ! h264parse ! mp4mux ! filesink location=rgb.mp4
```

The resulting MP4 can be played with VLC and helps verifying that the sender pipeline is healthy regardless of the Python receiver.
