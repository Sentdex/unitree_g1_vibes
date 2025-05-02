# RealSense Streaming Documentation

Welcome 👋 – this mini-site explains **how to stream depth & RGB data from the RealSense D435i that sits on the Unitree G1 Jetson NX to your laptop/work-station**.

It is split into two parts:

1. **Quick start** – for when you *just* want to get pictures flowing within seconds.
2. **In-depth** – learn what the individual scripts do, how GStreamer is wired up, network requirements, performance tips & troubleshooting.

Use the navigation bar on the left (or the hamburger on mobile) to jump between sections.

> **TL;DR**  
> On the Jetson run
>
> ```bash
> python3 jetson_realsense_stream.py --client-ip <IP-of-your-PC>
> ```
>
> On your PC run
>
> ```bash
> python3 receive_realsense_gst.py
> ```

That’s it! 🎉
