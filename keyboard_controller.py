#!/usr/bin/env python3
"""
keyboard_controller.py – simple WASD-style tele-op for Unitree G-1.

The script:
1. Runs the hanger boot sequence so the robot starts balancing.
2. Enters a curses UI where you can drive with the keys below.

Controls
---------
    W / S : forward / backward velocity
    A / D : yaw left / right (turn)
    Q / E : lateral left / right (optional, G-1 supports side-step)
    Space : stop (zero velocities)
    Z      : Damp (soft) and exit
    Esc    : emergency stop & exit (ZeroTorque)

Velocities are applied continuously – every key-press adjusts the target
values which are sent to the robot at 10 Hz.
"""

from __future__ import annotations

import argparse

# ------------------------------------------------------------------------
# Notes on the 2025-04-28 update
# ------------------------------------------------------------------------
#
# * Continuous command _while a key is physically held_. As soon as the key is
#   released the corresponding velocity is reset to **zero**.
# * Supports holding several keys together – e.g. **W + A** to move forward
#   while turning left.
#
# This requires real “key-up” events which the `curses` module cannot provide.
# We now use the lightweight third-party `pynput` package to poll the current
# key state.  It communicates with the X-server (Linux), Win32, or Quartz and
# therefore works for normal users on typical desktop sessions (no sudo).
# Curses is kept only for drawing the tiny on-screen HUD.
#
# When running under Wayland `pynput` might still fall back to reading
# `/dev/input/event*`; in that corner-case you’d again need the permissions or
# group/udev tweaks previously mentioned.
# ------------------------------------------------------------------------

import time

# Third-party ---------------------------------------------------------------

import curses

# We now use the cross-platform `pynput` library which reads key events via the
# X server / Win32 API / Quartz, so it works unprivileged on desktop Linux,
# macOS and Windows.  On Wayland sessions `pynput` falls back to /dev/input and
# may again need the permissions discussed earlier—but on X11 (the default on
# many distros) it works out-of-the-box.

try:
    from pynput.keyboard import Listener, Key, KeyCode  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'pynput' package is required for keyboard_controller.py.\n"
        "Install with:  pip install pynput"
    ) from exc

from hanger_boot_sequence import hanger_boot_sequence


# ---------------------------------------------------------------------------
# Parameter defaults
# ---------------------------------------------------------------------------

LIN_STEP = 0.05  # m/s per press
ANG_STEP = 0.2   # rad/s per press

SEND_PERIOD = 0.1  # seconds (10 Hz)


def clamp(value: float, limit: float = 0.6) -> float:
    return max(-limit, min(limit, value))


def drive_loop(stdscr: "curses._CursesWindow", bot) -> None:
    # Curses setup for a tiny on–screen HUD.
    curses.cbreak()
    stdscr.nodelay(True)  # Make getch() non-blocking so the UI stays alive.

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------
    vx = vy = omega = 0.0  # target velocities that will be sent to the robot

    last_send = 0.0

    # ------------------------------------------------------------------
    # Keyboard listener setup (pynput)
    # ------------------------------------------------------------------

    pressed_keys: set[object] = set()  # holds Key / single-char strings

    def _on_press(key):  # noqa: D401 – tiny helper
        """Callback – store the key object / char in *pressed_keys*."""
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.add(key.char.lower())
        else:
            pressed_keys.add(key)

    def _on_release(key):
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.discard(key.char.lower())
        else:
            pressed_keys.discard(key)

    listener = Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    def key(name: str) -> bool:  # helper similar to keyboard.is_pressed
        if name == "space":
            return Key.space in pressed_keys
        if name == "esc":
            return Key.esc in pressed_keys
        return name in pressed_keys

    try:
        while True:
            # ------------------------------------------------------------------
            # 1.  Build/refresh the target velocity based on current key states.
            # ------------------------------------------------------------------

            if key("w") and not key("s"):
                vx = clamp(vx + LIN_STEP)
            elif key("s") and not key("w"):
                vx = clamp(vx - LIN_STEP)
            else:
                vx = 0.0

            if key("q") and not key("e"):
                vy = clamp(vy + LIN_STEP)
            elif key("e") and not key("q"):
                vy = clamp(vy - LIN_STEP)
            else:
                vy = 0.0

            if key("a") and not key("d"):
                omega = clamp(omega + ANG_STEP)
            elif key("d") and not key("a"):
                omega = clamp(omega - ANG_STEP)
            else:
                omega = 0.0

            # Space bar – an emergency stop of sorts that zeroes everything no
            # matter what other keys are held.
            if key("space"):
                vx = vy = omega = 0.0

            # ------------------------------------------------------------------
            # 2.  Handle exit conditions.
            # ------------------------------------------------------------------
            if key("z"):
                bot.Damp()
                break

            if key("esc"):
                bot.StopMove()
                bot.ZeroTorque()
                break

            # ------------------------------------------------------------------
            # 3.  Send the command at the configured rate and update HUD.
            # ------------------------------------------------------------------
            now = time.time()
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now

                stdscr.erase()
                stdscr.addstr(0, 0, "Hold keys to drive – Z: quit  ESC: e-stop")
                stdscr.addstr(2, 0, f"vx: {vx:+.2f}  vy: {vy:+.2f}  omega: {omega:+.2f}")
                stdscr.refresh()

            # A very small sleep keeps CPU usage civilised (<1 % on typical PCs).
            time.sleep(0.005)

    finally:
        # Ensure the listener thread is stopped before leaving curses context.
        listener.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="enp68s0f1", help="network interface connected to robot")
    args = parser.parse_args()

    # Boot sequence – returns initialised LocoClient in FSM-200
    bot = hanger_boot_sequence(iface=args.iface)

    curses.wrapper(drive_loop, bot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – sending Damp …")
        try:
            bot.Damp()  # type: ignore[name-defined]
        except Exception:
            pass
