"""
hanger_boot_sequence.py – utility to bring a hanging Unitree G-1 from
power-on (Damp) to a balanced stand ready for walking.  The helper will now
detect when the robot is already in that balanced stand (FSM-200) and simply
return the client immediately, avoiding a redundant second bring-up cycle.

Call `hanger_boot_sequence()` from any script and it returns an initialised
`LocoClient` instance that is already in FSM-200.  The helper now performs a
sanity-check after the leg-extension sweep: if the firmware still reports
“feet unloaded” (mode = 2) we pause, print a warning and wait for the
operator to tweak the hanger height and press <Enter>.  The sweep is then
repeated until mode 0 (feet loaded) is observed.  All parameters are
optional and identical to those we used during development.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rpc_get_int(client: LocoClient, api_id: int) -> Optional[int]:
    try:
        code, data = client._Call(api_id, "{}")  # type: ignore[attr-defined]
        if code == 0 and data:
            import json

            return json.loads(data).get("data")
    except Exception:
        pass
    return None


def _fsm_id(client: LocoClient) -> Optional[int]:
    return _rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_ID)


def _fsm_mode(client: LocoClient) -> Optional[int]:
    return _rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_MODE)


def hanger_boot_sequence(
    iface: str = "enp68s0f1",
    step: float = 0.02,
    max_height: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> LocoClient:
    """Run the hanger-to-stand sequence.

    Returns a LocoClient instance that is in FSM-200 and ready to receive
    Move / Velocity commands.
    """

    if logger is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger("hanger_boot")

    # DDS initialisation ---------------------------------------------------
    ChannelFactoryInitialize(0, iface)

    bot = LocoClient()
    bot.SetTimeout(10.0)
    bot.Init()

    # ------------------------------------------------------------------
    # Early-out: If the robot is already in a balanced stand (feet loaded
    # and balance controller running) there is no need to repeat the whole
    # hanger bring-up sequence.  Re-running it would at best waste time and
    # at worst jolt a robot that is happily standing on the ground.
    #
    # Our working definition of “balanced stand ready for walking” is:
    #     • FSM-ID 200  (Start – balance controller engaged)
    #     • SportModeState.mode != 2  (feet *loaded*)
    #
    # When this condition is met we simply log the situation and return the
    # initialised LocoClient so callers can proceed to send velocity
    # commands straight away.
    # ------------------------------------------------------------------

    try:
        cur_id = _fsm_id(bot)
        cur_mode = _fsm_mode(bot)

        if cur_id == 200 and cur_mode is not None and cur_mode != 2:
            logger.info(
                "Robot already in balanced stand (FSM 200, mode %s) – skipping boot sequence.",
                cur_mode,
            )

            # Leave the existing balance mode unchanged – if the operator
            # previously enabled continuous gait it will remain active; if
            # the robot was static it will stay still.  This avoids toggling
            # modes unnecessarily and prevents inadvertent “stepping in
            # place” when no motion is desired.

            return bot
    except Exception:
        # Fallback to the full sequence if any check fails (e.g. communication
        # hiccup right after power-up).  Better to run the safe, proven
        # routine than to guess incorrectly.
        pass

    def show(tag: str) -> None:
        logger.info("%-12s → FSM %s   mode %s", tag, _fsm_id(bot), _fsm_mode(bot))

    # 1. Damp --------------------------------------------------------------
    bot.Damp(); show("damp")

    # 2. Stand-up ----------------------------------------------------------
    bot.SetFsmId(4); show("stand_up")  # stand-up helper missing in wrapper

    # ------------------------------------------------------------------
    # 3. Increment stand-height until the firmware reports that the feet
    #    are *loaded* (SportModeState.mode == 0).  If we reach the maximum
    #    extension without seeing the transition 2 -> 0 it usually means
    #    the hanging frame is too high or too low and the soles never make
    #    solid contact with the ground.  In that case we pause, prompt the
    #    user to adjust the hanger height, then try the extension sweep
    #    again.
    # ------------------------------------------------------------------

    while True:  # retry loop – exits as soon as mode-0 (feet loaded) seen
        height = 0.0

        while height < max_height:
            height += step
            bot.SetStandHeight(height)
            show(f"height {height:.2f} m")

            # Break early once the firmware acknowledges feet contact
            if _fsm_mode(bot) == 0 and height > 0.2:
                break

        # Success condition – mode 0 means the robot is now supporting its
        # weight on the ground and we can continue to balance stand.
        if _fsm_mode(bot) == 0:
            break

        # Otherwise we failed to load the feet.  Tell the user and let them
        # tweak the hanging frame before we repeat the sweep.
        logger.warning(
            "Feet still unloaded (mode %s) after reaching %.2f m.\n"
            "Adjust hanger height (raise/lower until the soles are just in\n"
            "contact with the ground) then press <Enter> to try again…",
            _fsm_mode(bot),
            height,
        )

        try:
            # Reduce stand height so the operator can reposition safely.
            bot.SetStandHeight(0.0)
            show("reset")
        except Exception:
            pass

        input()  # wait for operator acknowledgement

    # 4. Balance stand -----------------------------------------------------
    bot.BalanceStand(0); show("balance")
    bot.SetStandHeight(height); show("height✔")

    # 5. Start the balance controller (FSM 200) ---------------------------
    # Leave the robot in balance-mode 0 (static) – callers can switch to
    # continuous gait (balance-mode 1) when they actually want to walk.

    bot.Start(); show("start")

    # Caller can now send velocity commands.
    return bot


__all__ = ["hanger_boot_sequence"]
