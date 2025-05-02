# Unitree G-1 – Locomotion FSM & Mode Cheat-Sheet

This document lists the most common finite-state-machine (FSM) IDs and *mode*
values you will see when working with the **loco** service on a Unitree G-1
robot via `unitree_sdk2py`.

## 1 . FSM IDs (robot state machine)

| ID  | Name / action                 | Remarks |
|----:|--------------------------------|---------|
| 0   | **Zero Torque**               | Motors off, gravity-droop allowed. |
| 1   | **Damp**                      | Motors apply viscous damping only – legs are “soft”. |
| 2   | Squat                         | Low squat posture (static). |
| 3   | Sit                           | Dog-sit (hips flexed). |
| 4   | Stand-up                      | Raises body to nominal height; used after Damp. |
| 200 | **Start** (balance / gait)    | Main balance controller & gait planner; enables walking. |
| 702 | Lie-to-Stand                  | From lying on the back. |
| 706 | Squat-to-Stand-up             | Smooth stand from deep squat. |

These IDs are the values you pass to `LocoClient.SetFsmId()` or that you see
returned by `ROBOT_API_ID_LOCO_GET_FSM_ID`.

Other IDs exist (tricks, arm tasks, etc.) but the ones above are the staples
needed for day-to-day bring-up and tele-operation.

## 2 . Mode (from `SportModeState_.mode`)

`SportModeState_` is published by the firmware and contains a single-byte
`mode` field that reflects the feet-contact state – very useful during the
transition from a hanging start-up to balanced stand.

| Mode | Meaning                                                     |
|-----:|-------------------------------------------------------------|
| 0    | Feet loaded, <strong>static stand</strong>.                |
| 1    | Feet loaded, <strong>dynamic / gait active</strong>.       |
| 2    | Feet **un-loaded** (hanging or airborne).                   |

Firmware may report additional modes (1, 3, …) but for bring-up you generally
Mode 1 is typically seen when the gait planner or other dynamic behaviours
are active (e.g. immediately after you send <code>Start</code> or while the
robot is commanded to walk).  Mode 0 means the robot is standing but not
trying to step.  Mode 2 remains the unmistakable “feet unloaded” condition.

## 3 . Typical bring-up sequence (quick recap)

1. `Damp`   → joints relaxed so you can align the feet.
2. `Stand-up` (FSM 4)   → robot’s internal routine extends legs part-way.
3. Increment `SetStandHeight` in small steps until `mode` flips 2 → 0.
4. `BalanceStand(0)` (or `SetBalanceMode(0)`).
5. Re-send the final `SetStandHeight` now that balance is engaged.
6. Optionally enable continuous gait (`SetBalanceMode(1)`) and finally `Start`
   (FSM 200) to walk.

With the cheat-sheet above you can interpret the debug output from
`quick_stand.py` (or your own scripts) and know exactly where in the state
machine the robot currently sits.

*Last updated: 2025-04-25*
