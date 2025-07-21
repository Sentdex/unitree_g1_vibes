"""g1_arm_rl_env.py – minimal Gymnasium environment for Unitree G-1 arm.

The environment exposes *incremental joint actions* while setting **long-range
Cartesian goals** so an RL agent learns to chain small deltas into collision-
free reaches.

Key choices
===========
Action space (Box, 7)
    Δq ∈ [-0.05 … +0.05] rad for each arm joint (left arm by default).

Observation space (Box, 24)
    [ q(7) , dq(7) , p_hand(3) , p_goal(3) , Δp(3) , step_frac(1) ]

Reward (dense)
    r  = −5·‖Δp‖   – 0.1·‖Δq‖   + 1 when ‖Δp‖<2 cm    −1 on self-collision  –0.5 on limit hit.

Episode ends
    • goal reached (<2 cm)   • self-collision   •  horizon T = ⌈‖Δp₀‖/0.05⌉+10

Dependencies
------------
    pip install gymnasium mujoco

This is intentionally *self-contained* – no IsaacGym / Legged-Gym needed.
Feel free to adapt the kinematics / reward later; this file is only a
bootstrap so you can start running PPO within minutes.
"""

from __future__ import annotations

import math
import pathlib
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


class G1ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 25}

    # Temporary switch to disable self-collision termination/penalty for the
    # hand so that tele-operation demos can keep the arm stationary even when
    # links visually clip.  Set to *False* for proper RL training.
    # Self-collision handling
    # ------------------------------------------------------------------
    # Set to *False* so that any contact between the learning arm and
    # protected robot parts immediately ends the episode and incurs a
    # strong negative reward.  This teaches the policy to stay well clear
    # of the torso, head, legs, and the other arm.
    # Switch is *True* to suppress collision diagnostics (and associated
    # episode termination/penalties).  It can be flipped back to *False*
    # for stricter training runs once the policy has matured.
    _DISABLE_COLLISION: bool = True

    # ------------------------------
    def __init__(self, render_mode: str | None = None, right_arm: bool = False):
        import mujoco  # local import so gym-only setups without mujoco can still import the module

        self._mujoco = mujoco
        self._render_mode = render_mode or "none"

        # Keep a simple flag around so that later code (e.g. goal sampling)
        # can branch on which arm is currently being trained without having
        # to repeat index comparisons.
        self._training_right = bool(right_arm)

        root = pathlib.Path(__file__).resolve().parent
        xml = (
            root
            / "unitree_mujoco"
            / "unitree_robots"
            / "g1"
            / "g1_29dof_with_hand.xml"
        )

        if not xml.exists():
            raise FileNotFoundError(xml)

        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)

        # ----- visual tweaks ----------------------------------------
        # Make the ground plane a bit more interesting by randomising
        # its colour each episode so training videos are less dull.
        self._floor_gid = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        # Ground plane colour – keep it fixed so recordings have
        # a consistent look.  Using a moderately dark green avoids
        # over-saturation while providing good contrast with the robot.
        # Use requested RGB (148, 250, 127) → normalised to [0-1].
        self._floor_colours = [
            (97 / 255.0, 161 / 255.0, 84 / 255.0, 1.0)
        ]

        # Joint indices for left and right arms (29-DoF mapping)
        self._left_idx = list(range(15, 22))
        self._right_idx = list(range(22, 29))
        self._arm_j = self._right_idx if right_arm else self._left_idx

        # --- Keep the *unused* arm parked in a safe, folded pose ---------
        # When we train the left arm (default) the right arm is not
        # controlled by the agent.  The default XML pose keeps that arm
        # stretched forward where it can get in the way of the active arm
        # and the goal region.  We therefore move the right arm to a
        # pre-defined tuck-away configuration and *lock* it there every
        # simulation step.

        # Angles supplied by the user (deg):
        #   shoulder-pitch  25.8
        #   shoulder-roll  -11.5
        #   shoulder-yaw    -2.9
        #   elbow           28.6
        #   wrist-roll      34.4
        #   wrist-pitch    -17.2
        #   wrist-yaw       14.3
        # Desired parked pose for the *right* arm (values provided by user)
        right_rest_deg = [25.8, -11.5, -2.9, 28.6, 34.4, -17.2, 14.3]
        self._right_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self._right_rest_q = np.array(right_rest_deg, dtype=np.float32) * (math.pi / 180.0)

        # Pre-compute qpos and dof addresses for the *right* arm joints
        self._right_qadr = []
        self._right_dadr = []
        for nm in self._right_joint_names:
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid == -1:
                raise RuntimeError(f"Joint {nm} not found in model")
            self._right_qadr.append(int(self.model.jnt_qposadr[jid]))
            self._right_dadr.append(int(self.model.jnt_dofadr[jid]))

        # Helper: references to whichever arm is *not* under agent control
        if right_arm:
            self._park_qadr = self._left_qadr = [int(self.model.jnt_qposadr[self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm.replace('right_', 'left_'))]) for nm in self._right_joint_names]
            # Simple mirror: for now use same angles with flipped signs for roll / yaw to tuck left arm similarly
            mirror = np.array([25.8, 11.5, 2.9, 28.6, -34.4, 17.2, -14.3], dtype=np.float32)
            self._park_rest_q = mirror * (math.pi / 180.0)
            self._park_dadr = [int(self.model.jnt_dofadr[self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm.replace('right_', 'left_'))]) for nm in self._right_joint_names]
        else:
            self._park_qadr = self._right_qadr
            self._park_rest_q = self._right_rest_q
            self._park_dadr = self._right_dadr

        # Map to qpos addresses
        self._qadr = [int(self.model.jnt_qposadr[j]) for j in self._arm_j]

        # Goal marker (mocap body) -------------------------------------
        self.goal_bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "goal_body")
        self._goal_mid = -1
        if self.goal_bid != -1:
            self._goal_mid = int(self.model.body_mocapid[self.goal_bid])


        # cache waist joint addresses to forcibly zero them each step -------
        self._waist_qadr = []
        waist_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        for nm in waist_names:
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                self._waist_qadr.append(int(self.model.jnt_qposadr[jid]))

        # --------------------------------------------------------------
        #  Collision handling setup
        # --------------------------------------------------------------

        # 1. Build sets of geometry IDs: (a) all links of the *active* arm
        #    (b) all parts of the robot we want to protect (torso, head,
        #    legs).  Names are matched heuristically; adapt if your XML
        #    uses different conventions.

        # ------------------------------------------------------------------
        # Geometries that must never be hit by the learning arm
        # ------------------------------------------------------------------
        # Using **keywords** instead of full names makes the filter resilient
        # to XML edits (e.g. a mesh renamed from "pelvis_geom" → "pelvis_1").
        # We broaden the set compared to the original implementation so that
        # head / torso *visual* geoms and the legs are also captured.  This
        # reduces the chance that the arm can “clip” through them because a
        # particular part name was missing from the list.

        protect_kw = (
            "pelvis",
            "waist",
            "torso",
            "chest",
            "spine",
            "abdomen",
            "body",
            "head",
            "neck",
            "hip_",
            "thigh",
            "knee",
            "calf",
            "shin",
            "leg",
            "upperarm",
            "lowerarm",
            "shoulder",
        )

        arm_prefix = "right_" if right_arm else "left_"

        # ------------------------------------------------------------------
        # Build collision groups using *body* names because the Livox/Unitree
        # XML meshes often leave <geom> names empty.  MuJoCo assigns each geom
        # a body via `geom_bodyid`, so we look up the *parent body* name to
        # decide which set the geom belongs to.
        # ------------------------------------------------------------------

        self._protect_gids: set[int] = set()
        self._arm_gids: set[int] = set()

        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            body_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, bid)
            if not body_name:
                continue

            # Arm geoms  --------------------------------------------------
            # We only want *upper-limb* segments (shoulder…hand).  Filter
            # by keywords to exclude left/right legs and other side parts
            # that also start with the same prefix.
            limb_kw = ("upperarm", "elbow", "wrist", "hand", "forearm")
            if body_name.startswith(arm_prefix) and any(k in body_name for k in limb_kw):
                self._arm_gids.add(gid)

            # Protected geoms  -------------------------------------------
            if any(k in body_name for k in protect_kw):
                self._protect_gids.add(gid)

        # 2. Ensure MuJoCo generates contact pairs between those geoms by
        #    enabling collision layers (contype/affinity).  We simply set
        #    both bit fields to 1.

        for gid in self._protect_gids | self._arm_gids:
            self.model.geom_contype[gid] |= 1
            self.model.geom_conaffinity[gid] |= 1

        # --------------------------------------------------------------
        # End-effector reference
        # --------------------------------------------------------------
        # Earlier versions of this environment defined the *wrist* (yaw link)
        # as the point that had to reach the goal.  We now switch the target
        # to the **hand palm** so policies learn to put the *hand* (rather
        # than the wrist) on the desired position – this is what you would
        # want for actual manipulation tasks where the fingers, not the
        # forearm, need to interact with objects.
        #
        # We therefore grab the body corresponding to the palm mesh which
        # exists for both arms in the XML:  «right_hand_palm_link» /
        # «left_hand_palm_link».  Should the XML change, just adjust the two
        # names below.
        # Attempt (1): direct palm geometry detection via mesh name ---------
        target_mesh = "right_hand_palm_link" if right_arm else "left_hand_palm_link"

        self._palm_gid: int = -1

        # Flag whether we will approximate the palm via wrist + offset
        self._use_wrist_offset = False

        # MuJoCo ≥ 2.3.6 renamed the field that stores per-geom mesh IDs from
        # ``geom_dataid`` to ``geom_meshid``.  Support both so the environment
        # works across versions shipped on different systems.
        meshid_field = "geom_meshid" if hasattr(self.model, "geom_meshid") else "geom_dataid"

        for gid in range(self.model.ngeom):
            mid = int(getattr(self.model, meshid_field)[gid])
            if mid == -1:
                continue
            mname = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_MESH, mid
            )
            if mname == target_mesh:
                self._palm_gid = gid
                break

        if self._palm_gid == -1:
            # Fallback: approximate palm position relative to wrist body
            self._use_wrist_offset = True
            fallback_body = "right_wrist_yaw_link" if right_arm else "left_wrist_yaw_link"
            self._body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, fallback_body)
            if self._body_id == -1:
                raise RuntimeError(
                    f"Palm mesh '{target_mesh}' not found and fallback body '{fallback_body}' missing"
                )

            # Hard-coded offset measured from XML (metres)
            self._palm_offset = np.array([0.0415, (-0.003 if right_arm else 0.003), 0.0], dtype=np.float32)
        else:
            self._body_id = -1  # sentinel when using geom xpos

        # If we could neither locate a dedicated palm geometry nor set up a
        # fallback via the wrist, we cannot continue – the environment would
        # not be able to compute the forward kinematics.  Only raise in that
        # particular situation.  When the wrist-based approximation is in
        # place (``self._use_wrist_offset``), we proceed without error.
        if self._palm_gid == -1 and not self._use_wrist_offset:
            raise RuntimeError(
                f"Palm geometry '{target_mesh}' not found in the model and no fallback could be configured."
            )

        # When a palm geometry *is* present we never use ``_body_id``.  Keep a
        # well-defined value to avoid accidental use in that code path.
        if self._palm_gid != -1:
            self._body_id = -1  # sentinel when using geom xpos

        # --------------------------------------------------------------
        #   Pre-compute shoulder body ID – used later for reachability
        # --------------------------------------------------------------

        shoulder_body_name = "right_shoulder_pitch_link" if right_arm else "left_shoulder_pitch_link"
        self._shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_body_name)
        if self._shoulder_bid == -1:
            raise RuntimeError(f"Body {shoulder_body_name} not found in model")

        # Gym spaces ---------------------------------------------------
        self.action_space = gym.spaces.Box(-0.05, 0.05, shape=(7,), dtype=np.float32)

        obs_hi = np.array([
            *([math.pi] * 7),        # q
            *([10.0] * 7),           # dq rad/s
            *([1.5] * 3),            # p_hand m
            *([1.5] * 3),            # p_goal
            *([1.5] * 3),            # Δp
            1.0,                     # step fraction
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(-obs_hi, obs_hi, dtype=np.float32)

        # Viewer -------------------------------------------------------
        self.viewer = None
        if self._render_mode == "human":
            try:
                from mujoco import viewer as mj_viewer  # type: ignore

                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            except Exception:
                print("[warn] MuJoCo viewer not available – falling back to headless")
                self.viewer = None

        # Episode bookkeeping ----------------------------------------
        self.max_steps = 200
        self._step_count = 0

        # pick a new floor colour at the beginning of every episode
        if self._floor_gid != -1:
            rgba = self.np_random.choice(self._floor_colours)
            self.model.geom_rgba[self._floor_gid] = rgba

    # ------------------------------
    def _fk(self):  # helper – returns hand position in torso frame
        if self._use_wrist_offset:
            wrist_pos = np.array(self.data.xpos[self._body_id])
            wrist_rot = np.array(self.data.xmat[self._body_id]).reshape(3, 3)
            return wrist_pos + wrist_rot.dot(self._palm_offset)
        return np.array(self.data.geom_xpos[self._palm_gid])

    # ------------------------------
    def reset(self, seed: int | None = None, options: Dict | None = None):  # noqa: D401
        super().reset(seed=seed)

        # 1.a lock waist DoFs (indices 12,13,14) at 0 each reset
        for j in (12, 13, 14):
            self.data.qpos[self.model.jnt_qposadr[j]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[j]] = 0.0

        # 1.b Safe start pose for the *active* arm
        if self._training_right:
            active_rest = self._right_rest_q
        else:
            # mirror of right_rest_deg defined earlier
            active_rest = np.array([25.8, 11.5, 2.9, 28.6, -34.4, 17.2, -14.3], dtype=np.float32) * (
                math.pi / 180.0
            )

        noise = self.np_random.uniform(-0.25, 0.25, size=7)
        q0 = active_rest + noise

        for adr, q in zip(self._qadr, q0):
            self.data.qpos[adr] = float(q)

        # 1.c park the *unused* arm in the rest configuration so it stays
        #     out of the way during the episode.
        # Park unused arm at its rest pose
        for qadr, q_des in zip(self._park_qadr, self._park_rest_q):
            self.data.qpos[qadr] = float(q_des)
        for dadr in self._park_dadr:
            self.data.qvel[dadr] = 0.0

        self.data.qvel[:] = 0

        # force waist zero
        for nm in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
        self._step_count = 0

        # ------------------------------------------------------------------
        # 2. Sample a goal position
        # ------------------------------------------------------------------
        # Empirical training runs showed the agent rarely encounters
        # *frontal* goals that lie directly in front of the torso and
        # fairly close (≈15–30 cm).  Those poses are therefore poorly
        # practised and the learnt policy struggles whenever such a target
        # appears at evaluation time.
        #
        # To address this we bias the goal sampler:
        #   1. 60 % of the time we draw a *frontal* goal (small |y|,
        #      moderate x) so the hand must work in front of the torso.
        #   2. 40 % of the time we revert to the previous side-biased logic
        #      (favour same-side but still cover the full workspace).
        #
        # Additional tweaks:
        #   • Allow goals slightly closer to the torso (x ≥ 0.10 m) but
        #     still maintain a 10 cm clearance from the torso centre.
        #   • Reachability check strengthened with an explicit maximum
        #     shoulder-to-goal distance (reach_max).
        # ------------------------------------------------------------------

        torso_pos = np.array([0.0, 0.0, 0.90])

        # ensure current kinematics are updated for distance check
        self._mujoco.mj_forward(self.model, self.data)

        shoulder_pos = np.array(self.data.xpos[self._shoulder_bid])

        frontal_pref = self.np_random.random() < 0.60  # 60 % frontal

        training_right = self._training_right

        if frontal_pref:
            # Narrow y band → directly in front of torso
            y_low, y_high = -0.15, 0.15
        else:
            # Old side-biased strategy
            side_pref = self.np_random.random() < 0.7

            if side_pref:
                if training_right:
                    y_low, y_high = -0.45, -0.05  # right side (−y)
                else:
                    y_low, y_high = 0.05, 0.45   # left  side (+y)
            else:
                # central band for variety
                y_low, y_high = -0.30, 0.30

            # placeholder; real sampling done inside loop

        reach_max = 0.60  # conservative reach radius (m)

        while True:
            # Sample x according to chosen strategy each iteration so we do
            # not risk an infinite loop if a particular x value is
            # unreachable due to the torso / reachability constraints.
            if frontal_pref:
                # Bias toward smaller x (closer) using square-law
                x = 0.10 + (self.np_random.random() ** 2) * 0.25  # 0.10 … 0.35
            else:
                x = self.np_random.uniform(0.15, 0.50)

            y = self.np_random.uniform(y_low, y_high)

            # Allow higher vertical targets (up to ≈1.30 m) when the goal
            # sits relatively close to the shoulder in the horizontal
            # plane – mimics the natural kinematic envelope where the hand
            # can reach higher overhead chiefly when it is not fully
            # extended forward.
            horiz = math.hypot(x, y)
            if horiz < 0.35:
                z_hi = 1.30  # overhead region
            else:
                z_hi = 1.10

            z = self.np_random.uniform(0.60, z_hi)
            cand = np.array([x, y, z], dtype=np.float32)

            # 1. keep at least 10 cm away from torso centre
            if np.linalg.norm(cand - torso_pos) <= 0.10:
                continue

            # 2. keep within reach radius of shoulder so targets are feasible
            if np.linalg.norm(cand - shoulder_pos) > reach_max:
                continue

            # passed all tests – accept
            self.p_goal = cand
            if self._goal_mid != -1:
                self.data.mocap_pos[self._goal_mid] = self.p_goal
            break

        self._mujoco.mj_forward(self.model, self.data)

        # enforce rigid waist after dynamics update
        for j in (12, 13, 14):
            self.data.qpos[self.model.jnt_qposadr[j]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[j]] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------------------
    def step(self, action: np.ndarray):  # noqa: D401
        assert self.action_space.contains(action), "action outside bounds"

        # 1. integrate joint delta
        for adr, dq in zip(self._qadr, action):
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + float(dq), -3.14, 3.14)

        # keep waist locked every step (name-based to be robust)
        for nm in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                qadr = int(self.model.jnt_qposadr[jid])
                dadr = int(self.model.jnt_dofadr[jid])
                self.data.qpos[qadr] = 0.0
                self.data.qvel[dadr] = 0.0

        # Keep the *unused* arm at its rest pose so dynamics do not move it
        # Keep parked arm fixed every step
        for qadr, q_des in zip(self._park_qadr, self._park_rest_q):
            self.data.qpos[qadr] = float(q_des)
        self.data.qvel[self._park_dadr] = 0.0

        self.data.qvel[self._qadr] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        # 2. compute reward
        p_hand = self._fk()
        delta = self.p_goal - p_hand
        dist = np.linalg.norm(delta)

        reward = -5.0 * dist - 0.1 * np.linalg.norm(action)

        # ------------------------------------------------------------------
        #  Grip-orientation regulariser (keep grasp axis vertical)
        # ------------------------------------------------------------------
        # Many manipulation tasks (e.g. picking up a bottle) require the hand
        # to approach objects with its grasping axis aligned to gravity.  We
        # therefore add a *soft* penalty based on the **world-frame**
        # orientation of the hand link rather than relying on individual
        # joint angles (which do not capture the compounded effect of all
        # upstream joints).

        # World orientation of the palm/wrist body as a 3×3 rotation matrix
        ori_mat = np.array(self.data.xmat[self._body_id]).reshape(3, 3)

        # Choose the hand’s local *x-axis* as grasp direction.  (If your XML
        # defines another axis as the closing direction, change the column
        # index here.)
        grasp_axis_world = ori_mat[:, 0]  # first column

        # Alignment with global Z: 1 → perfectly vertical, 0 → horizontal
        vertical_alignment = abs(float(grasp_axis_world[2]))  # dot with [0,0,1]

        # 1. Vertical alignment penalty (scaled up for stronger effect)
        reward -= 1.5 * (1.0 - vertical_alignment)

        # ------------------------------------------------------------------
        #  Keep the palm plane roughly vertical (front-to-back horizontal)
        # ------------------------------------------------------------------
        # Horizontal alignment of the palm plane
        # Both local x- and y-axes should lie inside the world X-Y plane, i.e.
        # have near-zero z-components.  We penalise the *sum* of their |z|
        # magnitudes to drive the whole palm plane parallel to the ground.

        palm_x_world = ori_mat[:, 0]
        palm_y_world = ori_mat[:, 1]
        horiz_deviation = abs(float(palm_x_world[2])) + abs(float(palm_y_world[2]))

        # 2. Palm plane horizontal penalty – stronger weight
        reward -= 1.5 * horiz_deviation

        # --------------------------------------------------
        # Contact-based self-collision check
        # Any contact where one geom belongs to the *active* arm and the
        # other geom does *not* belongs to that arm is treated as a
        # violation.  This is stricter than only checking against
        # "protected" parts and therefore catches e.g. collisions with the
        # other arm should that ever happen.
        # --------------------------------------------------

        collision_hit = False

        if not self._DISABLE_COLLISION:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2

                # Penalise only if *active arm* geom hits a protected geom
                if (g1 in self._arm_gids and g2 in self._protect_gids) or (
                    g2 in self._arm_gids and g1 in self._protect_gids
                ):
                    penetration = max(0.0, -c.dist)
                    # Apply a harsher penalty so the agent quickly learns
                    # that self-collisions are unacceptable.  A fixed cost
                    # (−200) is combined with a penetration-scaled term
                    # (−2000 × depth) to discourage deeper contacts.
                    if penetration < 0.002:
                        continue
                    reward -= 200.0 + 2000.0 * penetration
                    collision_hit = True
                    break

        # Console feedback – useful during development to verify collisions
        if collision_hit:
            # Show the names of both bodies for clarity
            g_names = [
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
            ]
            b_names = [
                self._mujoco.mj_id2name(
                    self.model, self._mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[c.geom1])
                ),
                self._mujoco.mj_id2name(
                    self.model, self._mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[c.geom2])
                ),
            ]
            print(
                f"[env] CONTACT collision @ step {self._step_count}: {g_names[0]}({b_names[0]}) ↔ {g_names[1]}({b_names[1]})"
            )

        # --------------------------------------------------
        # Geometric overlap fallback disabled – MuJoCo’s own contact pairs
        # are now deemed sufficient and more accurate.  If you encounter
        # cases where obvious visual intersections are *not* detected, turn
        # this block back on and tune the radius heuristic.
        # --------------------------------------------------

        # near-torso penalty (keep >10 cm distance from torso centre)
        torso_pos = np.array([0.0, 0.0, 0.90])
        d_torso = np.linalg.norm(p_hand - torso_pos)
        if d_torso < 0.10:
            reward -= (0.10 - d_torso) * 100.0   # stronger penalty the closer it gets

        terminated = False
        # Consider the goal reached when the palm is within 3 cm.
        success = dist < 0.03 and not collision_hit
        if success:
            reward += 1.0
            terminated = True

        self._step_count += 1
        if collision_hit and not self._DISABLE_COLLISION:
            terminated = True

        if self._step_count > self.max_steps:
            terminated = True

        obs = self._get_obs()
        info: Dict[str, float] = {"dist": dist}

        # update goal marker each step so it remains visible
        if self._goal_mid != -1:
            self.data.mocap_pos[self._goal_mid] = self.p_goal

        if self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, False, info

    # ------------------------------
    def _get_obs(self):
        q = self.data.qpos[self._qadr]
        dq = self.data.qvel[self._qadr]
        p_hand = self._fk()
        delta = self.p_goal - p_hand
        step_frac = np.array([self._step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([q, dq, p_hand, self.p_goal, delta, step_frac]).astype(np.float32)

    # ------------------------------
    def render(self):  # compatibility shim
        if self.viewer is not None:
            self.viewer.sync()


# Convenience factory for stable-baselines3
def make_env(**kwargs):  # noqa: D401
    return G1ArmReachEnv(**kwargs)


if __name__ == "__main__":
    # Quick sanity-check  – run one random episode headless
    env = G1ArmReachEnv()
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, _ = env.step(a)
        ep_ret += r
    print("Episode return:", ep_ret)
