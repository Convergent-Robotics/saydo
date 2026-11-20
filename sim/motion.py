"""Collision-aware motion for the Franka: cuRobo plans, the arm follows the plan as joint targets.

cuRobo (NVlabs/curobo, v0.7.8, the last version with the MotionGen API; sim/setup_brev.sh builds
it) plans smooth, collision-free joint trajectories on the GPU, for the arm and for what it holds,
round a world of boxes and meshes. Isaac Sim 5.1 ships no planner of its kind (its docs point to
cuRobo; Lula RRT is in it but plans no held object and gives jerky paths). cuRobo's main branch
(v0.8, "V2") needs warp 1.18, and Isaac Sim 5.1's own core breaks on any warp but its 1.8.2
(measured 21 Oct 2026: isaacsim.core.utils.warp fails to import): v0.7.8 runs on 1.8.2.

The world is read off the stage before each plan (`refresh`): every collision prim near the robot
as a box fitted to it in its own frame, except an enclosure's parts, which go in as meshes (a
fridge's body is one concave mesh round its cavity, and its box would fill the cavity). The held
object is not in the world: it is attached to the hand (`attach`), and the plan keeps it clear too.

Measured 21 Oct 2026, task 17's scene: cuRobo's forward kinematics of Isaac's joint positions put
`panda_hand` where Isaac has it to 0.1 mm, same orientation; LabUtopia's gripper position (the
tool centre, between the fingertips) is 0.1034 m along the hand's z; a plan followed as one joint
position target per physics step ends within 1.5 mm of its goal.

Import only after SimulationApp exists.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R, Slerp

from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

from curobo.geom.types import Cuboid, Mesh, WorldConfig  # noqa: E402
from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.rollout.cost.pose_cost import PoseCostMetric  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.state import JointState  # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # noqa: E402
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # noqa: E402

TCP_M = 0.1034        # tool centre ahead of panda_hand, along its z (measured, above)
NEAR_M = 2.0          # obstacles further than this from the robot's base are left out
OBB_CACHE = 400     # the planner's room for collision boxes (cuboids); _world raises past it
SLOW = 0.25           # cuRobo's trajectories slowed to this fraction of the Franka's limits. At 0.5 the
                      # joints lagged their targets by up to 14 degrees on long moves (7 at 0.25), and a
                      # carried beaker cut a corner into its neighbour (21 Oct 2026)
ATTACHED_SPHERES = 40 # collision spheres fitted to a held object
LINE_OFF_M = 0.03     # a free plan stands in for a straight line only if the tool stays this close to it
LINE_TURN_DEG = 10    # and the hand turns no more than this beyond the goal's own turn
LINE_TRIES = 4        # free plans tried for a line before it gives up
IK_TRIES = 3  # cuRobo's IK starts from random seeds: a pose at the edge of reach is found one solve in two (the knife in the toy box, 7 Nov 2026), so a failed solve is tried again
STRAIGHT_STEP_M = 0.01        # a straight path solves the arm's pose every centimetre
STRAIGHT_JUMP_RAD = 0.3       # and gives up if a joint jumps more than this between two of them
STRAIGHT_RAD_PER_STEP = 0.006 # its fastest joint turns at most this per physics step (0.36 rad/s)
TURN_STEP_DEG = 3              # a turn about a point (a pour's tip) solves the arm's pose every 3 degrees
SETTLE_RAD = 0.005    # follow holds a plan's last pose until every joint is this close to it
SETTLE_STEPS = 120    # or this many physics steps have passed
SNAP_RAD = 0.035      # an arm within this of the pose its last plan ended in (2 degrees) plans from that pose


class Motion:
    def __init__(self, bench, physics_dt):
        self.b = bench
        self.base = np.array(bench.cfg["robot"]["position"], float)
        self.dt = physics_dt
        self.tensor = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        # The base link stands in the bench top (the robot is mounted at bench height): never a collision.
        kin = robot_cfg["kinematics"]
        kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n != "panda_link0"]
        # Spheres for a held object: franka.yml allows 4, too coarse for a 15 cm beaker.
        kin["extra_collision_spheres"] = {"attached_object": ATTACHED_SPHERES}
        self.ignore = set()
        world = self._world()
        self.mg = MotionGen(MotionGenConfig.load_from_robot_config(
            robot_cfg, world, self.tensor, interpolation_dt=physics_dt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": OBB_CACHE, "mesh": 12}))
        self.mg.warmup()
        self.names = self.mg.kinematics.joint_names
        lim = self.mg.kinematics.get_joint_limits().position.cpu().numpy()
        self.limits = (lim[0], lim[1])
        dofs = bench.robot.dof_names
        self.idx = [dofs.index(n) for n in self.names]
        self.attached = None
        self.followed_to = None  # the last pose a followed plan ended in (follow), for joints()

    # ---- the world ----

    def _world(self, ignore=(), only=None):
        """Every collision prim near the robot, as cuRobo obstacles in the robot's base frame. Left
        out: the robot, the room shell, the held object, the objects in `ignore` (what the hand is
        about to touch) and anything far off (hidden props are parked 10 m away). With `only` (an
        object's name), that object's prims alone, wherever it is."""
        b = self.b
        skip_roots = [b.robot.prim_path, "/World/spai_room", "/World/spai_liquid"]
        for n in set(ignore) | ({b.held_object} if b.held_object and only is None else set()):
            skip_roots.append(b.objects[n])
        boxes = {n: b.objects[n] for n in b.enclosures}
        # Guides too: the kitchen's colliders are unrendered boxes (sim/import_meshes.py).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        cuboids, meshes = [], []
        for prim in b.stage.Traverse():
            path = str(prim.GetPath())
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsActive():
                continue
            if any(path == r or path.startswith(r + "/") for r in skip_roots) or "/spai_look" in path:
                continue
            if only is not None and not (path == b.objects[only] or path.startswith(b.objects[only] + "/")):
                continue
            # A look shown over a vessel's own meshes (World._relook) is only seen: left out of its bounds.
            looks = [c.GetPath() for c in prim.GetChildren() if c.GetName() == "spai_look"]
            bbox = (cache.ComputeWorldBoundWithOverrides(
                prim, looks, UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), {}) if looks
                    else cache.ComputeWorldBound(prim))
            rng = bbox.GetRange()
            if rng.IsEmpty():
                continue
            m = bbox.GetMatrix()
            centre = np.array(m.Transform(rng.GetMidpoint()))
            if np.linalg.norm(centre[:2] - self.base[:2]) > NEAR_M:
                continue
            name = path.replace("/", "_")
            in_box = any(path.startswith(r + "/") for r in boxes.values())
            if in_box and prim.IsA(UsdGeom.Mesh):
                meshes.append(self._mesh(name, prim))
                continue
            rows = np.array([[m[i][j] for j in range(3)] for i in range(3)])
            scale = np.linalg.norm(rows, axis=1)
            dims = np.maximum(np.array(rng.GetSize()) * scale, 0.002)
            x, y, z, w = R.from_matrix((rows / scale[:, None]).T).as_quat()
            cuboids.append(Cuboid(name=name, pose=[*(centre - self.base), w, x, y, z], dims=dims.tolist()))
        # Past the cache, cuRobo plans with the world cut short and reports only ik_fail: the drying mat's
        # props (18 Nov 2026, 120) and the candle scene's clutter (19 Nov, 240; 268 boxes) did that.
        if len(cuboids) > OBB_CACHE:
            raise RuntimeError(f"{len(cuboids)} collision boxes near the arm, over the planner's cache of "
                               f"{OBB_CACHE}: raise OBB_CACHE in sim/motion.py")
        return WorldConfig(cuboid=cuboids, mesh=meshes)

    def _mesh(self, name, prim):
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get(), float)
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = pts @ m[:3, :3] + m[3, :3] - self.base
        counts, idx = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        faces, k = [], 0
        for c in counts:  # fan-triangulate quads and n-gons
            faces += [[idx[k], idx[k + i], idx[k + i + 1]] for i in range(1, c - 1)]
            k += c
        return Mesh(name=name, pose=[0, 0, 0, 1, 0, 0, 0], vertices=pts.tolist(), faces=faces)

    def refresh(self, ignore=()):
        """Read the world off the stage again: objects move."""
        self.mg.update_world(self._world(ignore))

    # ---- state ----

    def joints(self):
        """The arm's joint state now, as a start for cuRobo. Within SNAP_RAD of the pose the last
        followed plan ended in, that pose: the arm stops a little short of a plan's end, and where
        it stops can lie inside cuRobo's self-collision margin though the plan's end did not. It
        then refused to plan on from there (invalid_start_state_self_collision): task 17's pick
        failed at the grasp, in 2 of 4 scripted replays, 27 Oct 2026. Otherwise the measured pose,
        clipped into cuRobo's joint limits: a followed plan can end a hair past one (the wrist
        after a side grasp), and cuRobo refuses that too (invalid_start_state_joint_limits,
        21 Oct 2026)."""
        q = self.b.robot.get_joint_positions()
        q = np.array([q[i] for i in self.idx])
        if self.followed_to is not None and np.max(np.abs(q - self.followed_to)) < SNAP_RAD:
            q = self.followed_to
        q = np.clip(q, self.limits[0] + 1e-3, self.limits[1] - 1e-3)
        return JointState.from_position(self.tensor.to_device(q).view(1, -1), joint_names=self.names)

    def hand_goal(self, tool, rot):
        """A goal for panda_hand (cuRobo's end effector) from the tool centre's world position and the
        hand's world rotation (scipy Rotation; z the approach axis, y the fingers' closing line)."""
        z = rot.apply([0, 0, 1])
        p = np.asarray(tool, float) - TCP_M * z - self.base
        x, y, zq, w = rot.as_quat()
        return Pose(position=self.tensor.to_device([p.tolist()]), quaternion=self.tensor.to_device([[w, x, y, zq]]))

    def hand_rotations(self, traj):
        """The hand's world rotation at each row of a joint trajectory (n, 7)."""
        st = self.mg.kinematics.get_state(self.tensor.to_device(traj))
        q = st.ee_quaternion.cpu().numpy()
        return R.from_quat(q[:, [1, 2, 3, 0]])

    # ---- planning ----

    def line(self, tool, rot, hold, ignore=()):
        """A short straight move (`hold` "axis" or "vertical"), or, if cuRobo finds no plan that
        holds the line, a free plan to the same pose that stays near the line: the tool within
        LINE_OFF_M of it, the hand turning at most LINE_TURN_DEG more than the goal asks. A free
        plan can wind the arm round through another configuration to the same pose (task 17,
        22 Oct 2026: 956 steps to lower a beaker 3 cm, turning it over and knocking the fridge
        door from 44 to 110 degrees open); plans differ from call to call, so a few are tried."""
        traj, why = self.plan(tool, rot, hold=hold, ignore=ignore)
        if traj is not None:
            return traj, None
        traj, why2 = self.straight(tool, rot, ignore=ignore)
        if traj is not None:
            return traj, None
        self.b.note(f"line: no straight plan ({why}; {why2}); free plan")
        for _ in range(LINE_TRIES):
            traj, why = self.plan(tool, rot, ignore=ignore)
            if traj is None:
                break
            off, turn = self.stray(traj, tool, rot)
            if off <= LINE_OFF_M and turn <= LINE_TURN_DEG:
                return traj, None
            why = f"every free plan strays ({off * 100:.0f} cm off the line, hand turned {turn:.0f} deg more)"
        self.b.note(f"line: {why}")
        return None, why

    def stray(self, traj, tool, rot):
        """How far a trajectory strays from the straight line between its start and `tool` (the
        tool centre's furthest distance from it, m) and how much more the hand turns on the way
        than it does from start to `rot` (degrees)."""
        rots = self.hand_rotations(traj)
        hand = self.mg.kinematics.get_state(self.tensor.to_device(traj)).ee_position.cpu().numpy() + self.base
        tips = hand + TCP_M * rots.apply([0, 0, 1])
        a, d = tips[0], np.asarray(tool, float) - tips[0]
        t = np.clip((tips - a) @ d / max(float(d @ d), 1e-9), 0, 1)
        off = float(np.linalg.norm(tips - (a + t[:, None] * d), axis=1).max())
        turned = np.degrees((rots * rots[0].inv()).magnitude()).max()
        asked = np.degrees((rot * rots[0].inv()).magnitude())
        return off, float(turned - asked)

    def plan(self, tool, rot, hold=None, ignore=(), attempts=6, _retry=True):
        """A joint trajectory (n, 7) to put the tool centre at `tool` with the hand at `rot`, or
        (None, why). `hold`: None (free), "axis" (a straight line along the hand's own z, its
        orientation held: in to a grasp, back out) or "vertical" (straight up or down, orientation
        held). The world is read again first, without `ignore`. Where the arm starts touching
        something, cuRobo plans nothing at all (invalid_start_state_world_collision): once more,
        without what is at the hand. A put-down lowered directly into the toy box left the hand on
        its wall, and every call after it failed so (the knife task, replay-1112, 13 Nov 2026)."""
        self.refresh(ignore)
        metric = None
        if hold:
            # "axis", "vertical": the orientation and two of the three position axes held; free,
            # the goal's z (the hand's axis) or the base frame's z. "orientation": the orientation
            # only. cuRobo checks that start and goal agree on what is held
            # (MotionGen.update_pose_cost_metric): it holds the whole orientation or none of it,
            # so "upright, any heading" cannot be asked for (Bench._carry works round that).
            weights = [1, 1, 1, 0, 0, 0] if hold == "orientation" else [1, 1, 1, 1, 1, 0]
            metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=self.tensor.to_device(weights),
                                    project_to_goal_frame=(hold == "axis"))
        cfg = MotionGenPlanConfig(max_attempts=attempts, time_dilation_factor=SLOW, pose_cost_metric=metric,
                                  enable_graph_attempt=None if hold else 3)
        r = self.mg.plan_single(self.joints(), self.hand_goal(tool, rot), cfg)
        if not r.success.item():
            why = str(r.status).split(".")[-1].lower()
            if why == "invalid_start_state_world_collision" and _retry:
                near = set(self.b.at_hand()) - set(ignore)
                if near:
                    self.b.note(f"plan: the arm starts touching; planned without {sorted(near)}")
                    return self.plan(tool, rot, hold, set(ignore) | near, attempts, _retry=False)
            return None, why
        return r.get_interpolated_plan().position.cpu().numpy().reshape(-1, len(self.names)), None

    def reachable(self, tool, rot, ignore=()):
        """Whether the arm can put the tool centre at `tool`, the hand at `rot`, without touching
        anything but `ignore`: an inverse kinematics solve, no path."""
        self.refresh(ignore)
        goal = self.hand_goal(tool, rot)
        return any(bool(self.mg.ik_solver.solve_single(goal).success.item()) for _ in range(IK_TRIES))

    def straight(self, tool, rot, ignore=(), step=STRAIGHT_STEP_M):
        """A straight line for the tool centre from where it is to `tool`, the hand turning evenly
        to `rot` on the way: inverse kinematics every `step` metres, each solved from the last
        (the arm keeps its configuration), slowed in and out with half a cosine, then checked for
        collisions. cuRobo plans no such path: its held orientation is a cost it trades away (a
        carried beaker went 175 degrees over, task 17, 22 Oct 2026). (traj, None) or (None, why)."""
        q = self.joints().position.cpu().numpy().reshape(-1)
        rots = self.hand_rotations(q[None])
        start = self.b.robot.get_gripper_position()
        tool = np.asarray(tool, float)
        n = max(2, int(np.ceil(np.linalg.norm(tool - start) / step)) + 1)
        turn = Slerp([0, 1], R.concatenate([rots[0], rot]))
        self.refresh(ignore)
        rows = [q]
        for f in np.linspace(0, 1, n)[1:]:
            q_next = self._ik_from(start + f * (tool - start), turn([f])[0], rows[-1])
            if q_next is None:
                return None, f"no arm pose {f * 100:.0f}% of the way along"
            if np.abs(q_next - rows[-1]).max() > STRAIGHT_JUMP_RAD:
                self.b.note(f"straight: joints {np.round(rows[-1], 2)} -> {np.round(q_next, 2)}, "
                            f"limits {np.round(self.limits[0], 2)} {np.round(self.limits[1], 2)}")
                return None, f"the arm would turn over {f * 100:.0f}% of the way along"
            rows.append(q_next)
        traj = self._eased(np.array(rows))
        if not self.clear(traj, ignore):
            return None, "a straight line would hit something"
        return traj, None

    def turn_about(self, pivot, axis, angle, ignore=(), q=None, tool=None, rot=None, step_deg=TURN_STEP_DEG):
        """The hand, and what it holds, turned `angle` radians about the line through `pivot` along
        `axis`: inverse kinematics every `step_deg`, each solved from the last, eased and checked
        for collisions, as straight() does a line. From the arm as it is, or from joints `q` with
        the tool centre at `tool` and the hand at `rot` (to try a turn before the arm gets there).
        A pour's tip: RMPflow, which turned the hand about a point before (_tip), drifts and
        reaches no further than it can, and a tip that ended beside the target poured nothing
        (task 27, 3 of 5 replays, 27 Oct 2026). (traj, None) or (None, why)."""
        if q is None:
            q = self.joints().position.cpu().numpy().reshape(-1)
            tool, rot = self.b.robot.get_gripper_position(), self.hand_rotations(q[None])[0]
        pivot, axis = np.asarray(pivot, float), np.asarray(axis, float) / np.linalg.norm(axis)
        self.refresh(ignore)
        rows = [np.asarray(q, float)]
        n = max(2, int(np.ceil(np.degrees(abs(angle)) / step_deg)))
        for f in np.linspace(0, 1, n + 1)[1:]:
            turn = R.from_rotvec(axis * angle * f)
            q_next = self._ik_from(pivot + turn.apply(np.asarray(tool, float) - pivot), turn * rot, rows[-1])
            if q_next is None:
                return None, f"no arm pose {np.degrees(angle * f):.0f} degrees over"
            if np.abs(q_next - rows[-1]).max() > STRAIGHT_JUMP_RAD:
                return None, f"the arm would turn over at {np.degrees(angle * f):.0f} degrees"
            rows.append(q_next)
        traj = self._eased(np.array(rows))
        if not self.clear(traj, ignore):
            return None, "the turn would hit something"
        return traj, None

    def _eased(self, rows):
        """Joint waypoints as a trajectory: even steps of at most STRAIGHT_RAD_PER_STEP for the
        furthest-moving joint, slowed in and out with half a cosine."""
        k = max(2, int(np.abs(np.diff(rows, axis=0)).max(axis=1).sum() / STRAIGHT_RAD_PER_STEP))
        s = (1 - np.cos(np.pi * np.linspace(0, 1, int(k * np.pi / 2)))) / 2 * (len(rows) - 1)
        i = np.minimum(s.astype(int), len(rows) - 2)
        return rows[i] + (s - i)[:, None] * (rows[i + 1] - rows[i])

    def _ik_from(self, tool, rot, q):
        """Joints near `q` that put the tool centre at `tool` and the hand at `rot`, or None."""
        seed = self.tensor.to_device(q).view(1, -1)
        goal = self.hand_goal(tool, rot)
        for _ in range(IK_TRIES):
            r = self.mg.ik_solver.solve_single(goal, retract_config=seed, seed_config=seed.view(1, 1, -1),
                                               return_seeds=self.mg.ik_solver.num_seeds)
            ok = r.success.view(-1).cpu().numpy().astype(bool)
            if ok.any():
                break
        else:
            return None
        sols = r.solution.view(-1, len(self.names)).cpu().numpy()[ok]
        return sols[np.argmin(np.abs(sols - q).max(axis=1))]

    def ik_near(self, tool, rot, ignore=(), near=None):
        """Joints that put the tool centre at `tool` and the hand at `rot` without touching anything
        but `ignore`: of the solutions found, the one nearest `near` (the arm's joints now if None),
        or None. The Franka holds one hand pose in many ways (seven joints); a plan to any of them
        can end with a wrist joint at its limit, and the arm then cannot carry level from there (a
        held beaker went 168 degrees over and back, task 17, 22 Oct 2026)."""
        self.refresh(ignore)
        return self._ik_from(tool, rot, self.joints().position.cpu().numpy().reshape(-1) if near is None else np.asarray(near, float))

    def plan_joints(self, q_goal, ignore=()):
        self.refresh(ignore)
        goal = JointState.from_position(self.tensor.to_device(np.asarray(q_goal, float)).view(1, -1), joint_names=self.names)
        r = self.mg.plan_single_js(self.joints(), goal, MotionGenPlanConfig(max_attempts=4, time_dilation_factor=SLOW))
        if not r.success.item():
            return None, str(r.status).split(".")[-1].lower()
        return r.get_interpolated_plan().position.cpu().numpy().reshape(-1, len(self.names)), None

    def clear(self, traj, ignore=()):
        """Whether every row of a joint trajectory (n, 7) keeps the arm and what it holds off
        everything (cuRobo's collision check of each state, no path between them)."""
        self.refresh(ignore)
        q = self.tensor.to_device(np.asarray(traj, float))
        m = self.mg.check_constraints(JointState.from_position(q, joint_names=self.names))
        return bool(m.feasible.all().item())

    def lean(self, traj, rot0=None):
        """The most a held object leans along a trajectory, in degrees: how far the hand's turn since
        its first row tips the vertical."""
        rots = self.hand_rotations(traj)
        rot0 = rots[0] if rot0 is None else rot0
        up = (rots * rot0.inv()).apply([0, 0, 1])
        return float(np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))).max())

    # ---- following a plan ----

    def follow(self, traj, stop=None, hold=SETTLE_STEPS):
        """Drive the arm along a planned trajectory, one row per physics step, then hold the last
        row up to `hold` steps, until every joint is within SETTLE_RAD of it. `stop(i)` is asked
        every step: True ends it there, the arm held where it is. Returns whether it stopped early."""
        b = self.b
        traj = np.asarray(traj)
        lag = 0.0
        self.followed_to = None
        for i, q in enumerate(traj):
            b.step(ArticulationAction(joint_positions=q, joint_indices=self.idx))
            b.trace("follow", i)
            lag = max(lag, float(np.max(np.abs(b.robot.get_joint_positions()[self.idx] - q))))
            if stop is not None and stop(i):
                self._hold_here()
                return True
        off = float(np.max(np.abs(b.robot.get_joint_positions()[self.idx] - traj[-1])))
        for _ in range(hold):
            if off < SETTLE_RAD:
                break
            b.step(ArticulationAction(joint_positions=traj[-1], joint_indices=self.idx))
            off = float(np.max(np.abs(b.robot.get_joint_positions()[self.idx] - traj[-1])))
        b.note(f"follow: {len(traj)} steps, joints at most {np.degrees(lag):.1f} deg behind, "
               f"{np.degrees(off):.2f} deg off the end after holding")
        self.followed_to = traj[-1].copy()
        return False

    @staticmethod
    def paced(traj, max_rad=STRAIGHT_RAD_PER_STEP):
        """traj with rows put in between where a joint would move more than `max_rad` in one physics
        step, so the drives keep up with it; unchanged where none does."""
        traj = np.asarray(traj)
        if len(traj) < 2:
            return traj
        n = np.ceil(np.abs(np.diff(traj, axis=0)).max(axis=1) / max_rad).astype(int)
        if (n <= 1).all():
            return traj
        rows = [traj[:1]]
        for a, b, k in zip(traj[:-1], traj[1:], np.maximum(n, 1)):
            rows.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
        return np.concatenate(rows)

    def _hold_here(self, n=10):
        q = self.b.robot.get_joint_positions()[self.idx].copy()
        for _ in range(n):
            self.b.step(ArticulationAction(joint_positions=q, joint_indices=self.idx))

    # ---- what the hand holds ----

    def attach(self, obj):
        """Plan with `obj` as part of the hand from now on: spheres fitted to its boxes where it is
        now. (attach_objects_to_robot takes the object's pose from the world MotionGen was built
        with, the scene's start: a lifted beaker was then planned for 12 cm below where it hung,
        and carried through its neighbours, 21 Oct 2026.)"""
        parts = self._world(only=obj).objects
        if parts:
            self.mg.attach_external_objects_to_robot(self.joints(), parts, surface_sphere_radius=0.005)
            self.attached = obj

    def detach(self):
        if self.attached:
            self.mg.detach_object_from_robot()
            self.attached = None
