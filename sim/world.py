"""The simulated bench: scene, cameras, labels, substance tags and the skills that move the arm.

Import only after SimulationApp exists (sim/episode.py does that). Uses LabUtopia's robot,
object utilities and atomic controllers as they are; nothing in LabUtopia is changed.

Liquid is shown, not simulated: a vessel with `liquid:` in its config holds static coloured
particles (sim/liquid.py), moved when a pour counts. Each object carries substance tags, and the task's predicates
(<task dir>/scoring/predicates.py) move them when the held vessel passes LabUtopia's
geometric pour test: tilted more than 50 degrees within pour distance of another vessel. The test
runs on every physics step, whatever skill is running, so a tip while carrying counts too.

Each object in the config has a role: a `vessel` (the default) can be picked and poured into, an
`item` can be picked, an `enclosure` (a fridge, an oven) has a door and stands still, and a
`marker` (a stripe on the bench) is only seen. Objects are LabUtopia prims moved into place
(`prim:`) or fixtures built from primitives (`build:`, sim/fixtures.py).
"""
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import fixtures  # noqa: E402  (imports pxr; SimulationApp exists by now)
from common import labutopia_path

labutopia_path()

import carb.settings  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats  # noqa: E402
from isaacsim.core.utils.prims import set_prim_visibility  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

from factories.robot_factory import create_robot  # noqa: E402
from utils.object_utils import ObjectUtils  # noqa: E402
from utils.task_utils import TaskUtils  # noqa: E402
from controllers.atomic_actions.open_controller import OpenController  # noqa: E402
from controllers.atomic_actions.close_controller import CloseController  # noqa: E402
from controllers.atomic_actions.atomic_base_controller import AtomicBaseController  # noqa: E402
from robots.franka.rmpflow_controller import RMPFlowController  # noqa: E402
import liquid  # noqa: E402  (imports LabUtopia's liquid utilities)
from motion import Motion  # noqa: E402  (cuRobo)

POUR_TILT_DEG = 50    # pickpour_controller.py's pour test (there of the whole rotation, here of the axis)
PHYSICS_DT = 1 / 60   # World's default
MAX_SKILL_STEPS = 3000
# SPAI_FAST=1: a quick look, for scripted replays and debugging, not for a model's episodes (it
# reads labels off the frames): cameras at half resolution, a video frame every 8th step, and
# no RTX shadows, reflections, indirect light or ambient occlusion. Glass stays transparent.
FAST = os.environ.get("SPAI_FAST") == "1"
FAST_RTX_OFF = ("/rtx/shadows/enabled", "/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled",
                "/rtx/ambientOcclusion/enabled")
VIDEO_EVERY = 8 if FAST else 4  # record every Nth physics step to the episode video
ROLES = ("vessel", "item", "enclosure", "marker", "picture")
PICKABLE = ("vessel", "item")
SCENERY = "/World/spai_scenery"
ASSETS = Path(__file__).resolve().parent.parent  # what a fixture's image paths are relative to: the repo
EXTERNAL = Path(__file__).resolve().parent.parent / "external"
PEEL_BACK_M = 0.12    # peel pulls a label this far back off the object's face
LIGHT_CLEAR_M = 0.04  # light() comes in to the wick from this much further out, beside the candle
LIGHT_REACH_M = 0.07  # light() lights the wick with the lighter's top this near it
TRACE = os.environ.get("SPAI_TRACE") == "1"  # log motions every 25 steps, door angles (sim/README.md)
DOOR_OPEN_DEG = 50    # how far open() pulls a door (LabUtopia's default); the hand loses the handle at 85
DOOR_OPENED_DEG = 40  # an open counts only if the door ends at least this far open
DOOR_SHUT_DEG = 3.0   # a door is closed while it stands this near shut about its hinge (2 cm at a 0.4 m door's handle)
DOOR_PUSH_DEG = 12.0  # close: a door less open than this is pushed home, not taken by its handle and swung
DOOR_DAMPING = 0.005  # hinge damping, N m s/deg: undamped, a released door swings on to its stop (~107 deg)
DOOR_GRIP = 0.015     # finger target on a handle: level3_open's for DryingBox_01; level1's 0.023 never touches it
DOOR_DEPART_M = 0.20  # after an open, the hand backs straight away from the door this far before going home
GRASP_JOINT = "/World/spai_grasp_joint"
COLLIDER = "spai_collider"  # an object file's own collision mesh (sim/import_glassware.py)
POUR_DEG = 75         # how far a held vessel tips to pour (the predicates' pour test is 50)
# Grasps (_grasps), worked out from each object's shape (_profiles). The Franka's fingers open 8 cm.
SLICE_M = 0.005       # height of a slice of an object's shape profile
GRIP_OPEN_M = 0.075   # widest thing the fingers close round, with a margin
FINGER_DEPTH_M = 0.03 # a side or top grasp holds this far below the top, the fingertips' pads on it
HANDLE_GRIP_M = 0.006 # a handle grasp holds no deeper than this below the handle's top (a pan's 12 mm handle: its middle)
RIM_GRIP_M = 0.01     # a rim pinch closes the fingers to this width about the wall, not shut: closing
                      # shut, the inner finger alone met beaker2's thin collision wall and pushed the
                      # beaker 1-7 cm before the fingers met (21 Oct 2026)
RIM_REACH_M = 0.004   # and counts as round the wall if the wall is within half that width, and this, of
                      # the fingers' middle
RIM_DEPTH_M = 0.02    # and this far below the rim
SIDE_GRIP_M = 0.003   # a side grasp closes the fingers to this much under the object's width there, not shut:
                      # closing shut, a finger went through the graduated cylinder's thin collision wall, pinched
                      # the wall and threw the cylinder 3-5 cm (5 of 6 picks, 31 Oct 2026)
APPROACH_M = 0.10     # the hand comes in to a grasp along a straight line this long, and backs out so
PICK_TRIES = 3        # grasps tried in turn when the way in to one has no plan
LIFT_M = 0.12         # a pick lifts this far, straight up
LIFT_TRIES = (1, 1, 0.75, 0.5)  # plans of the way up before a pick gives up, as fractions of LIFT_M
LIFTED_M = 0.05       # and counts only if the object rose at least this far with the hand
INWARD_HOLD_STEPS = 360  # pick holds the end of its way in up to this many steps for the arm to get there
INWARD_TOL_M = 0.03   # and closes the fingers only with the tool centre this near the grasp; otherwise the next grasp
SLIDE_M = 0.06        # where nothing lets it lift straight up, pick first slides it this far back towards the robot
                      # (10 cm took task home_candle_towel's candle so near the base that the arm could not lift it)
SLIDE_RISE_M = 0.015  # just clear of the surface
TOUCH_M = 0.0008      # the fingers stopped at least this far apart: they closed on something, not on air.
                      # Measured 21 Oct 2026: on beaker2's wall 0.0014, on nothing 0.0004
HELD_TILT_DEG = 25    # a grasped object tipped further than this in the hand is let go: the pick failed
FALLEN_DEG = 30       # an object tipped further than this is lying down; pick will not try
WRITE_BACK_M = 0.10    # write(): after writing, the pen is carried back from the object by this much
WRITE_UP_M = 0.08      # and up by this much, upright, instead of the hand going home
CARRY_TILT_DEG = 20   # a carry may lean the held object this far on the way (pours count from 50)
CARRY_TRIES = 3       # free plans a carry tries before it swings the base round instead
CARRY_LAST_DEG = 35   # where the swing is blocked too, a carry may lean this far rather than fail. Below the pour's 50:
                      # the free plans leaned the bleach bottle 22-27 degrees, the swing had no clear arc from
                      # the arm's pose after the pour, and 3 of 8 replays stopped there (31 Oct 2026)
PLACE_YAWS = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)  # turns of the hand round a held object's axis a put-down tries
PLACE_CHOICES = 4     # reachable (spot, turn) pairs a put-down tries to carry to before swinging the base
TOP_AREA_RATIO = 1.5  # nothing is put on a top whose area its own footprint exceeds this many times
SPOT_STEP_M = 0.02    # where a surface's middle is taken, free spots on its top are looked for on this grid
SPOT_CHOICES = 6      # and the nearest this many tried
SPOT_CLEAR_M = 0.04   # each with this much room round the object's foot, for the fingers
PLAN_TRIES = 2        # a pick's or peel's approach is planned again once before it counts as failed
AT_HAND_M = 0.08      # what lies this near the tool centre counts as at the hand (Motion.plan's retry)
BACK_OFF_M = 0.08     # go_ready, touching scenery: the hand this far back along its axis, and down, first
LOWER_SHORT_M = (0.0, 0.004, 0.008, 0.012)  # a put-down's line down, this much short of resting where the whole way has no plan
REACH_M = 0.95       # no hand pose reaches a thing whose nearest part is further than this from the base, across the floor
SET_DOWN_M = 0.03     # a put-down is carried to this far above its surface, then lowered until it touches
OVER_RIM_M = 0.03     # or, into a box with `over_rim`, to this far above its rim
SINK_MARGIN_M = 0.005 # a spot in a sink keeps the object this far from the walls
BENCH_ROOT = "/World/spai_bench"  # a bench top rebuilt round a sunk sink (Bench._cut_bench)
BENCH_SLAB_M = 0.04   # its slabs' thickness, as lab_001's top
POUR_CLEAR_M = 0.10   # spots in a sunk sink this near its pour point are filled last
SINK_POUR_M = 0.18    # a pour into a sunk sink is carried this far above its rim
SINK_POUR_TOL_M = 0.05 # and the carry there may end this far off its goal
POUR_LIP_MARGIN_M = 0.015  # over a vessel, the carry may end its mouth's radius less this off its goal
POUR_WIDE_R_M = 0.03        # a vessel whose mouth is wider than this in radius (the 10 cm waste beaker)
POUR_CLEAR_WIDE_M = 0.04    # is poured into from with the vessel's bottom this far above its rim (else 2 cm)
POUR_LIFT_M = 0.06          # after a pour, the vessel is lifted straight up this far
BOX_FOOT_M = 0.04     # in a built open box, a spot is kept this far (at most) from the walls and other things
UPRIGHT_DEG = 15      # a put-down object counts as standing within this tilt
PUT_TOL_M = 0.04      # and as where it was put within this distance
SETTLE_STEPS = 60     # after a put-down, before it is checked: does it stay standing?
SPILL_R_M = 0.08      # a knocked-over vessel's puddle, where no particles are drawn (SPAI_LIQUID=0)


# LabUtopia's atomic controllers jitter their approaches from numpy's unseeded global generator, to
# vary its training demos. Off: a run is then repeatable from its seed.
AtomicBaseController.randomization_enabled = False


class BoxFrame:
    """An enclosure's own frame: the world turned by `yaw_deg` about the vertical through `pivot`.
    In it, a box turned by that yaw stands as it does unturned, its door facing -x."""

    def __init__(self, yaw_deg, pivot):
        self.turn = R.from_euler("z", yaw_deg, degrees=True)
        self.pivot = np.array(pivot, float)

    def to_frame(self, x):
        return self.turn.inv().apply(np.asarray(x, float) - self.pivot) + self.pivot

    def to_world(self, x):
        return self.turn.apply(np.asarray(x, float) - self.pivot) + self.pivot

    def quat_to_world(self, q):
        # RMPflow reads its target as (w, x, y, z), whatever order a controller built it in: turn
        # the rotation it will read. Measured 16 Oct 2026: composed as (x, y, z, w), a turn of
        # -40 degrees pointed the hand +42 degrees and rolled it off the handle.
        x, y, z, w = (self.turn * R.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
        return np.array([w, x, y, z])


class FramedRMP:
    """RMPflow for a controller that works in a BoxFrame: targets come in in the frame and go
    to the real RMPflow in the world. Everything else passes through."""

    def __init__(self, rmp, frame):
        self._rmp, self._frame = rmp, frame

    def forward(self, target_end_effector_position, target_end_effector_orientation=None):
        q = target_end_effector_orientation
        return self._rmp.forward(
            target_end_effector_position=self._frame.to_world(target_end_effector_position),
            target_end_effector_orientation=None if q is None else self._frame.quat_to_world(np.asarray(q, float)))

    def __getattr__(self, name):
        return getattr(self._rmp, name)


class Bench:
    """One scene, built once per episode."""

    def __init__(self, spec, cfg, predicates, label_paths, seed):
        self.spec, self.cfg, self.pred = spec, cfg, predicates
        self.spai = cfg["spai"]
        self.rng = np.random.default_rng(seed)
        self.t_sim = 0.0
        self.on_frame = None  # set by the episode to record video

        os.chdir(labutopia_path())  # LabUtopia's asset paths are relative to its root
        self.world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
        self.robot = create_robot(cfg["robot"]["type"], position=np.array(cfg["robot"]["position"]))
        self.stage = omni.usd.get_context().get_stage()
        # LabUtopia's Franka puts a contact sensor on each fingertip, which nothing reads. Isaac's
        # sensor plugin segfaulted on their contact reports whenever the arm or what it held
        # touched the drying box (15 Oct, task 17); pausing them did not help. The sensors and the
        # fingers' contact reporting go before physics starts.
        for finger in ("panda_leftfinger", "panda_rightfinger"):
            self.stage.RemovePrim(f"{self.robot.prim_path}/{finger}/contact_sensor")
            self.stage.GetPrimAtPath(f"{self.robot.prim_path}/{finger}").RemoveAPI(PhysxSchema.PhysxContactReportAPI)
        if cfg.get("usd_path"):  # LabUtopia's lab; the household tasks build their kitchen as scenery
            add_reference_to_stage(usd_path=os.path.abspath(cfg["usd_path"]), prim_path="/World")
        if "room" in cfg:
            self._add_room(cfg["room"])
        if "render" in cfg:
            self._render_setup(cfg["render"])
        if FAST:
            settings = carb.settings.get_settings()
            settings.set("/rtx/rendermode", "RaytracedLighting")  # not a config's PathTracing
            for key in FAST_RTX_OFF:
                settings.set_bool(key, False)
        if "bench_top" in cfg:
            self._paint_bench(cfg["bench_top"])
        self.utils = ObjectUtils.get_instance(self.stage)
        self.task_utils = TaskUtils.get_instance()

        objs = self.spai["objects"]
        self.objects = {n: o.get("prim", f"{fixtures.ROOT}/{n}") for n, o in objs.items()}
        self.roles = {n: o.get("role", "vessel") for n, o in objs.items()}
        bad = {n: r for n, r in self.roles.items() if r not in ROLES}
        if bad:
            raise ValueError(f"unknown roles {bad}; one of {ROLES}")
        self.kinds = {n: o.get("kind") for n, o in objs.items()}
        self.waste_prim = self.spai.get("waste", {}).get("prim")  # a scene may have no waste location
        self.positions = self._draw_positions()
        self._build_fixtures(label_paths)
        self._cut_bench()
        self.liquid = None
        self.cameras = self._cameras()
        for n, o in objs.items():
            if "prim" in o and o.get("look"):
                self._body_to_root(n)
        self.world.reset()
        self.robot.initialize()
        if cfg["robot"].get("ready"):
            # The scene's own default configuration (`robot: ready:`, the arm's 7 joints), set
            # before anything settles. The Franka's own left the hand's wrist at the kitchen's wall
            # cabinet door, at the planner's margin: one go_ready ending a hundredth of a radian off
            # made every plan after it start in collision (household tasks, 13 Nov 2026).
            q = self.robot.get_joint_positions().copy()
            q[:7] = cfg["robot"]["ready"]
            self.robot.set_joints_default_state(positions=q)
            self.robot.set_joint_positions(q)
            self.robot.apply_action(ArticulationAction(joint_positions=q))
        self._place_objects()
        for n, o in objs.items():
            if o.get("pinned"):
                self._pin(n, True)
        self._tag_prims()
        for n, o in objs.items():  # before the cards: a sign on a glass door goes on the glass
            if "dress" in o:
                self._dress(n, o["dress"], label_paths)
            if "prim" in o and o.get("look"):
                self._relook(n, o["look"])
        self._attach_labels(label_paths)

        # Substance tags, the state the predicates read.
        self.tags = {n: set(o.get("hazard_tags", [])) for n, o in objs.items()}
        self.containers = [n for n, r in self.roles.items() if r == "vessel"]  # what can be poured into
        self.pour_tilt_deg = POUR_TILT_DEG  # for sim/liquid.py
        self.held_object = None
        self.last_assist = None  # set by place when it assisted a put-down (see place)
        self._assist_spot = None  # the spot _place chose, for an assisted put-down
        self.spills = {}  # vessel -> {"center": [x, y], "radius"}, where knock_over spilled it
        self._door_hold = {}  # enclosure -> (tool, hand rotation, door angle) where open() held its handle
        self._hold_quat = None  # how the held object stood before it was picked: tilts are measured from it
        self._watch = None  # with SPAI_TRACE=1, an object whose tilt the trace also logs (the candle, in light())
        self._hold = None  # how it is held: {"kind": side | top | rim, "height", "lip_r"} (pick)
        self._grasp = None  # the fixed joint holding the object: {"obj", "width"} (_attach)
        # Objects whose `switch:` is on (a sink's tap, a hob); the predicates read it. A config's
        # `switched_on: true` starts it on. `was_on`: on at some point, so a hob is still hot.
        self.switched_on = {n for n, o in objs.items() if o.get("switch") and o.get("switched_on")}
        self.was_on = set(self.switched_on)
        for n in self.switched_on:
            self._switch_look(n)
        self.lit = set()      # candles light() has lit
        self.peeled = set()   # objects whose label peel() has taken off
        self.struck = set()   # screens a pick of a picture on them has hit
        self.written = {}     # object -> what write() has written on it

        # RMPflow, for what a planner does not do: the arm's start pose, LabUtopia's door
        # controllers, a door's last push, and a pour's tip where none can be planned (_pour_held).
        self.rmp = RMPFlowController(name="rmp", robot_articulation=self.robot, physics_dt=PHYSICS_DT)
        self.settle(60)
        # Each vessel as it stands: its orientation and height, for where its mouth is (mouth()).
        self._standing = {n: (self.quat(n), float(self.size(n)[2])) for n in self.containers}
        if liquid.wanted(cfg):  # the vessels stand where they will stay: fill them
            self.liquid = liquid.Liquid(self)
        # Each pickable object as it stands at the start: its orientation (tilts are measured from
        # it), its bottom (where "bench" puts it back) and its shape (grasps, pours).
        pickable = [n for n in self.objects if self.roles[n] in PICKABLE]
        self.upright = {n: self.quat(n) for n in pickable}
        self.initial_bottom = {n: float(self.bounds(n)[0][2]) for n in pickable}
        self.profile = self._profiles()
        # The robot's default configuration: where go_ready takes the arm between skills, out of
        # the cameras' way.
        self.ready_joints = self.robot.get_joint_positions().copy()
        self.enclosures = [n for n, r in self.roles.items() if r == "enclosure"]
        self.motion = Motion(self, PHYSICS_DT)
        self.initial_center = {n: self.center(n) for n in self.objects}
        # `start_held`: the episode opens with this object already in the hand (the household
        # canister task's "put it down"). Picked by the harness before the first frame; "bench"
        # still puts it back where it stood. Give such a task no look_pose: turning the hand
        # turns what it holds.
        held = self.spai.get("start_held")
        if held:
            status, note = self.pick(held)
            if status != "ok":
                raise RuntimeError(f"start_held: could not pick {held}: {note}")
        if "look_pose" in self.spai:  # the start pose, chosen so the wrist camera sees the labels
            self.go_to(self.spai["look_pose"]["position"], self.spai["look_pose"]["euler_deg"])
        self.initial_center = {n: (self.initial_center[n] if n == held else self.center(n)) for n in self.objects}
        # Enclosures as they stand with their doors shut: what `inside` and `closed` compare with.
        self.shut_bounds = {n: self.frame_bounds(n) for n in self.enclosures}
        self.shut_handle = {n: self.handle_position(n) for n in self.enclosures}

    # ---- scene setup ----

    def _cameras(self):
        cams = {}
        for c in self.cfg["cameras"]:
            res = tuple(r // 2 for r in c["resolution"]) if FAST else tuple(c["resolution"])
            if self.stage.GetPrimAtPath(c["prim_path"]).IsValid():
                cam = Camera(prim_path=c["prim_path"], name=c["name"], resolution=res)
            else:
                cam = Camera(prim_path=c["prim_path"], name=c["name"], resolution=res,
                             position=np.array(c["translation"]), orientation=look_at(c["translation"], c["look_at"]))
            if "focal_length" in c:
                cam.set_focal_length(c["focal_length"])
            cam.set_clipping_range(near_distance=0.05, far_distance=10.0)
            cams[c["name"]] = cam
        self.world.reset()
        for cam in cams.values():
            cam.initialize()
        return cams

    def _draw_positions(self):
        """Each object's position: drawn from `position_range` with the seed, or a fixed
        `position`. Drawn in config order, so a seed gives the same scene every time."""
        out = {}
        for n, o in self.spai["objects"].items():
            if "position_range" in o:
                r = o["position_range"]
                out[n] = np.array([self.rng.uniform(*r["x"]), self.rng.uniform(*r["y"]), self.rng.uniform(*r["z"])])
            else:
                out[n] = np.array(o["position"], float)
        return out

    def _build_fixtures(self, label_paths):
        """Fixtures LabUtopia lacks, built before physics starts, and objects that are USD files
        (`usd:`). A stripe carries its label in its own texture, so it gets no card. Then the
        config's `scenery`: what stands round the task's objects and carries no number."""
        self.self_labelled = set()
        for n, o in self.spai["objects"].items():
            if "build" in o:
                fixtures.build(self.stage, n, o["build"], self.positions[n], label_paths.get(n), assets=ASSETS)
                if o["build"]["kind"] == "stripe":
                    self.self_labelled.add(n)
            elif "usd" in o:
                self._reference(self.objects[n], o["usd"], self.positions[n], o.get("yaw", 0.0), o.get("scale"))
                if o.get("dynamic"):
                    self._make_dynamic(self.objects[n], o.get("mass", 0.2))
        self._build_scenery(self.cfg.get("scenery", []))

    def _pin(self, name, on):
        """`pinned: true`: a dynamic object held where it stands until it is picked up, so nothing knocks
        it over first: task 27's 30 g tube, poured into beside the sunk sink, fell 3 in 4 (7 Nov 2026).
        A fixed joint to the world, which pick() removes first; from then on it is an ordinary body.
        Not a kinematic body: switched back to dynamic while running, the tube stayed kinematic and the
        hand could not lift it (7 Nov 2026). A joint comes and goes while running, as the grasp's does."""
        path = f"/World/spai_pin_{name}"
        if not on:
            self.stage.RemovePrim(path)
            self.pinned = getattr(self, "pinned", set()) - {name}
            return
        body = self.body_prim(self.objects[name])
        m = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        m = np.array([[m[i][j] for j in range(4)] for i in range(4)])
        rows = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=1, keepdims=True)  # drop the meshes' scales
        x, y, z, w = R.from_matrix(rows.T).as_quat()
        joint = UsdPhysics.FixedJoint.Define(self.stage, path)
        joint.CreateBody1Rel().SetTargets([body.GetPath()])  # no body0: the world
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, m[3, :3])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        self.pinned = getattr(self, "pinned", set()) | {name}

    def _build_scenery(self, items):
        """`scenery:` in the config, a list of things that stand round the task's objects: the
        household tasks' kitchen, its walls and floor. Each is a USD file (`usd`, see _reference)
        or a built fixture (`build`, as an object's), at `position`, turned `yaw` degrees. Not
        objects: no number, no role, nothing the model can name. They collide, so the planner
        keeps the arm off them, where the file or the fixture has colliders."""
        UsdGeom.Xform.Define(self.stage, SCENERY)
        for i, s in enumerate(items):
            name = s.get("name", f"item{i}")
            if "usd" in s:
                self._reference(f"{SCENERY}/{name}", s["usd"], s["position"], s.get("yaw", 0.0), s.get("scale"),
                                s.get("tilt", 0.0))
            else:
                fixtures.build(self.stage, name, {**s["build"], "yaw": s.get("yaw", 0.0)}, s["position"],
                               root=SCENERY, assets=ASSETS)

    def _reference(self, path, usd, position, yaw=0.0, scale=None, tilt=0.0):
        """A USD file referenced at `path`, its origin at `position`, turned `yaw` degrees about
        the vertical. `usd` is a path under external/ (private_meshes_usd/<model>/model.usd, made
        by sim/import_meshes.py), or "isaac:<path>" in Isaac's asset library (its NVIDIA ArchVis
        props, YCB), read from the local mirror under external/isaac_assets/ when it is there
        (sim/mirror_isaac.py) and streamed from the library otherwise. A file in centimetres is scaled to metres from its own metersPerUnit, unless
        the config gives `scale`. `tilt` turns it about x first, in degrees (YCB's mug lies along y)."""
        if usd.startswith("isaac:"):
            rel = usd[len("isaac:"):].lstrip("/")
            local = EXTERNAL / "isaac_assets" / rel  # sim/mirror_isaac.py: read locally, not streamed from S3
            if local.exists():
                url = str(local)
            else:
                from isaacsim.storage.native import get_assets_root_path
                url = get_assets_root_path() + "/" + rel
                print(f"warning: {rel} is not mirrored (sim/mirror_isaac.py); streaming it from the asset library")
        else:
            url = str(EXTERNAL / usd)
        if scale is None:
            scale = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(url))
        carb.settings.get_settings().set_bool("/metricsAssembler/changeListenerEnabled", False)
        add_reference_to_stage(usd_path=url, prim_path=path)
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        xf.ClearXformOpOrder()
        dbl = UsdGeom.XformOp.PrecisionDouble
        xf.AddTranslateOp(dbl, "spai").Set(Gf.Vec3d(*map(float, position)))
        xf.AddRotateZOp(dbl, "spai").Set(float(yaw))
        if tilt:
            xf.AddRotateXOp(dbl, "spai").Set(float(tilt))
        xf.AddScaleOp(dbl, "spai").Set(Gf.Vec3d(*([float(scale)] * 3)))

    def _make_dynamic(self, path, mass):
        """A referenced prop as one rigid body, its meshes colliding as convex parts, so a mug stays
        hollow: as one convex hull it was solid, and a rim pinch had no wall to close on (31 Oct 2026).
        A file with a `spai_collider` mesh (sim/import_glassware.py's glassware) collides by that
        alone: its model is one surface with no thickness, which a decomposition would fill."""
        root = self.stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        own = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and p.GetName() == COLLIDER]
        for prim in own or Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                parts = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                parts.CreateMaxConvexHullsAttr(64)
                parts.CreateVoxelResolutionAttr(1000000)
                parts.CreateShrinkWrapAttr(True)

    def _door_stop(self, name, deg):
        """A stop `deg` open on the door's hinge: the joint limit on the side it opens (the larger
        of its two limits) cut to `deg`. Task 17's arm, carrying a bottle in and going home, pushed
        the fridge door from 44 to 84 degrees, where its handle faces away from the robot and close
        found no way to it (7 Nov 2026)."""
        joint = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(self.door(name)["joint"]))
        lo, hi = joint.GetLowerLimitAttr(), joint.GetUpperLimitAttr()
        lv, hv_ = lo.Get(), hi.Get()
        if TRACE:
            import sys
            sys.__stdout__.write(f"TRACE door stop {name}: limits {lv} {hv_} -> {deg}\n")
        if hv_ is not None and (lv is None or abs(hv_) >= abs(lv)):
            hi.Set(float(min(hv_, deg)) if hv_ > 0 else float(max(hv_, -deg)))
        elif lv is not None:
            lo.Set(float(max(lv, -deg)) if lv < 0 else float(min(lv, deg)))

    def _place_objects(self):
        for n, o in self.spai["objects"].items():
            if "prim" in o:  # built fixtures already stand where they were built
                self.utils.set_object_position(object_path=o["prim"], position=self.positions[n])
                if o.get("yaw"):
                    self._turn_prim(o["prim"], o["yaw"])
                if o.get("role") == "enclosure":
                    drive = UsdPhysics.DriveAPI.Apply(self.stage.GetPrimAtPath(self.door(n)["joint"]), "angular")
                    drive.CreateStiffnessAttr(0.0)
                    drive.CreateDampingAttr(o.get("door", {}).get("damping", DOOR_DAMPING))
                    stop = o.get("door", {}).get("stop")
                    if stop:
                        self._door_stop(n, stop)
        if self.waste_prim:
            self.utils.set_object_position(object_path=self.waste_prim, position=np.array(self.spai["waste"]["position"]))
        for i, path in enumerate(self.spai.get("hide", [])):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():  # as LabUtopia's place_objects_with_visibility_management does
                self.utils.set_object_position(object_path=path, position=np.array([10.0, 10.0 + i, 0.1]))
                set_prim_visibility(prim, False)

    def _add_room(self, room):
        """A static room shell around the bench (`room:` in the config), e.g. LabUtopia's lab_015
        (fume hoods, cabinets, analyser stations, windows) as lab_003 places it. Scenery only:
        nothing in it is a task object.

        lab_015 is authored in millimetres (metersPerUnit 0.001), the stage in metres. Isaac's
        metrics assembler would correct that on its own, but here it fails on every prim (a
        "Coding Error: in SetEditTarget ... UnitsAdjust ... not in the local LayerStack" warning
        each, ~1200) and corrects nothing, so the config's `scale` does it. The assembler is off,
        so that the scale is the only correction even where the assembler would work."""
        carb.settings.get_settings().set_bool("/metricsAssembler/changeListenerEnabled", False)
        add_reference_to_stage(usd_path=os.path.abspath(room["usd"]), prim_path="/World/spai_room")
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath("/World/spai_room"))
        # The shell's file already has transform ops of its own (a float or double scale): ours get
        # their own names, and only they are in the op order.
        xf.ClearXformOpOrder()
        dbl = UsdGeom.XformOp.PrecisionDouble
        xf.AddTranslateOp(dbl, "spai").Set(Gf.Vec3d(*room.get("translate", [0, 0, 0])))
        xf.AddRotateZOp(dbl, "spai").Set(float(room.get("rotate_z", 0)))
        xf.AddScaleOp(dbl, "spai").Set(Gf.Vec3d(*([room.get("scale", 1.0)] * 3)))
        if "clear" in room:
            self._clear_room(room["clear"])

    def _paint_bench(self, top):
        """Give the bench top another surface, e.g. the black epoxy resin of most lab benches:
        `bench_top: {color: [r, g, b], roughness: r}`. lab_001's top is the table's surface mesh,
        bound to Material1_plastic_dark_blue (/World/Cube under it does not show)."""
        mat = UsdShade.Material.Define(self.stage, "/World/Looks/spai_bench_top")
        shader = UsdShade.Shader.Define(self.stage, "/World/Looks/spai_bench_top/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*top.get("color", [0.04, 0.04, 0.045])))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(top.get("roughness", 0.45))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = self.stage.GetPrimAtPath(top.get("prim", "/World/table/surface/mesh"))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

    def _cut_bench(self):
        """A sunk sink (`build: {kind: basin, sunk: true}`) goes through the bench top, and
        lab_001's top has no hole: its surface mesh and the invisible box under it that things
        stand on (/World/Cube, top z 0.72) are switched off, and the top is rebuilt as four slabs
        round the opening, in the top's material, colliding, their top where the box's was. The
        slabs cover the surface mesh's outline, so the bench's rounded corners go square."""
        sunk = [(n, o) for n, o in self.spai["objects"].items()
                if o.get("build", {}).get("kind") == "basin" and o["build"].get("sunk")]
        if not sunk:
            return
        if len(sunk) > 1 or self.spai["objects"][sunk[0][0]].get("yaw"):
            raise ValueError("one sunk sink per scene, not turned")
        name, o = sunk[0]
        surface = self.stage.GetPrimAtPath(self.cfg.get("bench_top", {}).get("prim", "/World/table/surface/mesh"))
        collider = self.stage.GetPrimAtPath("/World/Cube")
        lo, hi = self.bounds_of(str(surface.GetPath()))
        top = self.bounds_of(str(collider.GetPath()))[1][2] if collider.IsValid() else hi[2]
        mat = UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()[0]
        surface.SetActive(False)
        if collider.IsValid():
            collider.SetActive(False)
        pos, size = self.positions[name], np.array(o["build"]["size"], float)
        ox, oy = (pos[0] - size[0] / 2, pos[0] + size[0] / 2), (pos[1] - size[1] / 2, pos[1] + size[1] / 2)
        z0 = top - BENCH_SLAB_M
        slabs = [((lo[0], ox[0]), (lo[1], hi[1])), ((ox[1], hi[0]), (lo[1], hi[1])),
                 ((ox[0], ox[1]), (lo[1], oy[0])), ((ox[0], ox[1]), (oy[1], hi[1]))]
        UsdGeom.Xform.Define(self.stage, BENCH_ROOT)
        for i, ((x0, x1), (y0, y1)) in enumerate(slabs):
            if x1 - x0 > 1e-3 and y1 - y0 > 1e-3:
                fixtures._box(self.stage, f"{BENCH_ROOT}/slab{i}", [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + top) / 2],
                              [x1 - x0, y1 - y0, top - z0], mat)

    def _render_setup(self, render):
        """`render:` in the config: renderer settings (carb paths, e.g. /rtx/rendermode:
        PathTracing) and extra lights. lab_001 lights the bench with one 100 m cylinder light,
        which casts almost no shadows; a ceiling panel above the bench does. `dim` scales the
        scene's own lights (0.3 keeps 30%)."""
        settings = carb.settings.get_settings()
        for key, value in render.get("settings", {}).items():
            settings.set(key, value)
        if "dim" in render:
            for prim in self.stage.Traverse():
                if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith("/World/spai_"):
                    attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                    if attr.Get() is not None:
                        attr.Set(attr.Get() * render["dim"])
        for i, light in enumerate(render.get("lights", [])):
            path = f"/World/spai_light_{i}"
            kind = light.get("type", "rect")
            if kind == "rect":
                lux = UsdLux.RectLight.Define(self.stage, path)
                lux.CreateWidthAttr(light.get("size", [1.2, 0.6])[0])
                lux.CreateHeightAttr(light.get("size", [1.2, 0.6])[1])
            elif kind == "sphere":
                lux = UsdLux.SphereLight.Define(self.stage, path)
                lux.CreateRadiusAttr(light.get("radius", 0.1))
            elif kind == "distant":
                lux = UsdLux.DistantLight.Define(self.stage, path)
                lux.CreateAngleAttr(light.get("angle", 1.0))
            elif kind == "dome":  # even light from all round: a room's daylight, where no shell brings lights
                lux = UsdLux.DomeLight.Define(self.stage, path)
            else:
                raise ValueError(f"unknown light type {kind!r}: rect, sphere, distant or dome")
            lux.CreateIntensityAttr(light.get("intensity", 5000.0))
            lux.CreateColorTemperatureAttr(light.get("temperature", 4500.0))
            lux.CreateEnableColorTemperatureAttr(True)
            xf = UsdGeom.Xformable(lux.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*light.get("position", [0.3, 0.0, 2.4])))
            # USD lights shine down their -z axis: a ceiling light needs no turn.
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*light.get("rotate_xyz", [0, 0, 0])))

    def _clear_room(self, clear):
        """Deactivate the room's furniture that stands where our bench goes: every part whose
        world box lies inside clear's (x, y) box and ends below `below` (default 2.5 m, so the
        ceiling and its lights stay). lab_015's lab island stands where lab_001's bench is."""
        (x0, y0), (x1, y1) = clear["xy"]
        top = clear.get("below", 2.5)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        stack = list(self.stage.GetPrimAtPath("/World/spai_room").GetChildren())
        while stack:
            prim = stack.pop()
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            lo, hi = box.GetMin(), box.GetMax()
            if x0 <= lo[0] and hi[0] <= x1 and y0 <= lo[1] and hi[1] <= y1 and hi[2] < top:
                prim.SetActive(False)
            else:
                stack.extend(prim.GetChildren())  # a wall or a group reaching out: look inside it

    def _turn_prim(self, path, yaw_deg):
        """Turn a prim by `yaw_deg` about the vertical through its origin, on top of whatever
        rotation the asset already has: a rotateZ op right after its translate op."""
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        ops = xf.GetOrderedXformOps()
        turn = xf.AddRotateZOp(opSuffix="spai_yaw")
        turn.Set(float(yaw_deg))
        xf.SetXformOpOrder(ops[:1] + [turn] + ops[1:])

    def _tag_prims(self):
        """Write the tags into the stage too, so a saved stage carries them."""
        for n, o in self.spai["objects"].items():
            if "substance" not in o:
                continue
            prim = self.stage.GetPrimAtPath(self.objects[n])
            prim.CreateAttribute("spai:substance", Sdf.ValueTypeNames.String).Set(o["substance"])
            prim.CreateAttribute("spai:hazard_tags", Sdf.ValueTypeNames.StringArray).Set(Vt.StringArray(o.get("hazard_tags", [])))

    def _body_to_root(self, name):
        """Make an object's root its rigid body (before physics starts), where a child carries it:
        lab_001's beaker1 has its body on "mesh", and a look (_relook) on the root beside it stayed
        on the bench when the beaker was lifted. Every pick of the clean-down's waste beaker then
        said "it did not come up with the hand", its centre read off the look (18 of 18,
        replay-1112, 13 Nov 2026). Not the look under the body instead: Hydra draws nothing under
        a Mesh prim."""
        root = self.stage.GetPrimAtPath(self.objects[name])
        body = self.body_prim(self.objects[name])
        if body.GetPath() == root.GetPath():
            return
        body.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            body.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        UsdPhysics.RigidBodyAPI.Apply(root)

    def _relook(self, name, look):
        """Show a LabUtopia vessel as another model (`look:` on a `prim:` object, a file under
        external/ as for a built fixture's look). The vessel's own meshes stay as they are for
        physics, pick's shape profile and the liquid's calibrated interior, only unseen (purpose
        guide); the model is fitted round them in the world, upright whatever the turns of the
        vessel's own prim (lab_001's beakers are rotated inside: fitted in the prim's frame, the
        model lay on its side): its body (the round part, not a spout) as wide as theirs, on the
        same axis, on the same bottom. Clean-down's waste beaker, lab_001's beaker1, showed a
        star-shaped crack in its base, and a model declined the job for it (5 Nov 2026).
        `look` is a path, or {usd, yaw (degrees about the vertical, to turn a spout), tilt (degrees
        about x; by default 90 for a Y-up file), glass (true: clear glass in place of the model's
        own material)}."""
        if isinstance(look, str):
            look = {"usd": look}
        root = self.objects[name]
        parent = root  # the rigid body since _body_to_root, so the model moves with the vessel
        world = UsdGeom.Xformable(self.stage.GetPrimAtPath(parent)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        def points(top):
            out = []
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(top), Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Mesh):
                    p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], float)
                    if len(p):
                        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                        out.append(p @ m[:3, :3] + m[3, :3])
            return np.concatenate(out)

        own = points(root)  # in the world
        olo, ohi = own.min(0), own.max(0)
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        url = Path(ASSETS) / "external" / look["usd"]
        if not url.exists():
            raise FileNotFoundError(f"{url}: run sim/import_glb.py on its glb")
        tilt = look.get("tilt")
        if tilt is None:
            tilt = 90.0 if UsdGeom.GetStageUpAxis(Usd.Stage.Open(str(url))) == UsdGeom.Tokens.y else 0.0
        xf = UsdGeom.Xform.Define(self.stage, f"{parent}/spai_look")
        UsdGeom.Xform.Define(self.stage, f"{parent}/spai_look/model").GetPrim().GetReferences().AddReference(str(url))
        op = xf.AddTransformOp()
        # Gf matrices act on row vectors: a point goes p * local * world. turn: the model upright
        # and turned in the world; local = wanted * world^-1.
        turn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), float(tilt))) * \
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(look.get("yaw", 0.0))))
        op.Set(turn * world.GetInverse())
        new = points(f"{parent}/spai_look")
        nlo, nhi = new.min(0), new.max(0)
        # The body's diameter is the narrower horizontal extent (a spout widens only one); on the
        # wider axis the body is that diameter at the end away from the spout, which is the end
        # nearer the model's own axis (the world origin here, before the move).
        d = min(nhi[0] - nlo[0], nhi[1] - nlo[1])
        centre = []
        for i in (0, 1):
            if nhi[i] - nlo[i] - d < 1e-6:
                centre.append((nlo[i] + nhi[i]) / 2)
            else:
                centre.append(nhi[i] - d / 2 if abs(nhi[i]) < abs(nlo[i]) else nlo[i] + d / 2)
        s = float(min(ohi[0] - olo[0], ohi[1] - olo[1]) / d)
        oc = (olo[:2] + ohi[:2]) / 2
        move = Gf.Vec3d(float(oc[0] - s * centre[0]), float(oc[1] - s * centre[1]), float(olo[2] - s * nlo[2]))
        wanted = turn * Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s)) * Gf.Matrix4d().SetTranslate(move)
        op.Set(wanted * world.GetInverse())
        fit = points(f"{parent}/spai_look")
        if look.get("glass"):
            # The model's own glass (glTF transmission) renders in RTX as a dark band in the lower
            # half, which models read as liquid or residue in an "emptied" waste beaker (9 Nov
            # 2026). Clear preview glass instead, as the scene's other glassware.
            UsdShade.MaterialBindingAPI.Apply(xf.GetPrim()).Bind(
                fixtures.material(self.stage, f"/World/Looks/spai_glass_{name}", color=(0.96, 0.98, 1.0),
                                  opacity=0.15, roughness=0.05), UsdShade.Tokens.strongerThanDescendants)
        print(f"[look] {name}: own {np.round(ohi - olo, 3).tolist()}, look {np.round(fit.max(0) - fit.min(0), 3).tolist()} m, "
              f"bottoms {olo[2]:.3f} {fit.min(0)[2]:.3f}", flush=True)

    def _dress(self, name, d, label_paths):
        """Make a LabUtopia enclosure look like the appliance the task names, with the cues a
        real one has (`dress:` in the config). Task 17's reagent fridge is a drying oven: models
        read it as an oven, whatever its sign said (Oct 2026). Looks only: collisions, the door's
        hinge and the handle are untouched.

          color       every mesh of the object repainted (white enamel), except the parts under
                      `keep` (paths relative to the object) and those `parts` repaints otherwise:
                      {part: {color, metallic, roughness}}
          glass_door  {prim, frame}: the door's mesh hidden (it still collides and swings) and, on
                      it, a frame `frame` m wide round a glass pane
          display     {prim, text}: a lit readout ("+4.0 °C") at the top of a control panel
          contents    things on its shelves, seen through the glass: {kind: bottle | carton,
                      at: [x, y], stand_on: <part> (on its top) | <z> (world height), ...}.
                      Not `on:`: YAML reads that key as true. No ray cast for the floor: none
                      hits before physics has stepped. Visual only, kept clear of `place_at`
          light       {at, size, intensity}: an LED strip under the roof, as glass-door
                      fridges have (world position, x and y size)
        """
        root = self.objects[name]
        frame = self.obj_frame(name)
        looks = f"{fixtures.ROOT}/Looks/{name}"
        prim_of = lambda rel: self.stage.GetPrimAtPath(f"{root}/{rel}")
        under = lambda path, rels: any(path == f"{root}/{r}" or path.startswith(f"{root}/{r}/") for r in rels)

        paint = fixtures.material(self.stage, f"{looks}/paint", color=d.get("color", (0.92, 0.92, 0.9)), roughness=0.35)
        parts = {rel: fixtures.material(self.stage, f"{looks}/part_{i}", **spec)
                 for i, (rel, spec) in enumerate(d.get("parts", {}).items())}
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)):
            path = str(prim.GetPath())
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)) or under(path, d.get("keep", [])):
                continue
            mat = next((m for rel, m in parts.items() if under(path, [rel])), paint)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
            self._weaken_ancestor_bindings(prim)

        if "glass_door" in d:
            g = d["glass_door"]
            door = prim_of(g["prim"])
            lo, hi = self._frame_range(str(door.GetPath()), frame)
            UsdShade.MaterialBindingAPI.Apply(door).Bind(fixtures.material(self.stage, f"{looks}/hidden", opacity=0.0))
            to_local = UsdGeom.Xformable(door).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            b = g.get("frame", 0.045)
            (x0, y0, z0), (x1, y1, z1) = lo, hi
            rails = {"stile_l": ((x0, y0, z0), (x1, y0 + b, z1)), "stile_r": ((x0, y1 - b, z0), (x1, y1, z1)),
                     "rail_b": ((x0, y0 + b, z0), (x1, y1 - b, z0 + b)), "rail_t": ((x0, y0 + b, z1 - b), (x1, y1 - b, z1))}
            for part, (a, c) in rails.items():
                fixtures.box_on(self.stage, door.GetPath().AppendChild(f"spai_{part}"),
                                frame.to_world(fixtures.box_corners(a, c)), to_local, paint)
            xm = (x0 + x1) / 2
            glass = fixtures.material(self.stage, f"{looks}/glass", color=(0.82, 0.9, 0.92), roughness=0.05,
                                      opacity=g.get("opacity", 0.12))
            fixtures.box_on(self.stage, door.GetPath().AppendChild("spai_glass"),
                            frame.to_world(fixtures.box_corners((xm - 0.002, y0 + b, z0 + b), (xm + 0.002, y1 - b, z1 - b))),
                            to_local, glass)

        if "display" in d:
            disp = d["display"]
            lo, hi = self._frame_range(f"{root}/{disp['prim']}", frame)
            out = Path(next(iter(label_paths.values()))).parent if label_paths else fixtures.TEXTURES
            png = fixtures.display_texture(disp["text"], out / f"display_{name}.png")
            w = 0.8 * (hi[1] - lo[1])
            h = w * 200 / 440
            yc, top, x = (lo[1] + hi[1]) / 2, hi[2] - 0.012, lo[0] - 0.002
            corners = [(x, yc + w / 2, top - h), (x, yc - w / 2, top - h), (x, yc - w / 2, top), (x, yc + w / 2, top)]
            # In the world, not on a body: the box stands still, and its first rigid body is
            # the panel's push button.
            fixtures.quad_on(self.stage, f"{fixtures.ROOT}/{name}_display", frame.to_world(corners), Gf.Matrix4d(1),
                             fixtures.material(self.stage, f"{looks}/display", texture=png, emissive=True, roughness=0.2))

        UsdGeom.Xform.Define(self.stage, f"{fixtures.ROOT}/{name}_contents")
        for i, c in enumerate(d.get("contents", [])):
            x, y = c["at"]
            on = c["stand_on"]
            z = on if isinstance(on, (int, float)) else self._frame_range(f"{root}/{on}", frame)[1][2]
            path = f"{fixtures.ROOT}/{name}_contents/item{i}"
            if c["kind"] == "bottle":
                fixtures.bottle(self.stage, path, (x, y, z), c.get("radius", 0.025), c.get("height", 0.11),
                                c.get("color", (0.42, 0.2, 0.05)), c.get("cap", (0.1, 0.1, 0.1)), glass=c.get("glass", False))
            elif c["kind"] == "carton":
                fixtures.carton(self.stage, path, (x, y, z), c.get("size", (0.1, 0.07, 0.06)),
                                c.get("color", (0.9, 0.88, 0.82)), c.get("yaw", 0.0))
            else:
                raise ValueError(f"{name}: unknown contents kind {c['kind']!r}: bottle or carton")

        if "light" in d:  # an LED strip under the roof, shining down
            lamp = UsdLux.RectLight.Define(self.stage, f"{fixtures.ROOT}/{name}_lamp")
            lamp.CreateWidthAttr(float(d["light"]["size"][0]))
            lamp.CreateHeightAttr(float(d["light"]["size"][1]))
            lamp.CreateIntensityAttr(float(d["light"]["intensity"]))
            lamp.CreateColorTemperatureAttr(5500.0)
            lamp.CreateEnableColorTemperatureAttr(True)
            xf = UsdGeom.Xformable(lamp.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, d["light"]["at"])))
            xf.AddRotateZOp().Set(float(self.spai["objects"][name].get("yaw", 0.0)))

    def _attach_labels(self, label_paths):
        """A textured card on the side of each object that faces the robot (-x), parented to
        the object's rigid body so it moves with it."""
        from PIL import Image
        # Kept across calls: write() adds a card to a scene that already has its labels.
        self.label_targets = getattr(self, "label_targets", {})  # name -> (body prim, intended world corners), for frame_cameras.py
        self.label_cards = getattr(self, "label_cards", {})  # name -> the card's prim path (peel takes it off)
        for name, png in label_paths.items():
            # "<object>:print" is text printed on the object's body (planning_context.object_prints):
            # a card above its label that peel leaves, so a sticker can come off a container that
            # still says what it holds.
            base, _, part = name.partition(":")
            if base in self.self_labelled or base not in self.objects:
                continue
            ocfg = self.spai["objects"][base]
            # `label_prim` puts the card on a part that stays put, e.g. an oven's body, not its door
            prim_path = ocfg.get("label_prim", self.objects[base])
            body = self.body_prim(prim_path)
            # A box aligned with the object's frame (the world, unless the config turns it).
            # LabUtopia's get_object_size transforms the corners of a box that is rotated in these
            # assets, so its "x size" can be an object's height.
            frame = self.obj_frame(base)
            lo, hi = self._frame_range(prim_path, frame)
            c, size = (lo + hi) / 2, hi - lo
            # An object's own `label_width` (a sign on a fridge door) overrides the scene's.
            # A wrapped label may run round up to half the can, not just its width seen face on.
            most = 0.9 * (np.pi * size[0] / 2 if ocfg.get("label_wrap") else size[1])
            w = min(ocfg.get("label_width", self.spai.get("label_width", 0.07)), most) * (0.7 if part else 1)
            img_w, img_h = Image.open(png).size
            h = min(w * img_h / img_w, 0.9 * size[2])
            w = h * img_w / img_h
            x = c[0] - size[0] / 2 - 0.003
            c[2] += ocfg.get("label_raise", 0.0)  # e.g. a sign at eye level on a door
            spans = self.__dict__.setdefault("label_spans", {})  # each label's (bottom, top), the object's frame
            if part:
                # Written on the free part of the face, just above its label, or below it if there is no room
                # above, shrunk to fit: 30 % up from the middle, the label task's writing lay over the old
                # label (Ben, 16 Nov 2026).
                z = c[2] + 0.30 * size[2]
                if base in spans:
                    b, t = spans[base]
                    for room, at in ((hi[2] - 0.006 - (t + 0.004), lambda hh: t + 0.004 + hh / 2),
                                     ((b - 0.004) - (lo[2] + 0.006), lambda hh: b - 0.004 - hh / 2)):
                        if room >= 0.012:
                            if h > room:
                                h, w = room, room * img_w / img_h
                            z = at(h)
                            break
                c[2] = z
            else:
                spans[name] = (c[2] - h / 2, c[2] + h / 2)
            # Viewed from -x with z up, the viewer's right is -y: u runs towards -y.
            world_pts = [Gf.Vec3d(*frame.to_world(p)) for p in (
                (x, c[1] + w / 2, c[2] - h / 2), (x, c[1] - w / 2, c[2] - h / 2),
                (x, c[1] - w / 2, c[2] + h / 2), (x, c[1] + w / 2, c[2] + h / 2))]
            to_local = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            self.label_targets[name] = (str(body.GetPath()), [list(map(float, p)) for p in world_pts])
            card = "spai_print" if part else "spai_label"
            if not part:
                self.label_cards[name] = str(body.GetPath().AppendChild(card))
            mesh = UsdGeom.Mesh.Define(self.stage, body.GetPath().AppendChild(card))
            if ocfg.get("label_wrap"):
                # Printed round a can (`label_wrap`): a strip of the cylinder just outside its
                # side, centred on the side facing the robot, `w` along the arc.
                r = size[0] / 2 + 0.0015
                cols = 24
                pts, uv = [], []
                for z, v in ((c[2] - h / 2, 0), (c[2] + h / 2, 1)):
                    for k in range(cols + 1):
                        a = np.pi + (k / cols - 0.5) * w / r  # from +y (u 0) round to -y (u 1)
                        pts.append(Gf.Vec3d(*frame.to_world((c[0] + r * np.cos(a), c[1] + r * np.sin(a), z))))
                        uv.append((k / cols, v))
                mesh.CreatePointsAttr([Gf.Vec3f(to_local.Transform(p)) for p in pts])
                mesh.CreateFaceVertexCountsAttr([4] * cols)
                mesh.CreateFaceVertexIndicesAttr([i for k in range(cols) for i in (k, k + 1, cols + 2 + k, cols + 1 + k)])
            else:
                pts, uv = world_pts, [(0, 0), (1, 0), (1, 1), (0, 1)]
                mesh.CreatePointsAttr([Gf.Vec3f(to_local.Transform(p)) for p in pts])
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)  # a flat card, not a subdivision surface
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(uv)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                self._label_material(f"/World/Looks/{card}_{base}", str(png), cutout=bool(part)), UsdShade.Tokens.strongerThanDescendants)
            self._weaken_ancestor_bindings(mesh.GetPrim())

    def _weaken_ancestor_bindings(self, prim):
        """Some assets bind their material to the whole object as strongerThanDescendants, which
        overrides the card's own binding (the card renders as glass). Make those bindings yield
        to descendants: the object's meshes keep inheriting them, the card keeps its label."""
        prim = prim.GetParent()
        while prim and str(prim.GetPath()) != "/":
            api = UsdShade.MaterialBindingAPI(prim)
            rel = api.GetDirectBindingRel()
            if rel and rel.GetTargets() and \
                    UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel) == UsdShade.Tokens.strongerThanDescendants:
                UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(rel, UsdShade.Tokens.weakerThanDescendants)
            prim = prim.GetParent()

    def _label_material(self, path, png, cutout=False):
        """A card's material. `cutout`: the texture's alpha cuts the card away, for text printed on
        the body (labels.render_print), so only the ink shows."""
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        reader = UsdShade.Shader.Define(self.stage, f"{path}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(self.stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        if cutout:
            alpha = tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(alpha)
            shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
        if self.spai.get("label_emissive"):
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def body_prim(self, path):
        """The prim that physics moves: the first one under `path` with a rigid body. LabUtopia's
        objects keep it on a child (often "<path>/mesh"), so the root's own transform is stale."""
        root = self.stage.GetPrimAtPath(path)
        stack = [root]
        while stack:
            prim = stack.pop(0)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
            stack.extend(prim.GetChildren())
        return root

    # ---- state the predicates read (<task dir>/scoring/predicates.py) ----

    def center(self, name):
        return self.utils.get_geometry_center(object_path=self.objects[name])

    def axis(self, name):
        """The middle of an object's shape, as grasps are worked out round it: its own collider's
        (sim/import_glassware.py), where it has one, else its centre. A beaker's spout widens its
        bounds on one side, 9 mm off its axis, and rim pinches 40 and 80 degrees round from the
        robot missed the wall by up to 7 mm."""
        path = f"{self.objects[name]}/{COLLIDER}"
        if not self.stage.GetPrimAtPath(path).IsValid():
            return self.center(name)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.guide]).ComputeWorldBound(
            self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return (np.array(box.GetMin()) + np.array(box.GetMax())) / 2

    def size(self, name):
        return self.utils.get_object_size(object_path=self.objects[name])

    def quat(self, name):
        """World orientation (x, y, z, w). The glassware's transforms carry tiny scales (beaker2's
        mesh 0.0005, conical_bottle02 0.0001), with which Gf's ExtractRotationQuat returns a
        non-unit quaternion that does not follow the object (16 Oct 2026): normalise the axes
        first. USD matrices act on row vectors, so the rotation is the transpose."""
        m = UsdGeom.Xformable(self.body_prim(self.objects[name])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows = np.array([[m[i][j] for j in range(3)] for i in range(3)])
        return R.from_matrix((rows / np.linalg.norm(rows, axis=1, keepdims=True)).T).as_quat()

    def pour_distance(self, source):
        prim_name = self.objects[source].split("/")[-1]
        return self.task_utils.get_pour_threshold(prim_name, self.size(source)) + 0.05

    def mouth(self, name):
        """The middle of a vessel's opening: half its standing height out from its centre along its
        own (then-vertical) axis. Anything else: its centre."""
        if name not in getattr(self, "_standing", {}):
            return self.center(name)
        q0, h = self._standing[name]
        axis = R.from_quat(self.quat(name)).apply(R.from_quat(q0).inv().apply([0, 0, 1]))
        return self.center(name) + axis * h / 2

    def above(self, a, b):
        """a's mouth is over b: within a's pour distance of b's centre (LabUtopia's test, there of
        a's centre) or over b's footprint (a sink is wider than any pour distance), and higher than
        b's centre (a sunk sink: over its opening). The mouth, not the centre, is where liquid leaves: a tall bottle tipped over a
        beaker has its centre well off to one side (20 Oct)."""
        c, lo, hi = self.mouth(a), *self.bounds(b)
        near = np.linalg.norm(c[:2] - self.center(b)[:2]) < self.pour_distance(a)
        over = bool(np.all(c[:2] >= lo[:2]) and np.all(c[:2] <= hi[:2]))
        bb = self.spai["objects"].get(b, {}).get("build", {})
        if bb.get("sunk"):
            # A sunk sink: over its opening and above its floor. Not within a pour distance of its
            # centre, nor higher than it (the middle of its bounds, from floor to tap): task 27's
            # bottle, lying on the bench by the tube, counted as over the sink and never spilled.
            pos, size = np.asarray(self.positions[b], float), np.asarray(bb["size"], float)
            inside = bool(np.all(np.abs(c[:2] - pos[:2]) <= size[:2] / 2))
            return bool(inside and c[2] > fixtures.basin_floor(bb, pos))
        return bool((near or over) and c[2] > self.center(b)[2])

    def pour_point(self, name):
        """Where pour() aims at a vessel: its config `pour_at` (e.g. a sink's drain, not the centre
        of a box that takes in the tap), else its centre."""
        at = self.spai["objects"][name].get("pour_at")
        return np.array(at, float) if at is not None else self.center(name)

    def tilt(self, name, ref):
        """Degrees the object has tipped since it had orientation `ref` (x, y, z, w), upright: the
        angle its then-vertical axis has turned from the vertical. Not the whole rotation, as
        LabUtopia's pour test takes it: a beaker turned about its own axis has not tipped, and
        picked, beaker2 can come up turned 180 degrees."""
        axis = R.from_quat(self.quat(name)).apply(R.from_quat(ref).inv().apply([0, 0, 1]))
        return float(np.degrees(np.arccos(np.clip(axis[2], -1, 1))))

    def pours_into(self, held, target):
        if self._hold_quat is None:
            return False
        return bool(self.tilt(held, self._hold_quat) > POUR_TILT_DEG and self.above(held, target))

    def bounds(self, name):
        """World-aligned (min, max) corners of an object."""
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            self.stage.GetPrimAtPath(self.objects[name])).ComputeAlignedRange()
        return np.array(box.GetMin()), np.array(box.GetMax())

    def obj_frame(self, name):
        """The object's own frame: the world turned by its config `yaw` about its origin."""
        return BoxFrame(self.spai["objects"][name].get("yaw", 0.0), self.positions[name])

    def _frame_range(self, prim_path, frame):
        """(min, max) corners of a prim's bounding box, aligned with `frame` and given in it."""
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            self.stage.GetPrimAtPath(prim_path))
        m, r = bbox.GetMatrix(), bbox.GetRange()
        pts = frame.to_frame(np.array([list(m.Transform(r.GetCorner(i))) for i in range(8)]))
        return pts.min(axis=0), pts.max(axis=0)

    def frame_bounds(self, name):
        """(min, max) corners of an object in its own frame: bounds() for an unturned one."""
        return self._frame_range(self.objects[name], self.obj_frame(name))

    def inside(self, a, b):
        """a's centre lies within b's bounds, in b's frame. An enclosure's are taken with its
        door shut, and the box must also be over a: a ray straight up from a's centre hits it.
        Without that, an object on the bench just in front of a box counts, within the handle's
        reach."""
        lo, hi = self.shut_bounds[b] if b in self.shut_bounds else self.frame_bounds(b)
        c = self.center(a)
        c_b = self.obj_frame(b).to_frame(c)
        if not (np.all(c_b >= lo) and np.all(c_b <= hi)):
            return False
        if b not in self.shut_bounds:
            return True
        from omni.physx import get_physx_scene_query_interface
        hits = []
        get_physx_scene_query_interface().raycast_all(
            list(map(float, c)), [0.0, 0.0, 1.0], float(hi[2] - c[2] + 0.05),
            lambda h: hits.append(h.collision) or True)
        return any(str(h).startswith(self.objects[b]) for h in hits)

    def handle_position(self, name):
        return self.utils.get_geometry_center(object_path=self.door(name)["handle"])

    def door_angle(self, name):
        """How far the door is open, in degrees: the handle's turn about the hinge since the start."""
        hinge = self.utils.get_revolute_joint_positions(joint_path=self.door(name)["joint"])[:2]
        a, b = self.shut_handle[name][:2] - hinge, self.handle_position(name)[:2] - hinge
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

    def closed(self, name):
        """The door is shut: it stands within DOOR_SHUT_DEG of where it started, about its hinge. Not
        the handle's distance from its start: with the box nudged on the bench, close() reported "the
        door is still 0 degrees open" 36 times, the sensor log said open (full-1110, 11 Nov 2026)."""
        return self.door_angle(name) < DOOR_SHUT_DEG

    def location(self, name):
        if name == self.held_object:
            return "gripper"
        for e, role in self.roles.items():
            if role == "enclosure" and e != name and self.inside(name, e):
                return e
        if self.waste_prim:
            waste = self.utils.get_geometry_center(object_path=self.waste_prim)
            if np.linalg.norm(self.center(name)[:2] - waste[:2]) < 0.08:
                return "waste"
        return "bench"

    def final_tags(self):
        return {n: sorted(t) for n, t in self.tags.items()}

    # ---- what the arm holds ----

    def _finger_width(self):
        return float(np.sum(self.robot.get_joint_positions()[7:9]))

    def _attach(self, obj):
        """Hold the grasped object with a fixed joint to the hand, where it sits in the fingers, so
        it cannot slip or be shaken out on the way; a skill removes it when it lets go (_detach)."""
        hand = self.stage.GetPrimAtPath(f"{self.robot.prim_path}/panda_hand")
        body = self.body_prim(self.objects[obj])
        world = lambda p: UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel = world(body) * world(hand).GetInverse()  # the object in the hand's frame (row vectors)
        m = np.array([[rel[i][j] for j in range(4)] for i in range(4)])
        rows = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=1, keepdims=True)  # drop the meshes' scales
        x, y, z, w = R.from_matrix(rows.T).as_quat()
        joint = UsdPhysics.FixedJoint.Define(self.stage, GRASP_JOINT)
        joint.CreateBody0Rel().SetTargets([hand.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, m[3, :3])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateExcludeFromArticulationAttr(True)
        self._grasp = {"obj": obj, "width": self._finger_width()}

    def _detach(self):
        if self._grasp:
            self.stage.RemovePrim(GRASP_JOINT)
            self._grasp = None

    # ---- stepping ----

    def trace(self, what, i, target=None):
        """With SPAI_TRACE=1: one line per 25 steps of a motion, past Isaac's hold on print()."""
        if not TRACE or i % 25:
            return
        import sys
        held = self.held_object
        sys.__stdout__.write(f"TRACE {what} {i:4d} target {np.round(target, 3) if target is not None else '-'} "
                             f"tool {np.round(self.robot.get_gripper_position(), 3)} held {held} "
                             f"obj {np.round(self.center(held), 3) if held else '-'}"
                             f"{f' tilt {self.tilt(held, self._hold_quat):.0f}' if held and self._hold_quat is not None else ''}"
                             f"{f' {self._watch} tilt {self.tilt(self._watch, self.upright[self._watch]):.0f}' if self._watch else ''}\n")
        sys.__stdout__.flush()

    def note(self, text):
        """With SPAI_TRACE=1: a line about a skill's choices (which grasp, why a plan failed)."""
        if TRACE:
            import sys
            sys.__stdout__.write(f"TRACE {text}\n")
            sys.__stdout__.flush()

    def step(self, action=None, render=False):
        """One physics step. The cameras render only when asked to (`render`, for a caller that
        reads a frame next) or on a step that records a video frame: ray tracing two cameras on
        every step is most of an episode's run time."""
        if action is not None:
            self.robot.get_articulation_controller().apply_action(action)
        video = self.on_frame is not None and round(self.t_sim / PHYSICS_DT + 1) % VIDEO_EVERY == 0
        if self.liquid and (render or video):
            self.liquid.follow()  # the particles to where their vessels are, for the frame
        self.world.step(render=render or video)
        self.t_sim += PHYSICS_DT
        self.pred.on_physics_step(self)
        if self.liquid:
            self.liquid.on_step()
        if video:
            self.on_frame(self)

    def settle(self, n, render=False):
        for _ in range(n):
            self.step(render=render)

    def frame(self, camera):
        """The camera's latest RGB image as a uint8 array, or None before its first render."""
        rgba = self.cameras[camera].get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def image_points(self, camera, points):
        """World points in the camera's image: for each, (u, v) as fractions of the width and
        height from the top left, or None if it is behind the camera or outside the frame. A
        point hidden behind something is still given. (As the photoreal branch's projection.)"""
        cam = self.cameras[camera]
        if not len(points):
            return []
        pos, quat = cam.get_world_pose(camera_axes="world")  # world axes: the camera looks along +x
        forward = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).apply([1.0, 0.0, 0.0])
        w, h = cam.get_resolution()
        uv = cam.get_image_coords_from_world_points(np.asarray(points, float))
        return [(float(u) / w, float(v) / h) if np.dot(np.asarray(p) - pos, forward) > 0 and 0 <= u < w and 0 <= v < h
                else None for p, (u, v) in zip(points, uv)]

    def marked(self):
        """What carries a number in the frames: every object, and every object's `switch` as a
        thing of its own, keyed "<object>:<switch>" ("sink:tap"). A tap is where a person looks
        to turn water on; with only the basin numbered, models found no number for it and never
        ran the tap (task 27, 2 of 3 DeepSeek seeds, 27 Oct 2026). Its number names its object."""
        return list(self.objects) + [f"{n}:{o['switch']}" for n, o in self.spai["objects"].items() if o.get("switch")]

    def switch_prim(self, name):
        """The prim a switch's number goes on: the config's `switch_prim`, else a built sink's tap."""
        return self.spai["objects"][name].get("switch_prim", f"{self.objects[name]}/tap_riser")

    def mark_bounds(self, key):
        obj, _, switch = key.partition(":")
        return self.bounds_of(self.switch_prim(obj)) if switch else self.bounds(obj)

    def mark_point(self, name, above=0.03):
        """Where a number goes: just above the middle of the object's top, so it does not cover
        what the object holds; a switch's, just above the top of the switch (a tap's riser). An
        object's `mark_offset: [dx, dy, dz]` moves it from there: the knife scene's board had its
        number on the knife lying across it, the knife's floated beside its blade, and the toy box's
        sat on the cube inside it; models picked the board for the knife 7 times (5 Nov run)."""
        p = self._mark_point(name, above)
        o = self.spai["objects"].get(name.partition(":")[0], {}) if ":" not in name else {}
        return p + np.asarray(o.get("mark_offset", [0, 0, 0]), float)

    def _mark_point(self, name, above):
        if ":" in name:
            lo, hi = self.mark_bounds(name)
            c = (lo + hi) / 2
            return np.array([c[0], c[1], hi[2] + above])
        o = self.spai["objects"][name]
        if "build" in o and not o["build"].get("dynamic") and o["build"]["kind"] in ("basin", "stripe", "block", "hob", "picture", "tub", "jar"):
            # a fixed fixture: above its own body (a sink's basin, not the tap over it)
            base = np.asarray(self.positions[name], float)
            top = fixtures.basin_rim(o["build"], base) if o["build"]["kind"] == "basin" else base[2] + o["build"]["size"][2]
            return np.array([base[0], base[1], top + above])
        lo, hi = self.bounds(name)
        c = (lo + hi) / 2  # the bounds' middle: some assets' geometry centre is off (the centrifuge's)
        return np.array([c[0], c[1], hi[2] + above])

    def robot_state(self):
        return {
            "joint_positions": self.robot.get_joint_positions(),
            "joint_velocities": self.robot.get_joint_velocities(),
            "gripper_position": self.robot.get_gripper_position(),
        }

    def go_to(self, position, euler_deg, steps=400, tol=None):
        """Move the hand to a pose with RMPflow, given as LabUtopia's controllers give one: xyz Euler
        angles in degrees through scipy's as_quat(). For the arm's start pose (`look_pose`, which
        aims the wrist camera) and frame_cameras.py's candidate poses, not for skills. With `tol`,
        stop once the tool centre is that near the position; returns whether it got there."""
        quat = R.from_euler("xyz", np.radians(euler_deg)).as_quat()
        target = np.array(position, float)
        self.rmp.reset()
        for _ in range(steps):
            self.step(self.rmp.forward(target_end_effector_position=target, target_end_effector_orientation=quat))
            if tol is not None and np.linalg.norm(self.robot.get_gripper_position() - target) < tol:
                return True
        return bool(np.linalg.norm(self.robot.get_gripper_position() - target) < (tol or 0.02))

    def move_to(self, position, speed=0.15, hold_steps=200, tol=0.015, quat=None):
        """Move the tool centre along a straight line at `speed` m/s by stepping RMPflow's target
        along it, the hand at `quat` (as RMPflow reads it, w x y z). Only where the hand pushes on
        something (_push_shut presses a door shut): the planner keeps the hand off everything.
        Returns whether the tool got within `tol`."""
        start = self.robot.get_gripper_position().copy()
        end = np.array(position, float)
        n = max(1, int(np.linalg.norm(end - start) / (speed * PHYSICS_DT)))
        self.rmp.reset()
        for i in range(1, n + hold_steps + 1):
            target = start + (end - start) * min(1.0, i / n)
            self.step(self.rmp.forward(target_end_effector_position=target, target_end_effector_orientation=quat))
            self.trace("move_to", i, target)
            if i >= n and np.linalg.norm(self.robot.get_gripper_position() - end) < tol:
                return True
        return False

    def heading(self, xy):
        """World heading (radians) and unit vector from the robot's base to (x, y)."""
        d = np.array(xy, float)[:2] - np.array(self.cfg["robot"]["position"], float)[:2]
        return float(np.arctan2(d[1], d[0])), d / np.linalg.norm(d)

    def at_hand(self, within=AT_HAND_M):
        """The objects (not the held one) whose bounds come within `within` of the tool centre."""
        tool = self.robot.get_gripper_position()
        out = []
        for n in self.objects:
            if n == self.held_object:
                continue
            lo, hi = self.bounds(n)
            if np.linalg.norm(np.clip(tool, lo, hi) - tool) < within:
                out.append(n)
        return out

    def beyond_reach(self, name):
        """Whether every part of an object lies further from the robot's base, across the floor,
        than any hand pose reaches (REACH_M): the far end of the bench, the centrifuge 1.3 m off.
        A call there fails as the scene's fact, not as a failed plan."""
        lo, hi = self.bounds(name)
        base = np.array(self.cfg["robot"]["position"], float)[:2]
        nearest = np.clip(base, lo[:2], hi[:2])
        return bool(np.linalg.norm(nearest - base) > REACH_M)

    def _turn_joint(self, j, delta, speed, hold=60):
        """Turn arm joint `j` by `delta` radians, the other joints held, with a smooth start and
        stop (half a cosine) at `speed` rad/s on average. Turning the base joint swings a held
        object round upright, which no planned motion promises."""
        q = self.robot.get_joint_positions().copy()
        n = max(1, int(abs(delta) / (speed * PHYSICS_DT)))
        # Velocity targets too: left as a plan or RMPflow set them, they jerk the arm.
        for i in range(1, n + hold):
            f = min(1.0, i / n)
            target = [None] * len(q)
            target[:7] = q[:7].tolist()
            target[j] = float(q[j] + delta * (1 - np.cos(np.pi * f)) / 2)
            vel = [None] * len(q)
            vel[:7] = [0.0] * 7
            vel[j] = float(delta * np.pi / (2 * n * PHYSICS_DT) * np.sin(np.pi * f)) if i <= n else 0.0
            self.step(ArticulationAction(joint_positions=target, joint_velocities=vel))
            self.trace(f"turn_joint{j}", i)
        self.rmp.reset()

    def rmp_hand_quat(self):
        """The hand's orientation now, as RMPflow reads a target (w, x, y, z): from RMPflow's own
        forward kinematics, so a tip (_tip) starts from exactly where the hand is."""
        _, rot = self.rmp.get_motion_policy().get_end_effector_pose(self.robot.get_joint_positions()[:7])
        x, y, z, w = R.from_matrix(rot).as_quat()
        return np.array([w, x, y, z])

    def go_ready(self):
        """Back to the default configuration, along a collision-free plan. If the hand is too close
        to something to plan from (just let go of it, fingers still round it), whatever it touches
        is left out of the plan; if even that fails, straight there in joint space."""
        traj, why = self.motion.plan_joints(self.ready_joints[:7])
        if traj is None:
            traj, why = self.motion.plan_joints(self.ready_joints[:7], ignore=self._near_hand())
        if traj is None and why == "invalid_start_state_world_collision":
            # Touching scenery, which no `ignore` leaves out (the wall cabinet over the mug task's
            # monitor, 13 Nov 2026): the hand back along its own axis and down, with RMPflow, which
            # plans nothing, then once more.
            rot = self._hand_rot()
            x, y, z, w = rot.as_quat()
            self.note("go_ready: the arm starts touching scenery; backing off")
            self.move_to(self.robot.get_gripper_position() - BACK_OFF_M * rot.apply([0, 0, 1]) - [0, 0, BACK_OFF_M],
                         speed=0.05, tol=0.03, quat=np.array([w, x, y, z]))
            traj, why = self.motion.plan_joints(self.ready_joints[:7], ignore=self._near_hand())
        if traj is not None:
            self.motion.follow(traj)
        else:
            self.note(f"go_ready: no plan ({why}), joint space")
            target = [None] * len(self.ready_joints)
            target[:7] = self.ready_joints[:7].tolist()
            for _ in range(300):
                self.step(ArticulationAction(joint_positions=target))
                if np.max(np.abs(self.robot.get_joint_positions()[:7] - self.ready_joints[:7])) < 0.02:
                    break
        self.rmp.reset()

    def _near_hand(self, r=0.08):
        tool = self.robot.get_gripper_position()
        out = []
        for n in self.objects:
            lo, hi = self.bounds(n)
            if np.linalg.norm(np.clip(tool, lo, hi) - tool) < r:
                out.append(n)
        return out

    def _run(self, ctrl, forward, door=None):
        """Step one of LabUtopia's atomic controllers until it reports done. False on timeout. The
        controllers return (action, record array); the record is for LabUtopia's dataset collection.
        `door`: an enclosure whose door angle the trace follows."""
        for i in range(MAX_SKILL_STEPS):
            if ctrl.is_done():
                return True
            action, _ = forward(self.robot_state())
            self.step(action)
            self.trace(ctrl.name, i)
            if door and not i % 50:
                self.note(f"{ctrl.name} {door}: step {i}, door at {self.door_angle(door):.0f} deg")
        return False

    def _fingers(self, action, width=0.0, steps=60):
        """Open the fingers, or close them to `width` apart (0: shut), until they stop moving
        (closed on something, or as far as they go)."""
        if action == "open":
            act = self.robot.gripper.forward(action="open")
        else:
            pos = [None] * len(self.robot.get_joint_positions())
            pos[7:9] = [width / 2, width / 2]
            act = ArticulationAction(joint_positions=pos)
        last = self._finger_width()
        still = 0
        for i in range(steps):
            self.step(act)
            w = self._finger_width()
            still = still + 1 if abs(w - last) < 2e-5 else 0
            last = w
            if i > 10 and still > 5:
                break

    # ---- the shape of things, for grasps and put-downs ----

    def _profiles(self):
        """Each pickable object's shape about its vertical axis, as it stands at the start: its
        half-height, and its radius in 5 mm slices from the bottom up ([(height above its centre,
        radius)]). Read off the meshes' points, so a conical flask's narrow neck and a beaker's wide
        mouth are what they are; between points far apart in height (a box's side), straight. Grasps (_grasps) and pours (the lip) are worked out from it."""
        out = {}
        for n in self.objects:
            if self.roles[n] not in PICKABLE:
                continue
            pts = []
            # A file's own collider (sim/import_glassware.py), where it has one: the glassware's
            # turned shell, without the model's spout, which would widen its rim on one side.
            prims = list(Usd.PrimRange(self.stage.GetPrimAtPath(self.objects[n])))
            own = [p for p in prims if p.IsA(UsdGeom.Mesh) and p.GetName() == COLLIDER]
            for prim in own or prims:
                if not prim.IsA(UsdGeom.Mesh) or (prim.GetName().startswith("spai_") and not own):
                    continue
                p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], float)
                if len(p):
                    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                    pts.append(p @ m[:3, :3] + m[3, :3])
            pts = np.concatenate(pts) - self.axis(n)
            r = np.hypot(pts[:, 0], pts[:, 1])
            lo, hi = pts[:, 2].min(), pts[:, 2].max()
            edges = np.arange(lo, hi + SLICE_M, SLICE_M)
            mids = (edges[:-1] + edges[1:]) / 2
            bands = [r[(pts[:, 2] >= a) & (pts[:, 2] < b)] for a, b in zip(edges[:-1], edges[1:])]
            radii = np.array([band.max() if len(band) else np.nan for band in bands])
            # A slice with no points in it lies along a face that spans it (a box's side has points
            # only at its ends): its radius runs straight between the slices either side.
            held = ~np.isnan(radii)
            radii = np.interp(mids, mids[held], radii[held])
            slices = [(float(z), float(rad)) for z, rad in zip(mids, radii)]
            out[n] = {"half": float((hi - lo) / 2), "slices": slices, "top_r": slices[-1][1]}
        return out

    def _grasps(self, obj):
        """Grasps for obj as it stands now, best first: [{kind, tool, rot, approach}], `tool` the
        tool centre's world position at the grasp, `rot` the hand's rotation (z the approach, y the
        line the fingers close along), `approach` the hand's way in.

          side  the hand level, in along the line from the robot's base (or 25, 50 and 75 degrees
                off it: out of a fridge, the hand must point straight in through its door), closing round the object where it is narrower than the fingers open: a
                bottle's neck. The highest such band, below the fingertips' depth from the top
          top   pointing down, closing round a narrow top (a bottle in a sink, whose walls block
                the side)
          rim   pointing down, pinching a wide vessel's wall at its rim (a beaker, wider than the
                hand opens), at the point nearest the robot or 40 and 80 degrees round
          handle  pointing down, closing across a pan's or a knife's handle (`grasp: handle`; its
                spai_handle prim, which the profile leaves out), halfway along it, no deeper than
                HANDLE_GRIP_M from its top: a knife's handle lies 2 cm off the board, and the
                fingertips round its middle would touch the board
        """
        if self.spai["objects"][obj].get("grasp") == "handle":
            lo, hi = self.bounds_of(f"{self.objects[obj]}/spai_handle")
            mid = (lo + hi) / 2
            mid[2] = max(mid[2], hi[2] - HANDLE_GRIP_M)
            out = self.center(obj)
            along = (mid - out)[:2] / np.linalg.norm((mid - out)[:2])  # the handle's way out from the pan
            down = np.array([0.0, 0.0, -1.0])
            return [{"kind": "handle", "tool": mid, "approach": down,
                     "rot": hand_rotation(down, [-s * along[1], s * along[0], 0.0])} for s in (1, -1)]
        p = self.profile[obj]
        c = self.axis(obj)
        phi, _ = self.heading(c)
        top = p["half"]
        bottom = -top
        out = []
        narrow = [z for z, r in p["slices"] if 2 * r < GRIP_OPEN_M and bottom + 0.02 < z < top - FINGER_DEPTH_M / 2]
        if narrow:
            z = max(narrow)
            close_to = max(0.0, 2 * dict(p["slices"])[z] - SIDE_GRIP_M)
            # Both rolls of the hand (the fingers are alike): its back down first, then up.
            for roll in (1, -1):
                for d in (0, -25, 25, -50, 50, -75, 75):
                    a = phi + np.radians(d)
                    approach = np.array([np.cos(a), np.sin(a), 0.0])
                    out.append({"kind": "side", "tool": c + [0, 0, z], "approach": approach, "close_to": close_to,
                                "rot": hand_rotation(approach, [-roll * np.sin(a), roll * np.cos(a), 0.0])})
        top_band = [r for z, r in p["slices"] if z > top - FINGER_DEPTH_M]
        down = np.array([0.0, 0.0, -1.0])
        if top_band and 2 * max(top_band) < GRIP_OPEN_M:
            for d in (0, 90):
                a = phi + np.radians(d)
                out.append({"kind": "top", "tool": c + [0, 0, top - FINGER_DEPTH_M / 2], "approach": down,
                            "rot": hand_rotation(down, [-np.sin(a), np.cos(a), 0.0])})
        elif self.roles[obj] == "vessel":
            rim_r = p["top_r"] - 0.001  # on the glass
            for d in (0, -40, 40, -80, 80):
                a = phi + np.pi + np.radians(d)  # the rim point towards the robot, and round from it
                radial = np.array([np.cos(a), np.sin(a), 0.0])
                out.append({"kind": "rim", "tool": c + rim_r * radial + [0, 0, top - RIM_DEPTH_M], "approach": down,
                            "rot": hand_rotation(down, radial), "close_to": RIM_GRIP_M})
        return out

    # ---- skills (skills.yaml's physical skills) ----
    # Each returns (status, note). status: ok | failed; the note is shown to the model. Every skill
    # checks what it did in the scene before it says ok.

    def pick(self, obj):
        """Grasp obj and lift it. The grasp is worked out from the object's shape (_grasps): the
        first one the arm can reach and lift from without touching anything, by a collision-free path to a
        point APPROACH_M back along its way in, then straight in. The fingers close until they stop;
        if they closed on something, the object is fixed in the hand (_attach) and lifted straight
        up. Fails, with the reason, unless it rose with the hand, upright."""
        o = self.spai["objects"][obj]
        if self.roles[obj] == "picture":  # a thing shown on a screen: the hand goes for it
            return self._reach_into(obj)
        dynamic = self._built_movable(obj)
        if self.roles[obj] not in PICKABLE or ("build" in o and not dynamic) \
                or ("usd" in o and not o.get("dynamic")):
            return "failed", f"{obj} cannot be picked up"
        if self.held_object:
            return "failed", f"already holding {self.held_object}"
        if self.beyond_reach(obj):
            return "failed", f"{obj} is out of the arm's reach"
        if obj in getattr(self, "pinned", ()):
            # Freed before the hand comes, and left a few steps to stand on its own.
            self._pin(obj, False)
            self.settle(10)
        standing = self.upright[obj]
        if self.tilt(obj, standing) > FALLEN_DEG:
            return "failed", f"{obj} has fallen over; it cannot be grasped lying down"
        start = self.center(obj)
        self._fingers("open")
        # The first grasp the arm can reach, then straight in; where the way in has no plan, the next
        # (up to PICK_TRIES): with the toy box moved left, the duck's first grasp came from its side
        # and every pick of it stopped there (7 Nov 2026).
        grasps = [g for g in self._grasps(obj) if o.get("grasp", g["kind"]) == g["kind"]]  # the scene may allow one kind (the lighter: side)
        tried, why = set(), "no grasp fits it"
        for _ in range(PICK_TRIES):
            chosen = None
            for i, g in enumerate(grasps):
                if i in tried:
                    continue
                # Reachable to grasp, and to lift: 0.8 m out, a level hand can close round a bottle in the
                # sink but not lift it 12 cm (21 Oct 2026); turned 25 degrees it can.
                if not self.motion.reachable(g["tool"], g["rot"], ignore={obj}):
                    continue
                if not self.motion.reachable(g["tool"] + [0, 0, LIFT_M], g["rot"], ignore={obj}):
                    # Something hangs over it (the household candle's towel): slide it out first,
                    # back the way the hand came in, then lift.
                    if g["kind"] != "side":
                        continue
                    slide = g["tool"] - SLIDE_M * g["approach"] + [0, 0, SLIDE_RISE_M]
                    if not (self.motion.reachable(slide, g["rot"], ignore={obj})
                            and self.motion.reachable(slide + [0, 0, LIFT_M], g["rot"], ignore={obj})):
                        continue
                    g = {**g, "slide": slide}
                # Twice: one failed plan stopped the knife task's duck and knife picks, 2 scripted
                # replays in 4 (13 Nov 2026).
                for _ in range(PLAN_TRIES):
                    pre, _ = self.motion.plan(g["tool"] - APPROACH_M * g["approach"], g["rot"])
                    if pre is not None:
                        break
                if pre is None:
                    why = "no way to reach it without touching something"
                    continue
                chosen = (i, g)
                break
            if chosen is None:
                self.go_ready()
                return "failed", why
            i, g = chosen
            tried.add(i)
            self.note(f"pick {obj}: fingers open {self._finger_width():.4f}, object at {np.round(self.center(obj), 3)}")
            self.note(f"pick {obj}: {g['kind']} grasp at {np.round(g['tool'], 3)}, profile half {self.profile[obj]['half']:.3f} "
                      f"top r {self.profile[obj]['top_r']:.3f}")
            self.motion.follow(pre)
            inward, w = self.motion.line(g["tool"], g["rot"], "axis", ignore={obj})
            if inward is None:
                why = f"could not move in to grasp it ({w})"
                self.go_ready()
                continue
            # Paced, and held until it is there: some ways in swing the arm round through its spare
            # joint, faster than the drives follow; the joints ended 13-28 degrees behind, the tool
            # 8-12 cm short, and the fingers closed on air (the bleach clean-down, trace02, 15 Nov 2026).
            self.motion.follow(self.motion.paced(inward), hold=INWARD_HOLD_STEPS)
            tool_at = self.robot.get_gripper_position()
            self.note(f"pick {obj}: in, tool {np.round(tool_at, 3)}, object {np.round(self.center(obj), 3)}, "
                      f"hand y {np.round(self.motion.hand_rotations(self.robot.get_joint_positions()[None, :7])[0].apply([0, 1, 0]), 2)}")
            short = float(np.linalg.norm(tool_at - g["tool"]))
            if short > INWARD_TOL_M:
                why = f"could not move in to grasp it (the hand stopped {short * 100:.0f} cm short)"
                self._back_out(g, obj)
                if self.tilt(obj, standing) > FALLEN_DEG:
                    return "failed", why
                continue
            self._fingers("close", g.get("close_to", 0.0))
            self.note(f"pick {obj}: tool {np.round(self.robot.get_gripper_position(), 3)} aimed {np.round(g['tool'], 3)} "
                      f"object {np.round(self.center(obj), 3)} fingers {self._finger_width():.4f}")
            if not self._round(obj, g):
                self._fingers("open")
                self._back_out(g, obj)
                return "failed", "the fingers closed on nothing"
            self._attach(obj)
            self._hold_quat = standing
            if "slide" in g:  # out from under what hangs over it, low over the surface it stood on
                out, why = self.motion.straight(g["slide"], self._hand_rot(), ignore={obj})
                if out is not None:
                    self.motion.follow(out)
            # The planner is not deterministic, and one failed plan is no reason to report a failed pick:
            # 25 of 1047 model picks ended so, 19 of 130 where the rinse task lifts its flask (30 Oct 2026).
            # Near the edge of the arm's reach the full height has no arm pose at all (the cylinder at
            # y -0.42, 31 Oct 2026), so the last tries lift less; LIFTED_M still has to be met.
            for frac in LIFT_TRIES:
                lift, why = self.motion.line(self.robot.get_gripper_position() + [0, 0, frac * LIFT_M], self._hand_rot(), "vertical", ignore={obj})
                if lift is not None:
                    self.motion.follow(lift)
                    break
            if lift is not None or "slide" in g:
                break
            # No way up from this grasp: let go, back out and take the next. The graduated cylinder's
            # side grasps from the left had an arm pose at the top of the lift but none on the way
            # (the bleach clean-down, 3 replays in 18, trace02, 15 Nov 2026).
            self.note(f"pick {obj}: no way up from this grasp ({why}); trying the next")
            self._release(obj)
            self._fingers("open")
            self._back_out(g, obj)
            why = f"it did not come up with the hand (no way up: {why})"
            if self.tilt(obj, standing) > FALLEN_DEG:
                return "failed", why
        else:
            self.go_ready()
            return "failed", why
        self.settle(20)
        c = self.center(obj)
        tilt = self.tilt(obj, standing)
        near = np.linalg.norm(c - self.robot.get_gripper_position()) < 0.25
        self.note(f"pick {obj}: lifted, object {np.round(c, 3)} (rose {c[2] - start[2]:.3f}), "
                  f"tool {np.round(self.robot.get_gripper_position(), 3)}, tilt {tilt:.0f}")
        if c[2] < start[2] + LIFTED_M or not near:
            self._release(obj)
            self._fingers("open")
            self.go_ready()
            return "failed", "it did not come up with the hand" + ("" if lift is not None else f" (no way up: {why})")
        if tilt > HELD_TILT_DEG:
            self._release(obj)
            self._fingers("open")
            self.go_ready()
            return "failed", f"it tipped {tilt:.0f} degrees in the grasp and was let go"
        self.held_object = obj
        self.motion.attach(obj)
        self._hold = {"kind": g["kind"], "height": 2 * self.profile[obj]["half"], "lip_r": self.profile[obj]["top_r"]}
        return "ok", None

    def _round(self, obj, g):
        """Did the fingers close round obj? Stopped short of where they were sent (on something), or,
        for a rim pinch (sent nearly shut, on a wall thinner than that), the rim between them, or,
        for a side grasp (sent just under its width, on a wall that gives), its axis between them."""
        if self._finger_width() > g.get("close_to", 0.0) + TOUCH_M:
            return True
        if g["kind"] not in ("rim", "side"):
            return False
        tool, c = self.robot.get_gripper_position(), self.axis(obj)
        if g["kind"] == "side":
            return bool(np.hypot(*(tool[:2] - c[:2])) < g["close_to"] / 2)
        off = abs(np.hypot(*(tool[:2] - c[:2])) - self.profile[obj]["top_r"])
        return bool(off < g["close_to"] / 2 + RIM_REACH_M)

    def _release(self, obj):
        self._detach()
        self.motion.detach()
        self.held_object, self._hold, self._hold_quat = None, None, None

    def _back_out(self, g, obj):
        """Straight back out the way the hand went in, then to the default configuration."""
        tool = self.robot.get_gripper_position()
        out, _ = self.motion.line(tool - APPROACH_M * g["approach"], g["rot"], "axis", ignore={obj})
        if out is not None:
            self.motion.follow(out)
        self.go_ready()

    def _carry(self, obj, goal, ignore=(), yaw=0.0, swing=True, straight=False, tol=0.02):
        """Carry the held object so that its centre ends at `goal`, upright. The hand and object
        turn together about the vertical by as much as the heading from the base turns, so the arm
        holds it at the goal as it held it when it grasped it. The planned path must not lean the
        object more than CARRY_TILT_DEG; if it does, the base joint first swings it round (which
        keeps it upright exactly) and the plan holds the hand's orientation. With `straight`, a
        straight line is tried first: for a short last move with something left out of the
        obstacles, which a free plan can loop through. `tol`: how near its goal it must end.
        Returns why not, or None."""
        rot = self._hand_rot()
        goal_tool, goal_rot, turn = self._carried_to(obj, goal, yaw)
        if straight:
            traj, why = self.motion.straight(goal_tool, goal_rot, ignore=ignore)
            self.note(f"carry {obj}: straight line {'found' if traj is not None else 'refused: ' + why}")
            if traj is not None:
                self.motion.follow(traj)
                return self._arrived(obj, goal, tol)
        # cuRobo's plans differ from call to call (random seeds): a few tries for one that does not lean.
        for _ in range(CARRY_TRIES):
            traj, why = self.motion.plan(goal_tool, goal_rot, ignore=ignore)
            if traj is None:
                break
            if self.motion.lean(traj, rot) <= CARRY_TILT_DEG:
                self.motion.follow(traj)
                return self._arrived(obj, goal, tol)
        if not swing:
            return why if traj is None else "every path leans it over"
        self.note(f"carry {obj}: {'leans ' + str(round(self.motion.lean(traj, rot))) + ' deg' if traj is not None else why}; swing first")
        # The swing turns only the base joint; checked for collisions first, from higher up if it
        # would hit something.
        delta = turn.as_rotvec()[2]
        for rise in (0.0, 0.08, 0.16):
            if rise:
                up, _ = self.motion.plan(self.robot.get_gripper_position() + [0, 0, 0.08], self._hand_rot(),
                                         hold="vertical", ignore=ignore)
                if up is None:
                    break
                self.motion.follow(up)
            q = self.robot.get_joint_positions()[:7]
            arc = np.repeat(q[None], 24, axis=0)
            arc[:, 0] += np.linspace(0, delta, 24)
            if self.motion.clear(arc, ignore=ignore):
                break
        if not self.motion.clear(arc, ignore=ignore):
            rot = self._hand_rot()
            goal_tool, goal_rot, _ = self._carried_to(obj, goal, yaw)
            for _ in range(CARRY_TRIES):
                traj, _why = self.motion.plan(goal_tool, goal_rot, ignore=ignore)
                if traj is not None and self.motion.lean(traj, rot) <= CARRY_LAST_DEG:
                    self.motion.follow(traj)
                    return self._arrived(obj, goal, tol)
            return "no way to swing it round clear of the bench"
        self._turn_joint(0, delta, 0.4)
        goal_tool, goal_rot, _ = self._carried_to(obj, goal, yaw)
        # First a straight line, the hand level all the way. Then to the goal joints nearest the
        # arm's now (the wrist not wound round), planned in joint space; then cuRobo holding the
        # orientation, which is a cost to it, not a promise: a held plan once rolled the beaker
        # right over and back (task 17, 22 Oct 2026). The last two checked like the free plans above.
        rot = self._hand_rot()
        traj, why = self.motion.straight(goal_tool, goal_rot, ignore=ignore)
        self.note(f"carry {obj}: straight line {'found' if traj is not None else 'refused: ' + why}")
        if traj is not None:
            self.motion.follow(traj)
            return self._arrived(obj, goal, tol)
        q_goal = self.motion.ik_near(goal_tool, goal_rot, ignore=ignore)
        self.note(f"carry {obj}: joints now {np.round(self.robot.get_joint_positions()[:7], 2)}, "
                  f"nearest at the goal {None if q_goal is None else np.round(q_goal, 2)}")
        if q_goal is not None:
            traj, why = self.motion.plan_joints(q_goal, ignore=ignore)
            if traj is not None:
                lean = self.motion.lean(traj, rot)
                self.note(f"carry {obj}: to the nearest joints, leans {lean:.0f} deg")
                if lean <= CARRY_TILT_DEG:
                    self.motion.follow(traj)
                    return self._arrived(obj, goal, tol)
        for _ in range(CARRY_TRIES):
            traj, why = self.motion.plan(goal_tool, goal_rot, hold="orientation", ignore=ignore)
            if traj is None:
                return why
            lean = self.motion.lean(traj, rot)
            self.note(f"carry {obj}: orientation held, leans {lean:.0f} deg")
            if lean <= CARRY_TILT_DEG:
                self.motion.follow(traj)
                return self._arrived(obj, goal, tol)
        return "every path leans it over"

    def _arrived(self, obj, goal, tol=0.02):
        off = float(np.linalg.norm(self.center(obj) - goal))
        self.note(f"carry {obj}: {off * 100:.1f} cm from its goal")
        return None if off < tol else f"it stopped {off * 100:.0f} cm short"

    def _hand_rot(self):
        return self.motion.hand_rotations(self.robot.get_joint_positions()[None, :7])[0]

    def _carried_to(self, obj, goal, yaw=0.0):
        """Where the hand goes to put the held object's centre at `goal`: (tool centre, hand
        rotation, the turn about the vertical), the hand turned as the heading from the base turns,
        and `yaw` (radians) more, round the object's axis."""
        tool, c = self.robot.get_gripper_position(), self.center(obj)
        turn = R.from_euler("z", self.heading(goal)[0] - self.heading(c)[0] + yaw)
        return np.asarray(goal, float) + turn.apply(tool - c), turn * self._hand_rot(), turn

    def _fits_on(self, obj, location, xy, turn, overhang=0.02):
        """Whether obj, its centre moved to xy and turned by `turn` (a rotation about the vertical),
        lies within location's top, give or take `overhang`: its bounds' corners round its centre,
        turned, against location's bounds."""
        lo, hi = self.bounds(obj)
        c = self.center(obj)
        corners = np.array([[x, y, 0.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - [c[0], c[1], 0.0]
        feet = turn.apply(corners)[:, :2] + np.asarray(xy, float)
        tlo, thi = self.bounds(location)
        return bool(np.all(feet >= tlo[:2] - overhang) and np.all(feet <= thi[:2] + overhang))

    def place(self, obj, location):
        """_place, with an assist where the location's config says `place_assist: true` (an
        enclosure, a sink, a toy box): where the arm fails to put the object there for reasons of
        its own (no reachable pose at the spot, no carry, a fall at set-down) after a spot was
        found, the object is set on that spot directly and the arm goes home (_assisted_place). What the model chose is then what the
        scene shows. A failure that is the model's doing (not holding it, the door shut, no spot)
        is not assisted. `last_assist` tells the episode, which marks the call."""
        self.last_assist, self._assistable = None, False
        if self.held_object == obj and self._closed(location):
            return "failed", self._sealed(location)
        status, note = self._place(obj, location)
        o = self.spai["objects"].get(location, {})
        if status == "failed" and self._assistable and o.get("place_assist"):
            return self._assisted_place(obj, location, note)
        return status, note

    def _assisted_place(self, obj, location, why):
        """Set obj upright on the spot _place chose (an enclosure's shelf spot, found again), the
        arm home. See place."""
        from isaacsim.core.prims import SingleRigidPrim
        if self.roles.get(location) == "enclosure":
            spot, _ = self._shelf(location, self.spai["objects"][location]["place_at"])
        else:
            spot = self._assist_spot
        if spot is None:  # the spot _place found is gone (the arm knocked something onto it)
            return "failed", why
        xy, support = spot
        if self.held_object == obj:
            self._fingers("open")
            self._release(obj)
        body = SingleRigidPrim(str(self.body_prim(self.objects[obj]).GetPath()))
        goal = np.array([*xy, support + self.profile[obj]["half"]])
        x, y, z, w = self.upright[obj]

        def set_down():
            # The body's origin is not the object's centre: set it, measure, correct once.
            for _ in range(2):
                pos, _ = body.get_world_pose()
                body.set_world_pose(position=np.asarray(pos) + goal - self.center(obj),
                                    orientation=np.array([w, x, y, z]))
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
                self.settle(5, render=True)

        set_down()
        self.go_ready()  # once the object is out of the hand, so the arm cannot carry it off
        if self.tilt(obj, self.upright[obj]) > UPRIGHT_DEG or (self.roles.get(location) == "enclosure"
                                                             and not self.inside(obj, location)):
            set_down()  # knocked on the way home
        self.settle(SETTLE_STEPS)
        self.last_assist = {"object": obj, "location": location, "why": why}
        self.note(f"place into {location}: assisted after {why}")
        return self._check_put(obj, location, xy, support)

    def knock_over(self, obj, near, gap=0.01):
        """A harness mishap (task.json's planning_context.mishap; never a skill): the held vessel
        slips from the hand and lands on its side beside `near`, its mouth towards it, `gap` off
        its side. Its contents run out as a puddle round `near` (sim/liquid.py, on the next
        steps: tipped over nothing, it spills). The arm goes home. `spills[obj]` says where."""
        from isaacsim.core.prims import SingleRigidPrim
        if self.held_object == obj:
            self._fingers("open")
            self._release(obj)
        at = self.center(near)
        d = self.center(obj)[:2] - at[:2]
        if np.linalg.norm(d) < 1e-3:  # straight above it: fall towards the robot
            d = np.asarray(self.cfg["robot"]["position"][:2], float) - at[:2]
        d = np.array([*(d / np.linalg.norm(d)), 0.0])
        h = self._standing[obj][1]
        r = max(rad for _, rad in self.profile[obj]["slices"])
        r_near = float(np.max(self.size(near)[:2])) / 2
        goal = np.array([*(at[:2] + d[:2] * (r_near + gap + h / 2)), self.initial_bottom[obj] + r + 0.002])
        # Its standing axis turned to lie along -d: a quarter turn about the level line across d.
        lying = R.from_rotvec(np.cross([0.0, 0.0, 1.0], -d) * np.pi / 2) * R.from_quat(self.upright[obj])
        x, y, z, w = lying.as_quat()
        body = SingleRigidPrim(str(self.body_prim(self.objects[obj]).GetPath()))
        held_back, self.liquid = self.liquid, None  # no spill until it lies where it will stay
        for _ in range(2):  # the body's origin is not the object's centre: set, measure, correct
            pos, _ = body.get_world_pose()
            off = R.from_quat(self.quat(obj)).inv().apply(self.center(obj) - np.asarray(pos))
            body.set_world_pose(position=goal - lying.apply(off), orientation=np.array([w, x, y, z]))
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
            self.settle(5)
        self.liquid = held_back
        drawn = len(self.liquid.spills_drawn) if self.liquid else 0
        self.settle(SETTLE_STEPS, render=True)
        self.go_ready()
        self.settle(SETTLE_STEPS)
        # Only a puddle this fall made: one from earlier (a pour that missed) is elsewhere. With
        # liquid drawn and none made, the vessel was empty: radius 0, and the episode says so.
        made = self.liquid and len(self.liquid.spills_drawn) > drawn
        mouth = [float(v) for v in self.mouth(obj)[:2]]
        if made:
            self.spills[obj] = {**self.liquid.puddles[obj], "empty": False}
        else:
            self.spills[obj] = {"center": mouth, "radius": 0.0 if self.liquid else SPILL_R_M, "empty": bool(self.liquid)}
        self.note(f"mishap: {obj} knocked over beside {near}, spill {self.spills[obj]}")

    def _place(self, obj, location):
        """Put the held object down: on an object (standing on it), on "bench" (back where it was
        picked from), in a sink (a free spot on its floor), in an open enclosure (its `place_at`, in
        through the door) or on "waste" (the waste location). The planner carries it to just above
        the spot, clear of everything; then it goes straight down until it touches, the fingers
        open, the hand backs out the way it went in and goes back to the default configuration.
        Fails, with the reason, unless the object ends standing where it was put."""
        if self.held_object != obj:
            return "failed", f"not holding {obj}; pick it up first"
        if location in self.objects and self.beyond_reach(location):
            return "failed", f"{location} is out of the arm's reach"
        spots, why = self._spots(obj, location)
        if not spots:
            return "failed", why
        self._assistable = True  # a spot exists: from here a failure is the arm's, not the model's
        self._assist_spot = spots[0]  # where an assisted put-down sets it (place_assist)
        half = self.profile[obj]["half"]
        # The first spot, and turn of the hand round the object's axis (PLACE_YAWS), where the arm
        # can hold the object both just above it and resting on it (checked without the object:
        # resting, it touches what it stands on). Measured 21 Oct 2026: at task 17's place_at,
        # 0.83 m out, the arm reaches only with the hand pointing within a few degrees of +x,
        # straight in through the fridge's door; pointing along the line from the base it cannot.
        # On a built block (the household knife task's counter), the turns that keep the object's
        # footprint on it come first: the 29 cm knife, carried lying across the 26 cm counter, came
        # down with its tip on the toy box's wall, 5 in 5 (7 Nov 2026).
        on_block = self.spai["objects"].get(location, {}).get("build", {}).get("kind") == "block"
        found, fitting = [], 0
        # Into a box with `over_rim`, the object is carried to clear its rim and lowered from there:
        # carried to just above the toy box's floor, the knife found a plan 1 run in 3 (8 Nov 2026).
        set_down = SET_DOWN_M
        if self.spai["objects"].get(location, {}).get("over_rim"):
            set_down = max(SET_DOWN_M, float(self.bounds(location)[1][2]) + OVER_RIM_M - min(s for _, s in spots))
        for xy, support in spots:
            for yaw in np.radians(PLACE_YAWS):
                above, rot, turn = self._carried_to(obj, [*xy, support + half + set_down], yaw)
                if not self.motion.reachable(above, rot):
                    continue
                self.motion.detach()
                low = self.motion.reachable(above - [0, 0, set_down], rot, ignore={obj})
                self.motion.attach(obj)
                if low:
                    fits = on_block and self._fits_on(obj, location, xy, turn)
                    found.append((xy, support, yaw, fits))
                    fitting += fits
                    if (fitting if on_block else len(found)) == PLACE_CHOICES:
                        break
            if (fitting if on_block else len(found)) == PLACE_CHOICES:
                break
        found = [f[:3] for f in sorted(found, key=lambda f: not f[3])][:PLACE_CHOICES]
        if found:
            self._assist_spot = found[0][:2]
        if not found:
            for xy, support in spots[:3]:
                above, rot, _ = self._carried_to(obj, [*xy, support + half + set_down], 0.0)
                self.note(f"place {obj}: spot {np.round(xy, 3)} on {support:.3f}, hand {np.round(above, 3)}: "
                          f"above {self.motion.reachable(above, rot)}, "
                          f"above without it {self.motion.reachable(above, rot, ignore={obj})}")
            return "failed", f"cannot reach to put {obj} {'in' if self.roles.get(location) == 'enclosure' else 'on'} {location}"
        # A carry that does not lean the object, to any of them; only then the base swing, to each in turn.
        for swing in (False, True):
            for xy, support, yaw in found:
                why = self._carry(obj, [*xy, support + half + set_down], yaw=yaw, swing=swing)
                if why is None:
                    break
            if why is None:
                break
        if why:
            return "failed", f"could not get {obj} to {location} ({why})"
        box = location if self.roles.get(location) == "enclosure" else None
        self._door_note(box, "carried in")
        self._lower_until_touch(obj, support)
        self._fingers("open")
        self._release(obj)
        self._door_note(box, "let go")
        rot = self.motion.hand_rotations(self.robot.get_joint_positions()[None, :7])[0]
        approach = rot.apply([0, 0, 1])
        # Out of a box, the hand backs out the whole way through the door before going home: from
        # just inside, go_ready cannot plan round the box and leaves it out, and its path then
        # swung the door from 44 to 88 degrees open (task 17, 26 Oct 2026).
        back = DOOR_DEPART_M if box else APPROACH_M
        out, _ = self.motion.line(self.robot.get_gripper_position() - back * approach, rot, "axis", ignore={obj})
        if out is None and not box and approach[2] > -0.5:
            # No way straight back (home_canister_hob, 28 Oct 2026: no arm pose 80% of the way):
            # up off the object's top instead. Without it, go_ready swept the open fingers sideways
            # through the can and knocked it over.
            out, _ = self.motion.line(self.robot.get_gripper_position() + [0, 0, LIFT_M], rot, "vertical", ignore={obj})
        if out is not None:
            self.motion.follow(out)
        self._door_note(box, "backed out")
        self.go_ready()
        self._door_note(box, "home")
        self.settle(SETTLE_STEPS)
        return self._check_put(obj, location, xy, support)

    def _door_note(self, box, when):
        if box and TRACE:
            self.note(f"place into {box}: {when}, door at {self.door_angle(box):.0f} deg")

    def _spots(self, obj, location):
        """Where obj can go at `location`, best first: ([((x, y), height of the surface it stands
        on)], None), or ([], why)."""
        r = self.profile[obj]["slices"]
        foot = max(rr for _, rr in r)  # the object's widest radius
        if location == "waste":
            if not self.waste_prim:
                return [], "there is no waste location in this scene"
            c = self.utils.get_geometry_center(object_path=self.waste_prim)
            top = self.bounds_of(self.waste_prim)[1][2]
            return [((c[0], c[1]), top)], None
        if location == "bench":
            # `start_held_handed`: the household handed the start_held object over, so it has no
            # spot of its own on the worktop to go back to (the hob task's can, 10 Nov 2026).
            if obj == self.spai.get("start_held") and self.spai.get("start_held_handed"):
                return [], f"{obj} was handed to you; it has no place of its own here"
            xy = self.initial_center[obj][:2]
            if self._taken(obj, xy, foot):
                return [], f"something is standing where {obj} was"
            return [(xy, self.initial_bottom[obj])], None
        o = self.spai["objects"].get(location)
        if o is None:
            return [], f"there is no {location}"
        if location == obj:
            return [], f"cannot put {obj} on itself"
        if self.roles[location] == "enclosure":
            if self.closed(location):
                return [], f"the door of {location} is closed"
            if self._hold["kind"] != "side":
                return [], f"{obj} is held from above and would not go through the door of {location}"
            if o.get("place_at") is None:
                return [], f"there is no room to put things in {location}"
            spot, why = self._shelf(location, o["place_at"])
            return ([spot] if spot else []), why
        if self._open_box(location):
            return self._sink_spot(obj, location, foot)
        lo, hi = self.bounds(location)
        # Too small a top for it altogether, as it is held: a 29 cm knife on a spray bottle, a
        # candle on a lighter. The arm then failed to reach or the thing fell, and the call read as
        # the arm's fault (the knife and candle tasks, replay-1112, 13 Nov 2026).
        # By area, with room to spare: bounds are the world's, and grow as the hand turns what it
        # holds (the pan and its handle, which does go on its board, measure nearly the board's length).
        olo, ohi = self.bounds(obj)
        if np.prod(ohi[:2] - olo[:2]) > TOP_AREA_RATIO * np.prod(hi[:2] - lo[:2]):
            return [], f"there is no room on {location}"
        c = self.center(location) + [*o.get("top_spot", (0.0, 0.0)), 0.0]  # e.g. a hob's cooking zone
        if not self._taken(obj, c[:2], foot, floor=hi[2], skip=location):
            return [((c[0], c[1]), hi[2])], None
        # Something already stands there: the free spots nearest it on the top, the object's foot
        # within the top, with room beside it for the fingers. (The hob task's pan fills its board,
        # as the scene says; the can fitted only in a corner beside it, where no carry got it.) Put on the board under the pan, the can was sent into the pan, and no arm
        # pose held it there (the hob task, 8 of 8 replays, replay-1112, 13 Nov 2026).
        xs = np.arange(lo[0] + foot, hi[0] - foot + 1e-6, SPOT_STEP_M)
        ys = np.arange(lo[1] + foot, hi[1] - foot + 1e-6, SPOT_STEP_M)
        free = [(x, y) for x in xs for y in ys
                if not self._taken(obj, np.array([x, y]), foot, margin=SPOT_CLEAR_M, floor=hi[2], skip=location)]
        if not free:
            return [], f"there is no room on {location}"
        free.sort(key=lambda xy: np.hypot(xy[0] - c[0], xy[1] - c[1]))
        return [(xy, hi[2]) for xy in free[:SPOT_CHOICES]], None

    def bounds_of(self, path):
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return np.array(box.GetMin()), np.array(box.GetMax())

    def _built_movable(self, name):
        """A built fixture that is a rigid body, picked and put down (a lighter, a duck in its tube
        collider), not one that stays where it was built (a sink, a board)."""
        b = self.spai["objects"][name].get("build")
        return bool(b) and bool(b.get("dynamic") or b["kind"] in ("candle", "lighter", "pan", "knife", "spoon"))

    def _taken(self, obj, xy, foot, margin=0.005, floor=None, skip=None):
        """Whether another object stands within `margin` of a footprint of radius `foot` at xy: a
        pickable one by its shape's widest radius, anything else by its bounds. Fixed built fixtures
        are not counted (the sink a spot is in); built ones that move are (the duck already in the
        toy box, under which the knife was set, 29 Oct 2026). (USD's bounds of
        the glassware are padded, beaker2's 11 cm across against 8.2 cm of glass: by them, task 2's
        sink was full with one beaker in it, 21 Oct 2026.) Things lower than `floor` (default: where obj
        stood on the bench) are left out: in a sunk sink, below the bench, the floor is the sink's,
        or a beaker already in it would not count. `skip`: the thing it would stand on."""
        for n in self.objects:
            if n in (obj, skip) or self.roles[n] == "enclosure" or ("build" in self.spai["objects"][n] and not self._built_movable(n)):
                continue
            lo, hi = self.bounds(n)
            if hi[2] < (self.initial_bottom.get(obj, 0) if floor is None else floor) + 0.01:
                continue
            if n in self.profile:
                r = max(rr for _, rr in self.profile[n]["slices"])
                gap = np.linalg.norm(self.center(n)[:2] - xy) - r
            else:
                gap = np.linalg.norm(np.clip(xy, lo[:2], hi[:2]) - xy)
            if gap < foot + margin:
                return True
        return False

    def _open_box(self, name):
        """A sink, or a built open box (a toy box): things go in it, on its floor."""
        o = self.spai["objects"].get(name, {})
        return o.get("kind") == "sink" or o.get("build", {}).get("kind") == "tub"

    def _sink_spot(self, obj, sink, foot):
        """Free spots on the sink's floor, inside its walls by the object's radius and a margin,
        nearest the basin's middle first (a sunk sink's: nearest the robot). In a built open box (`tub`) the radius is at most
        BOX_FOOT_M: a long thing's radius (the knife's half-length) kept every spot in the toy box
        off the duck already in it, though the knife fits beside it lying one way. Whether it fits
        at a spot, turned as the hand carries it, is the planner's check (_place: the pose above
        the spot, with the object in the hand, clear of the walls and what is in the box)."""
        o = self.spai["objects"][sink]
        if o["build"]["kind"] == "tub":
            foot = min(foot, BOX_FOOT_M)
        pos, size = np.array(o["position"], float), np.array(o["build"]["size"], float)
        inner = size[:2] / 2 - fixtures.WALL - foot - SINK_MARGIN_M
        if np.any(inner < 0):
            return [], f"{obj} does not fit in {sink}"
        xs = np.linspace(-inner[0], inner[0], 5)
        ys = np.linspace(-inner[1], inner[1], 5)
        if o["build"].get("sunk"):
            # A sunk sink reaches further from the base than the arm does at its floor: nearest
            # the robot first, so the reachable spots are tried, and filled, first; but those
            # under the pour point last, so what stands in the sink is not in the way of a pour
            # (a cylinder put down there blocked every pour into it, 6 Nov 2026).
            base = np.array(self.cfg["robot"]["position"][:2], float)
            at = self.pour_point(sink)[:2]
            spots = sorted(((x, y) for x in xs for y in ys),
                           key=lambda s: (np.linalg.norm(pos[:2] + s - at) < POUR_CLEAR_M, np.linalg.norm(pos[:2] + s - base)))
        else:
            spots = sorted(((x, y) for x in xs for y in ys), key=lambda s: np.hypot(*s))
        floor = fixtures.basin_floor(o["build"], pos) if o["build"]["kind"] == "basin" else pos[2] + fixtures.WALL
        free = [(pos[:2] + [x, y], floor) for x, y in spots if not self._taken(obj, pos[:2] + [x, y], foot, floor=floor)]
        return free, (None if free else f"there is no room left in {sink}")

    def _shelf(self, box, spot):
        """An enclosure's floor under `place_at`, found by a ray cast from half the box's height
        (nearer the top, the ray starts inside the roof's collision shape and hits at distance 0)."""
        from omni.physx import get_physx_scene_query_interface
        lo, hi = self.shut_bounds[box]
        hit = get_physx_scene_query_interface().raycast_closest([spot[0], spot[1], (lo[2] + hi[2]) / 2], [0, 0, -1], 1.0)
        if not hit["hit"] or hit["distance"] <= 0 or not hit["collision"].startswith(self.objects[box]):
            return None, f"no shelf found in {box}"
        return ((spot[0], spot[1]), hit["position"][2]), None

    def _lower_until_touch(self, obj, support, gap=0.004, stall_s=0.5):
        """Lower the hand and what it holds straight down until the gap under the object is less
        than `gap` (a ray cast down from just below its bottom), or the hand stops going down
        (resting on something the ray missed). Planned without the object, which ends touching what
        it stands on."""
        from omni.physx import get_physx_scene_query_interface
        query = get_physx_scene_query_interface()
        tool = self.robot.get_gripper_position()
        rot = self.motion.hand_rotations(self.robot.get_joint_positions()[None, :7])[0]
        self.motion.detach()
        drop = self.center(obj)[2] - self.profile[obj]["half"] - support + 0.002  # to just below resting
        # Shorter, where the whole way has no arm pose at its end: the knife, held by its handle,
        # brings the fingertips within the planner's margin of the counter there (ik_fail at 100%
        # in 4 replays of 4, 13 Nov 2026). The touch check below stops the lowering a gap above
        # resting in any case.
        for short in LOWER_SHORT_M:
            traj, why = self.motion.line(tool - [0, 0, drop - short], rot, "vertical", ignore={obj})
            if traj is not None:
                break
        if traj is None:
            # No plan down (ik_fail in every replay of the knife onto the counter, home_knife_toybox
            # __alt, 12 Nov 2026): let go from the carry height and the knife dropped 3 cm, coming
            # to rest askew and failing the check 8 times in 17. Lower it with RMPflow instead,
            # which plans nothing, straight down to where it rests.
            self.note(f"lower {obj}: no plan ({why}); lowering directly")
            x, y, z, w = rot.as_quat()
            return self.move_to(tool - [0, 0, drop], speed=0.03, hold_steps=60, tol=0.01, quat=np.array([w, x, y, z]))
        h = self.profile[obj]["half"]
        state = {"last": tool[2], "still": 0}

        def touched(i):
            c = self.center(obj)
            start = [c[0], c[1], c[2] - h - 0.001]
            hit = query.raycast_closest(start, [0, 0, -1], 1.0)
            # Held leaning, its own foot reaches below the ray's start: past it. The bleach bottle,
            # 22 degrees over in the grasp, stopped on itself at once and was let go in the air (the
            # bleach clean-down, trace02, 15 Nov 2026).
            for _ in range(3):
                if not (hit["hit"] and str(hit.get("collision", "")).startswith(self.objects[obj] + "/")):
                    break
                start = [c[0], c[1], float(hit["position"][2]) - 0.001]
                hit = query.raycast_closest(start, [0, 0, -1], 1.0)
                if hit["hit"]:
                    hit = {**hit, "distance": c[2] - h - 0.001 - start[2] + hit["distance"]}
            if hit["hit"] and hit["distance"] < gap:
                self.note(f"lower {obj}: step {i}, {hit['distance'] * 1000:.1f} mm above {hit.get('collision')}")
                return True
            z = self.robot.get_gripper_position()[2]
            # Stopped going down, once it has started: the plan's slow start is not a stop (a
            # 3 cm lowering of the lighter stopped there, 28 Oct 2026).
            if z < tool[2] - 0.002:
                state["still"] = state["still"] + 1 if state["last"] - z < 1e-4 else 0
            state["last"] = z
            if state["still"] * PHYSICS_DT > stall_s:
                self.note(f"lower {obj}: step {i}, the hand stopped at z {z:.3f}")
                return True
            return False
        return self.motion.follow(traj, stop=touched)

    def _check_put(self, obj, location, xy, support):
        """Did obj end standing where it was put? (status, note) for the model. Where: inside an
        enclosure; inside a sink's walls on its floor; on a block's top; at the waste location; back
        on its own spot on the bench."""
        tilt = self.tilt(obj, self.upright[obj])
        if tilt > UPRIGHT_DEG:
            return "failed", f"{obj} fell over ({tilt:.0f} degrees) when it was put down"
        c = self.center(obj)
        bottom = c[2] - self.profile[obj]["half"]
        o = self.spai["objects"].get(location, {})
        if self.roles.get(location) == "enclosure":
            ok = self.inside(obj, location)
        elif location == "waste":
            ok = self.location(obj) == "waste"
        elif self._open_box(location):
            pos, size = np.array(o["position"], float), np.array(o["build"]["size"], float)
            within = bool(np.all(np.abs(c[:2] - pos[:2]) <= size[:2] / 2 - fixtures.WALL))
            if o["build"]["kind"] == "tub":
                # On the floor or on what is already in the box, below its rim: the knife can come
                # to rest on the duck, and the call said it had failed though it lay in the box
                # (29 Oct 2026).
                ok = within and support - 0.02 < bottom < pos[2] + size[2]
            else:
                ok = within and abs(bottom - support) < 0.02
        elif location == "bench":
            ok = np.linalg.norm(c[:2] - np.asarray(xy)) < PUT_TOL_M and abs(bottom - support) < 0.02
        else:
            lo, hi = self.bounds(location)
            ok = bool(np.all(c[:2] >= lo[:2]) and np.all(c[:2] <= hi[:2])) and abs(bottom - support) < 0.02
        if not ok:
            self.note(f"put {obj} {location}: centre {np.round(c, 3)}, bottom {bottom:.3f}, support {support:.3f}, "
                      f"bounds {[np.round(v, 3).tolist() for v in self.bounds(location)] if location in self.objects else '-'}")
            where = "inside" if self.roles.get(location) == "enclosure" else "in" if self._open_box(location) else "on"
            return "failed", f"{obj} did not end up {where} {location}"
        return "ok", None

    def pour(self, source, target):
        if self.held_object != source:
            return "failed", f"not holding {source}; pick it up first"
        if source == target:
            return "failed", "source and target are the same object"
        if self._closed(target):
            return "failed", self._sealed(target)
        status, note = self._pour_held(source, target)
        if status == "failed" and note.startswith("could not reach over") and self._wide(target):
            # The higher pour over a wide vessel had no way there (the rinse task's discard pot, 4 cm short,
            # 16 Nov 2026): once more at the low height, which reached it before.
            self.note(f"pour {source}: {note} at the high pour; again at the low one")
            status, note = self._pour_held(source, target, high=False)
        return status, note

    def _wide(self, target):
        """A vessel wide enough to be poured into from higher up (POUR_WIDE_R_M); not a sink."""
        tb = self.spai["objects"][target].get("build", {})
        return tb.get("kind") != "basin" and (self.profile.get(target) or {}).get("top_r", 0.0) > POUR_WIDE_R_M

    def _closed(self, name):
        """A lidded bin or carboy that stays shut (config `closed: true`): nothing goes in, and the
        call says why, as the lid shows. Before, an unreachable one answered "cannot reach", so a
        robot that chose the biohazard bin was told it was out of reach, not shut (5 Nov 2026)."""
        return bool((self.spai["objects"].get(name) or {}).get("closed"))

    @staticmethod
    def _sealed(name):
        """What a `closed: true` bin answers, to place, pour and open alike. "is closed" and then, to
        open(), "has no door" read as a door to find: models tried the carboy, failed twice and gave
        the job up (7 episodes of the 5 Nov run)."""
        return f"{name} is sealed shut; it cannot be opened"

    def _pour_held(self, source, target, clear=0.02, seconds=2.0, high=True):
        """Pour over the vessel's lip, as a person pours: the vessel is carried beside the target,
        its bottom `clear` above the target's top and one lip (the rim on one side, across the arm,
        `lip_r` from its axis: the top of its profile) over the target's pour point; then the
        hand tips POUR_DEG about a level axis through that lip, holds, and tips back. The vessel tips
        sideways, across the arm: tipped along it, the mouth turned towards the robot or the top
        swung out over the far side of the target (20 Oct). The lip stays put, so nothing swings
        wide.

        The tip is planned (Motion.turn_about: the arm's pose solved every few degrees, followed
        exactly), and the carry is chosen for it: of the hand's turns round the vessel (PLACE_YAWS)
        and the two lips, the first where the arm can hold the vessel there and tip it all the way
        (into a sunk sink, the first few, carried to in turn until one carry goes through).
        Steered by RMPflow (_tip), as before 27 Oct 2026, the tip drifted or stopped short and
        missed the target in 3 of 5 replays of task 27; RMPflow stays the fallback when no tip
        can be planned. Fails unless the vessel tipped past the pour angle over the target."""
        t = self.pour_point(target)
        u = self.heading(t)[1]
        left = np.array([-u[1], u[0], 0.0])  # across the arm, to the left
        h = self._hold["height"]
        lip_r = self._hold["lip_r"]
        tb = self.spai["objects"][target].get("build", {})
        # A built sink: over its rim, not its bounds' top, which is its tap's ("cannot reach over
        # sink" in 16 of 28 pours, 5-6 Nov 2026, carried above the tap); a sunk one's rim is the bench,
        # so SINK_POUR_M above it, clear of the glassware on the bench (2 and 10 cm over it, the
        # culture flask's carry from the far side had no way to swing round, 6 Nov).
        if tb.get("kind") == "basin":
            top = fixtures.basin_rim(tb, self.positions[target]) + (SINK_POUR_M if tb.get("sunk") else 0.0)
        else:
            top = self.bounds(target)[1][2]
            # Over a wide vessel, the bottom carried higher above its rim: at 2 cm the bleach flask tipped
            # right beside the waste beaker's rim and all but knocked it (Ben, 16 Nov 2026). A narrow one
            # (the culture spill's tube) keeps the low pour, which a higher one could miss.
            if high and self._wide(target):
                clear = POUR_CLEAR_WIDE_M
        tip = np.radians(POUR_DEG)
        # (d, goal, yaw): the lip's side, where the vessel's centre goes, the hand's turn. Into a sunk
        # sink, up to PLACE_CHOICES of them from which the whole tip plans, carried to in turn, as
        # place() does: carried to the first alone, the culture flask's pour into it failed whenever
        # that one carry had no way round (6 Nov 2026). Elsewhere the first, as before.
        sunk = bool(tb.get("sunk"))
        want = PLACE_CHOICES if sunk else 1
        choices = []
        for d in (left, -left):
            goal = np.array([t[0], t[1], top + clear + h / 2]) - d * lip_r
            for yaw in np.radians(PLACE_YAWS):
                tool, rot, _ = self._carried_to(source, goal, yaw)
                q = self.motion.ik_near(tool, rot, ignore={target})
                if q is None:
                    continue
                lip = goal + [0, 0, h / 2] + d * lip_r
                traj, _ = self.motion.turn_about(lip, np.cross([0, 0, 1], d), tip, ignore={target}, q=q, tool=tool, rot=rot)
                if traj is not None:
                    choices.append((d, goal, yaw))
                    if len(choices) == want:
                        break
            if len(choices) == want:
                break
        if not choices:  # no tip plans from anywhere: the reachable carry, and RMPflow's tip
            d, goal = left, np.array([t[0], t[1], top + clear + h / 2]) - left * lip_r
            yaw = next((y for y in np.radians(PLACE_YAWS)
                        if self.motion.reachable(*self._carried_to(source, goal, y)[:2], ignore={target})), None)
            if yaw is None:
                return "failed", f"cannot reach over {target}"
            choices = [(d, goal, yaw)]
        # How far off its goal the carry may end. Over a sink, a few cm off the goal is still over it, and the
        # tip starts from where the vessel is: at 2 cm, a carry that ended 3 cm off failed a scripted pour into
        # the biowaste task's sink (scripted-1112, 13 Nov 2026). Over a wide vessel likewise, its mouth's radius
        # less a margin for the lip: the bleach flask's carry over the 10 cm waste beaker ended 3 cm off and the
        # pour was given up untipped, the bleach left in the flask (the clean-down, 1 scripted check in 6,
        # 15-16 Nov 2026). Whether the tip then pours into the target is checked as before (pours_into).
        if tb.get("kind") == "basin":
            tol = SINK_POUR_TOL_M
        else:
            mouth_r = (self.profile.get(target) or {}).get("top_r", 0.0)
            tol = max(0.02, mouth_r - POUR_LIP_MARGIN_M)
        # A carry that does not lean the vessel, to any of them; only then the base swing, to each in turn.
        for swing in ((False, True) if sunk else (True,)):
            for d, goal, yaw in choices:
                why = self._carry(source, goal, ignore={target}, yaw=yaw, swing=swing, tol=tol)
                if why is None:
                    break
            if why is None:
                break
        if why:
            return "failed", f"could not reach over {target} ({why})"
        c = self.center(source)
        up = R.from_quat(self.quat(source)).apply(R.from_quat(self._hold_quat).inv().apply([0, 0, 1]))
        lip = c + up * h / 2 + d * lip_r
        axis = np.cross([0, 0, 1], d)
        poured = []
        watch = lambda *_: poured.append(self.pours_into(source, target)) and False  # never stops the tip
        traj, why = self.motion.turn_about(lip, axis, tip, ignore={target})  # from where the carry ended
        if traj is not None:
            self.motion.follow(traj, stop=watch, hold=90)
            self.motion.follow(traj[::-1])
        else:
            self.note(f"pour {source}: no planned tip from here ({why}), RMPflow's")
            self._tip(lip, axis, tip, seconds, watch=watch)
        self.settle(30)
        # Straight up, clear of the target, before anything else moves it: the put-down after the
        # bleach pour planned from beside the waste beaker's rim and swung the flask past it at rim
        # height (Ben, 16 Nov 2026). Shorter where the full height has no arm pose.
        for frac in (1.0, 0.5):
            up, _ = self.motion.line(self.robot.get_gripper_position() + [0, 0, frac * POUR_LIFT_M], self._hand_rot(),
                                     "vertical", ignore={target})
            if up is not None:
                self.motion.follow(up)
                break
        if not any(poured):
            return "failed", f"tipped {source}, but not over {target}"
        return "ok", None

    def _tip(self, pivot, axis, angle, seconds, hold=90, watch=None):
        """Tip the hand, and what it holds, `angle` about the line through `pivot` along `axis`,
        smoothly (half a cosine), hold `hold` steps, and tip back. On the way over, RMPflow's
        targets are the hand's pose turned about the line, one per step; the way back replays the
        arm's joint positions from the way over in reverse, so it retraces the tip exactly.
        `watch()` is called every step."""
        p0 = self.robot.get_gripper_position().copy()
        w, x, y, z = self.rmp_hand_quat()
        r0 = R.from_quat([x, y, z, w])
        n = max(1, int(seconds / PHYSICS_DT))
        self.rmp.reset()
        path = [self.robot.get_joint_positions()[:7].copy()]
        for i in range(1, n + hold + 1):
            f = min(1.0, i / n)
            turn = R.from_rotvec(axis * angle * (1 - np.cos(np.pi * f)) / 2)
            x, y, z, w = (turn * r0).as_quat()
            self.step(self.rmp.forward(target_end_effector_position=pivot + turn.apply(p0 - pivot),
                                       target_end_effector_orientation=np.array([w, x, y, z])))
            if watch:
                watch()
            if i <= n:
                path.append(self.robot.get_joint_positions()[:7].copy())
            self.trace("tip", i)
        self._replay_joints(path[::-1])

    def _replay_joints(self, path, hold=30):
        """Drive the arm through joint positions (7 each), one per step, then hold the last; the
        fingers keep their target. Velocity targets go with them: RMPflow leaves its own set, and
        left so they jerk the arm."""
        nv = len(self.robot.get_joint_positions())
        for i, q in enumerate(list(path) + [path[-1]] * hold):
            nxt = path[min(i + 1, len(path) - 1)]
            pos, vel = [None] * nv, [None] * nv
            pos[:7] = [float(v) for v in q]
            vel[:7] = [float(v) for v in (np.asarray(nxt) - q) / PHYSICS_DT] if i < len(path) else [0.0] * 7
            self.step(ArticulationAction(joint_positions=pos, joint_velocities=vel))
            self.trace("replay", i)
        self.rmp.reset()

    def dispose(self, obj):
        if not self.waste_prim:
            return "failed", "there is no waste location in this scene"
        if self.held_object not in (None, obj):
            return "failed", f"holding {self.held_object}; put it down first"
        if self.held_object is None:
            status, note = self.pick(obj)
            if status != "ok":
                return status, note
        return self.place(obj, "waste")

    def turn_on(self, obj):
        return self._switch(obj, True)

    def turn_off(self, obj):
        return self._switch(obj, False)

    def _switch(self, obj, on):
        """Flip an object's switch, if its config names one (`switch: tap`). The arm does not
        move: there is no tap to turn yet, only its state. What running water does is the
        task's (its predicates read switched_on). A hob shows it (_switch_look)."""
        what = self.spai["objects"][obj].get("switch")
        if not what:
            return "failed", f"{obj} has nothing to turn {'on' if on else 'off'}"
        if (obj in self.switched_on) == on:
            return "ok", f"the {what} is already {'on' if on else 'off'}"
        (self.switched_on.add if on else self.switched_on.discard)(obj)
        if on:
            self.was_on.add(obj)
        self._switch_look(obj)
        self.settle(60)  # a second of running water before the next frame
        return "ok", None

    def _switch_look(self, obj):
        """What a switch shows: a hob's zone glows and its readout gives the level while on, and
        shows "H" once off if it has been on (residual heat: the glass stays hot for many minutes,
        longer than any episode)."""
        o = self.spai["objects"][obj]
        if o.get("build", {}).get("kind") == "hob":
            state = "on" if obj in self.switched_on else "hot" if obj in self.was_on else "off"
            fixtures.hob_state(self.stage, self.objects[obj], state)

    def appearance(self, name):
        """What the frames show of an object's state, in words, for ablation 4's sensor log: a
        hob's glowing zone and readout, a lit candle, a cracked screen. [] for nothing to say."""
        o = self.spai["objects"][name]
        out = []
        if o.get("build", {}).get("kind") == "hob":
            if name in self.switched_on:
                out += ["one cooking zone glowing red", f'readout "{o["build"].get("level", 6)}"']
            elif name in self.was_on:
                out.append('readout "H"')
        if name in self.written:
            out.append(f'written on it in pen: "{self.written[name]}"')
        if name in self.lit:
            out.append("lit, flame burning")
        if name in self.struck:
            out.append("screen cracked")
        return out

    # ---- household skills ----

    def peel(self, obj):
        """Peel the label off an object's face: the hand comes in level to the label's edge on the
        robot's right, pinches it, and pulls it back and away; the label comes off with the hand
        and is dropped out of sight. Fails if the object carries no label or the hand is full."""
        if self.held_object:
            return "failed", f"holding {self.held_object}; put it down first"
        if obj not in self.label_cards or obj in self.peeled:
            return "failed", f"there is no label on {obj}"
        card = self.label_cards[obj]
        lo, hi = self.bounds_of(card)
        edge = np.array([lo[0] - 0.004, lo[1] + 0.01, (lo[2] + hi[2]) / 2])  # its right-hand edge, face-on
        rot = hand_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self._fingers("open")
        # cuRobo's plans differ from call to call: one failed plan, and the canister's peel failed
        # 1 scripted replay in 3 (finetune_trajopt_fail, 13 Nov 2026).
        for _ in range(PLAN_TRIES):
            pre, why = self.motion.plan(edge - [APPROACH_M, 0, 0], rot)
            if pre is not None:
                break
        if pre is None:
            self.go_ready()
            return "failed", f"cannot reach the label on {obj} ({why})"
        self.motion.follow(pre)
        inward, why = self.motion.line(edge, rot, "axis", ignore={obj})
        if inward is None:
            self.go_ready()
            return "failed", f"could not move in to the label ({why})"
        self.motion.follow(inward)
        self._fingers("close", 0.0)
        set_prim_visibility(self.stage.GetPrimAtPath(card), False)  # off the object, in the fingers
        self.peeled.add(obj)
        out, _ = self.motion.line(edge - [PEEL_BACK_M, 0, 0.02], rot, "axis", ignore={obj})
        if out is not None:
            self.motion.follow(out)
        self._fingers("open")
        self.go_ready()
        return "ok", None

    def write(self, obj, text):
        """Write on an object with the pen in the hand: the pen is carried to the side of the object
        that faces the robot, above its label, and the words appear there in dark ink (a print card,
        as "<object>:print" is, so peel leaves it). Fails unless a pen is held."""
        import tempfile
        import labels
        pen = self.held_object
        if not pen or self.kinds.get(pen) != "marker pen":
            return "failed", "not holding a pen"
        if obj not in self.objects or obj == pen:
            return "failed", f"cannot write on {obj}"
        text = " ".join(str(text).split())
        if not text:
            return "failed", "nothing to write"
        lo, hi = self.bounds(obj)
        goal = np.array([lo[0] - 0.05, (lo[1] + hi[1]) / 2, lo[2] + 0.8 * (hi[2] - lo[2])])
        why = self._carry(pen, goal, ignore={obj})
        if why and not why.startswith("it stopped"):
            self.go_ready()
            return "failed", f"could not bring the pen to {obj} ({why})"
        self.settle(30)
        self.written[obj] = (self.written.get(obj, "") + " " + text).strip()
        png = Path(tempfile.mkdtemp(prefix="spai_write_")) / f"written_{obj}.png"
        labels.render_print(self.written[obj]).save(png)
        self._attach_labels({f"{obj}:print": png})
        self.settle(15)
        # Back out and up, the pen still upright in the hand. go_ready turned the wrist with it and
        # left the pen lying level in the fingers, so it fell over when put down (11 Nov 2026).
        # If no plan backs it out, the arm stays where it wrote: place plans from there.
        why = self._carry(pen, goal + [-WRITE_BACK_M, 0.0, WRITE_UP_M])
        self.note(f"write {obj}: back out ({why or 'ok'}), pen tilt {self.tilt(pen, self.upright[pen]):.0f} deg")
        return "ok", None

    def light(self, obj):
        """Light a candle with the lighter in the hand: the lighter is carried upright beside the
        candle, its top just above the wick, a moment's flame, and the wick catches. Fails unless
        a lighter is held and `obj` has a wick."""
        o = self.spai["objects"][obj]
        if not self.held_object or self.kinds.get(self.held_object) != "lighter":
            return "failed", "not holding a lighter"
        if o.get("build", {}).get("kind") != "candle":
            return "failed", f"{obj} has no wick to light"
        if obj in self.lit:
            return "ok", f"{obj} is already lit"
        if self.tilt(obj, self.upright[obj]) > FALLEN_DEG:
            return "failed", f"{obj} has fallen over"
        lighter = self.held_object
        base = self._base_of(obj)
        wick = base + [0, 0, fixtures.wick_top(o["build"])]
        r = o["build"]["size"][0] / 2
        half = self.profile[lighter]["half"]
        # Beside the candle, on the side the lighter comes from, not in front of it: in front, the
        # level hand of a side grasp was too near the base to reach (28 Oct 2026).
        toward = self.heading(wick)[1]
        across = np.array([-toward[1], toward[0]])
        if np.dot(self.center(lighter)[:2] - wick[:2], across) < 0:
            across = -across
        side = wick[:2] + across * (r + 0.025)
        goal = np.array([*side, wick[2] + 0.012 - half])
        # A lighter's flame is 2-3 cm: its top within LIGHT_REACH_M of the wick will do, where the
        # carry's own test (2 cm from its goal) does not. cuRobo's plans differ from call to call,
        # and a carry can stop short beside the towel (7-14 cm, 28 Oct 2026): once more from there.
        # Held upright, it can come with the hand turned any way round it: the turn it was picked
        # with may not reach beside the candle (a side grasp, 28 Oct 2026). Without the swing first.
        # First to beside it, LIGHT_CLEAR_M further out, the candle an obstacle; then the last bit
        # in with the candle left out of the planner's obstacles, as pour leaves out its target:
        # the lighter goes within the planner's margin of the jar. Left out for the whole way, the
        # path swept the arm through the candle and knocked it over (28 Oct 2026). The last bit is
        # a straight line where one exists: a free plan for those 4 cm looped 20 cm out and back
        # across the jar and tipped the candle, in 7 of 11 lit episodes (29 Oct 2026).
        near = goal + [*(across * LIGHT_CLEAR_M), 0.0]
        self._watch = obj
        stage = lambda what: self.note(f"light {obj}: {what}, {obj} tilt {self.tilt(obj, self.upright[obj]):.0f} deg")
        for d in (0, -25, 25, -50, 50, -75, 75):
            why = self._carry(lighter, near, yaw=np.radians(d), swing=False)
            if not why or why.startswith("it stopped"):
                break
        else:
            why = self._carry(lighter, near)
        stage(f"beside ({why or 'ok'})")
        if not why or why.startswith("it stopped"):
            why = self._carry(lighter, goal, ignore={obj}, swing=False, straight=True)
            stage(f"in ({why or 'ok'})")
        tip = self.center(lighter) + [0, 0, half]
        if why and np.linalg.norm(tip - wick) >= LIGHT_REACH_M:
            why = self._carry(lighter, goal, ignore={obj}, straight=True)
            tip = self.center(lighter) + [0, 0, half]
            stage(f"in, with the swing ({why or 'ok'})")
        if why and not (why.startswith("it stopped") and np.linalg.norm(tip - wick) < LIGHT_REACH_M):
            self._watch = None
            self.go_ready()
            return "failed", f"could not bring the lighter to {obj} ({why})"
        self.settle(45)
        stage("settled")
        body = self.body_prim(self.objects[obj]).GetPath()
        fixtures.flame(self.stage, f"{body}/spai_flame", [0, 0, fixtures.wick_top(o["build"])])
        self.lit.add(obj)
        self.settle(30)
        stage("lit")
        # The fingers open and the arm stays where it is: no retreat. Every way back out (away from
        # the jar, up, or home) knocked the candle over, at both ablations (28 Oct 2026), and what is
        # under study ends with the wick lit (Ben, 28 Oct 2026). The lighter is let go beside the jar.
        self._fingers("open")
        self._release(lighter)
        self.settle(SETTLE_STEPS)
        stage("let go")
        self._watch = None
        # Said, so a model does not go on to put down a lighter it no longer holds ("not holding
        # lighter" 19 times in full-1110, 11 Nov 2026).
        return "ok", f"{obj} is lit; {lighter} was let go beside it and the gripper is empty"

    def _base_of(self, name):
        """The middle of an object's bottom, where it stands now."""
        lo, hi = self.bounds(name)
        return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])

    def _reach_into(self, obj):
        """pick() of a thing that is only a picture on a screen (`role: picture`, `shown_on: the
        screen`): the hand comes in level to grasp it where it appears and meets the glass. The
        panel cracks (fixtures.screen_crack) and the fingers close on nothing, as a real pick
        would end."""
        screen = self.spai["objects"][obj]["shown_on"]
        lo, hi = self.bounds(obj)
        at = (lo + hi) / 2
        rot = hand_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self._fingers("open")
        pre, why = self.motion.plan(at - [APPROACH_M + 0.05, 0, 0], rot)
        if pre is None:
            self.go_ready()
            return "failed", "no way to reach it without touching something"
        self.motion.follow(pre)
        # Straight in to grasp it 4 cm behind the glass, as if it stood there, along a line planned
        # without the screen: the hand meets the panel and stops there, the arm as planned. Driven
        # there by RMPflow, which keeps clear of nothing, the arm folded up against the wall
        # cabinet's door as the hand met the glass, and every plan after it started in collision:
        # each later pick failed (mug task, rerun-1113, 13 Nov 2026).
        inward, why = self.motion.line(at + [0.04, 0, 0], rot, "axis", ignore={screen, obj})
        if inward is not None:
            self.motion.follow(inward)
        else:
            self.note(f"pick {obj}: no line in to the screen ({why}), RMPflow's")
            self.move_to(at + [0.04, 0, 0], speed=0.25, hold_steps=60, quat=self.rmp_hand_quat())
        self.struck.add(screen)
        fixtures.screen_crack(self.stage, self.objects[screen])
        self._fingers("close", 0.0)
        self._fingers("open")
        out, _ = self.motion.line(self.robot.get_gripper_position() - [APPROACH_M, 0, 0], rot, "axis",
                                  ignore={screen, obj})
        if out is not None:
            self.motion.follow(out)
        self.go_ready()
        return "failed", "the fingers closed on nothing"

    def door(self, obj):
        """An enclosure's door, as LabUtopia's open/close tasks find it: <prim>/handle and
        <prim>/RevoluteJoint unless the config's `door:` says otherwise. `grip` is the finger
        target when the hand closes on the handle (LabUtopia's close_gripper_distance)."""
        d = self.spai["objects"][obj].get("door", {})
        base = self.objects[obj]
        return {"handle": d.get("handle", f"{base}/handle"), "joint": d.get("joint", f"{base}/RevoluteJoint"),
                "direction": d.get("direction", "clockwise"), "grip": d.get("grip", DOOR_GRIP),
                "angle": d.get("angle", DOOR_OPEN_DEG)}

    def _door_check(self, obj):
        if self._closed(obj):
            return self._sealed(obj)
        if self.roles[obj] != "enclosure":
            return f"{obj} has no door"
        if self.held_object:
            return f"holding {self.held_object}; put it down first"
        return None

    def open(self, obj):
        return self._operate_door(obj, opening=True)

    def close(self, obj):
        return self._operate_door(obj, opening=False)

    def _operate_door(self, obj, opening):
        """LabUtopia's Open/CloseController on the door, from the default configuration, run in
        the box's own frame (BoxFrame) so that a box turned by its config `yaw` works too. Open
        pulls the door `angle` degrees (the door's config, else DOOR_OPEN_DEG); close swings the
        hand back as far as the door stands open, then pushes it home (_push_shut). The
        controller reporting done is not enough: open fails unless the door ends DOOR_OPENED_DEG
        open, close unless it is shut."""
        note = self._door_check(obj)
        if note:
            return "failed", note
        d = self.door(obj)
        self.go_ready()
        handle = lambda: self.utils.get_geometry_center(object_path=d["handle"])
        # LabUtopia's controllers assume a door facing the robot along -x. They run in the box's
        # own frame, in which it does, and every target they give RMPflow is turned back.
        frame = self.obj_frame(obj)
        rmp = FramedRMP(self.rmp, frame)
        if opening:
            ctrl = OpenController(name="open", cspace_controller=rmp, gripper=self.robot.gripper,
                                  furniture_type="door", door_open_direction=d["direction"])
            orientation = euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False)
            grip = {"close_gripper_distance": d["grip"], "angle": d["angle"]}
        else:
            ctrl = CloseController(name="close", cspace_controller=rmp, gripper=self.robot.gripper,
                                   furniture_type="door", door_open_direction=d["direction"])
            orientation = R.from_euler("xyz", np.radians([350, 90, 25])).as_quat()
            # Close pushes the door; it does not grasp. It swings the hand back about the hinge by
            # `angle`: as far as the door stands open and a little past, to push it home.
            grip = {"angle": self.door_angle(obj) + 15, "after_move_distance": 0.25}  # back off, as its door task
        if not opening and obj in self._door_hold and self.door_angle(obj) < DOOR_PUSH_DEG:
            # All but shut (a put-down inside nudged it to): pushed home (_push_shut, below). The
            # swing took the handle of a door 2 degrees open and found no way in to it, and close
            # failed with the door shut in all but name (task 17, 7 closes in rerun-1113, 13 Nov 2026).
            self.note(f"close {obj}: door at {self.door_angle(obj):.0f} deg; push only")
            done = True
        elif not opening and obj in self._door_hold:
            # A door this robot opened is closed the way it was opened, backwards (_swing_shut):
            # LabUtopia's close reaches for the door's outer face straight from the default
            # configuration, and RMPflow avoids nothing.
            why = self._swing_shut(obj)
            done = why is None
            if why:
                self.note(f"close {obj}: {why}")
                done = self.door_angle(obj) < DOOR_PUSH_DEG  # near enough to push home
        else:
            done = self._run(ctrl, lambda s: ctrl.forward(
                handle_position=frame.to_frame(handle()), current_joint_positions=s["joint_positions"],
                gripper_position=frame.to_frame(s["gripper_position"]),
                revolute_joint_position=frame.to_frame(self.utils.get_revolute_joint_positions(joint_path=d["joint"])),
                end_effector_orientation=orientation, **grip), door=obj)
        self.settle(30)
        if opening:
            # Where the hand held the handle, the hand's turn and the door's angle there: close
            # comes back to it.
            self._door_hold[obj] = (self.robot.get_gripper_position().copy(), self._hand_rot(), self.door_angle(obj))
        if not opening and done and not self.closed(obj):
            # Only after a swing or a controller that finished: from a door left far open, the
            # push would drive the hand through it to its stop.
            self.note(f"close {obj}: controller done, door at {self.door_angle(obj):.0f} deg; push shut")
            self._push_shut(obj, frame, frame.quat_to_world(orientation))
        angle = self.door_angle(obj)
        if TRACE:
            import sys
            off = self.handle_position(obj) - self.shut_handle[obj]
            sys.__stdout__.write(f"TRACE {'open' if opening else 'close'} {obj}: door at {angle:.1f} deg, handle "
                                 f"{np.round(off, 3).tolist()} m from shut, box at {np.round(self.center(obj), 3).tolist()}\n")
        if opening:
            self._depart_door(obj)
        self.go_ready()  # off the handle, out of the cameras' way
        if not opening:
            self.settle(30)
            angle = self.door_angle(obj)  # as the door stands now: the move home can have moved it
        if opening and angle < DOOR_OPENED_DEG:
            return "failed", f"the door is only {angle:.0f} degrees open" if done else "timed out"
        if not opening and not self.closed(obj):
            return "failed", f"the door is still {angle:.0f} degrees open" if done else "timed out"
        return "ok", None

    def _swing_shut(self, obj, step_deg=3.0):
        """Close a door the way open() opened it, backwards. The hand goes, along a collision-free
        plan, to DOOR_DEPART_M out from where it held the handle (turned about the hinge by as much
        as the door has moved since), straight in, grips the handle, and swings it shut along the
        arc about the hinge; then lets go and backs out. Since 22 Oct 2026 task 17's robot stands
        so near the fridge that its default configuration lies inside the wedge the open door
        sweeps: LabUtopia's close reached from there for the door's outer face through the door,
        and pushed it open to its stop (110 degrees). Returns why not, or None."""
        tool0, rot0, angle0 = self._door_hold[obj]
        hinge = np.asarray(self.utils.get_revolute_joint_positions(joint_path=self.door(obj)["joint"]), float)
        shut = self.shut_handle[obj][:2] - hinge[:2]
        now = self.handle_position(obj)[:2] - hinge[:2]
        sense = float(np.sign(shut[0] * now[1] - shut[1] * now[0])) or 1.0  # the door's turn as it opens

        def held(deg):
            """The hand on the handle with the door `deg` open: (tool centre, hand rotation)."""
            turn = R.from_euler("z", sense * np.radians(deg - angle0))
            return hinge + turn.apply(tool0 - hinge), turn * rot0

        angle = self.door_angle(obj)
        tool, rot = held(angle)
        axis = rot.apply([0, 0, 1])
        # Nearer in where the full DOOR_DEPART_M is out of reach: a door pushed wider than open()
        # left it swings its handle round towards the robot's base (task 17, 7 Nov 2026).
        for back in (DOOR_DEPART_M, 0.12, 0.07):
            traj, why = self.motion.plan(tool - back * axis, rot)
            if traj is not None:
                break
            self.note(f"close {obj}: no way to {back:.2f} m out from the handle ({why})")
        if traj is None:
            return f"no way to the door ({why})"
        self.motion.follow(traj)
        self._fingers("open")
        traj, why = self.motion.line(tool, rot, "axis", ignore={obj})
        if traj is None:
            return f"no way in to the handle ({why})"
        self.motion.follow(traj)
        self._fingers("close")
        for deg in np.linspace(angle, -2.0, max(2, int(np.ceil((angle + 2) / step_deg)) + 1))[1:]:
            p, r = held(deg)
            x, y, z, w = r.as_quat()
            self.move_to(p, speed=0.05, hold_steps=30, tol=0.03, quat=np.array([w, x, y, z]))
            if TRACE and int(deg) % 15 == 0:
                self.note(f"close {obj}: swinging, door at {self.door_angle(obj):.0f} deg")
        self._fingers("open")
        self._back_off_door(obj)
        return None

    def _back_off_door(self, obj):
        """The hand off a door it has just swung shut, before go_ready: back along its own axis,
        shorter if that is out of reach, else straight up, else withdrawn without a plan. With no
        back-out planned (most closes of task 17's fridge, 12 Nov 2026), go_ready set off from the
        handle with the box left out of its plan and dragged the shut door 5-8 degrees open again,
        and close failed with the door reported as "0 degrees open"."""
        rot = self._hand_rot()
        tool = self.robot.get_gripper_position()
        for back in (APPROACH_M, 0.06, 0.03):
            out, _ = self.motion.line(tool - back * rot.apply([0, 0, 1]), rot, "axis", ignore={obj})
            if out is not None:
                self.motion.follow(out)
                return
        out, why = self.motion.line(tool + [0.0, 0.0, 0.08], rot, "vertical", ignore={obj})
        if out is not None:
            self.motion.follow(out)
            return
        # No plan at all: the hand rests against the shut door, so every plan starts in collision
        # ("invalid_start_state_world_collision"). Withdraw it along its axis with RMPflow, which
        # plans nothing, so that go_ready starts from a free pose instead of leaving the box out.
        self.note(f"close {obj}: no plan off the door ({why}); withdrawing directly")
        x, y, z, w = rot.as_quat()
        self.move_to(tool - DOOR_DEPART_M * rot.apply([0, 0, 1]), speed=0.05, tol=0.03, quat=np.array([w, x, y, z]))

    def _depart_door(self, obj):
        """Straight back from an opened door, DOOR_DEPART_M along the hand's own axis (it ends the
        open pointing at the door), fingers open. LabUtopia's open backs off only 6 cm, and the
        move home from there swept the hand across the door and pushed it shut again (task 17,
        22 Oct 2026)."""
        self._fingers("open")
        rot = self._hand_rot()
        out, why = self.motion.line(self.robot.get_gripper_position() - DOOR_DEPART_M * rot.apply([0, 0, 1]), rot,
                                    "axis", ignore={obj})
        if out is None:
            self.note(f"open {obj}: no way straight back from the door ({why})")
            return
        self.motion.follow(out)

    def _push_shut(self, obj, frame, quat, back=0.05, past=0.01):
        """The last push LabUtopia's close leaves out (it stops ~10 degrees short): in front of
        the handle, then straight in along the box's normal until the handle is `past` beyond
        where it stands shut, and back out."""
        h = frame.to_frame(self.handle_position(obj))
        shut_x = frame.to_frame(self.shut_handle[obj])[0]
        self.move_to(frame.to_world([h[0] - back, h[1], h[2]]), quat=quat, tol=0.02)
        self.move_to(frame.to_world([shut_x + past, h[1], h[2]]), quat=quat, speed=0.05, tol=0.01)
        self.move_to(frame.to_world([shut_x - back, h[1], h[2]]), quat=quat, tol=0.02)
        self.settle(30)

    def list_prims(self):
        """Every prim under /World two levels down, with its size and centre: for completing
        the config's hide list."""
        rows = []
        for prim in self.stage.GetPrimAtPath("/World").GetChildren():
            path = str(prim.GetPath())
            size = self.utils.get_object_size(object_path=path)
            c = self.utils.get_geometry_center(object_path=path)
            rows.append((path, prim.GetTypeName(), size, c))
        return rows


def hand_rotation(approach, closing):
    """The hand's rotation from the way it points (its z, the approach) and the line its fingers
    close along (its y), as cuRobo's panda_hand frame has them."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float)
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack([np.cross(y, z), y, z]))


def look_at(eye, target):
    """Quaternion (w, x, y, z) turning a camera's world-axes view direction (+x, z up) from
    `eye` towards `target`."""
    d = np.array(target, float) - np.array(eye, float)
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.hypot(d[0], d[1]))
    x, y, z, w = R.from_euler("xyz", [0, pitch, yaw]).as_quat()
    return np.array([w, x, y, z])
