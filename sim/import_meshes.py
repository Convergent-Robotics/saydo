"""Turn models from the private_meshes repository (URDF + glb, external/private_meshes) into static
USD scenery for the household tasks. Run once on the instance, after setup_brev.sh:

    python sim/import_meshes.py objects/articulated/kitchen-appliances/102-kitchen_run_appliances_long
    python sim/import_meshes.py --z-up objects/articulated/kitchen-appliances/044-kettle   # its glbs are Z up

Each glb part is converted with Isaac's asset converter and placed where the URDF puts its link
with every joint at rest (doors shut, drawers in). The URDF's collision boxes (one per carcass
panel, front and worktop piece) become invisible box colliders, so the planner (motion.py, which
takes each collision prim's bounding box) and physics see the cupboards as they are, not as one
block. Written to external/private_meshes_usd/<model>/model.usd, metres and Z up, origin at floor
level in the middle of the footprint, front towards -y, as the URDF has it.

The meshes are not committed: objects #001-#150 of private_meshes have no settled licence yet
(its NOTICE.md). Fetch them with setup_brev.sh or by hand at MESHES_COMMIT (common.py).
"""
import argparse
import asyncio
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("models", nargs="+", help="model folders, relative to external/private_meshes")
parser.add_argument("--force", action="store_true", help="convert the glb parts again")
parser.add_argument("--z-up", action="store_true",
                    help="the glb parts are authored Z up against the glTF convention (044-kettle's are): no turn")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()

import omni.kit.asset_converter as converter  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MESHES, MESHES_USD  # noqa: E402


def origin(el):
    """A URDF <origin> as a 4x4 matrix (column vectors)."""
    m = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        m[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        m[:3, :3] = R.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
    return m


def rest_poses(robot):
    """Each link's pose in the model frame with every joint at zero."""
    joints = {j.find("child").get("link"): j for j in robot.findall("joint")}
    poses = {}

    def pose(link):
        if link not in poses:
            j = joints.get(link)
            poses[link] = np.eye(4) if j is None else pose(j.find("parent").get("link")) @ origin(j)
        return poses[link]

    for link in robot.findall("link"):
        pose(link.get("name"))
    return poses


def gf(m):
    """A column-vector matrix as USD's row-vector Gf.Matrix4d."""
    return Gf.Matrix4d(*m.T.flatten().tolist())


async def convert(src, dst):
    ctx = converter.AssetConverterContext()
    ctx.keep_all_materials = True
    ctx.use_meter_as_world_unit = True
    ctx.embed_textures = True
    task = converter.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    if not await task.wait_until_finished():
        raise RuntimeError(f"{src}: {task.get_status()} {task.get_error_message()}")


def run(coro):
    loop = asyncio.get_event_loop()
    fut = asyncio.ensure_future(coro)
    while not fut.done():
        app.update()
        loop.run_until_complete(asyncio.sleep(0))
    return fut.result()


def import_model(rel):
    folder = MESHES / rel
    robot = ET.parse(folder / "model.urdf").getroot()
    out = MESHES_USD / folder.name
    (out / "parts").mkdir(parents=True, exist_ok=True)
    poses = rest_poses(robot)

    stage = Usd.Stage.CreateNew(str(out / "model.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(root.GetPrim())
    boxes = 0
    for link in robot.findall("link"):
        name = link.get("name")
        lp = f"/Model/{name}"
        UsdGeom.Xform.Define(stage, lp).AddTransformOp().Set(gf(poses[name]))
        for i, vis in enumerate(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            src = folder / mesh.get("filename")
            part = out / "parts" / (src.stem + ".usd")
            if args.force or not part.exists():
                run(convert(src, part))
            # glTF is Y up: turn the part about x so its +y (the model's height) points up.
            up = UsdGeom.GetStageUpAxis(Usd.Stage.Open(str(part)))
            xf = UsdGeom.Xform.Define(stage, f"{lp}/visual_{i}")
            m = origin(vis)
            if up == UsdGeom.Tokens.y and not args.z_up:
                turn = np.eye(4)
                turn[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()
                m = m @ turn
            xf.AddTransformOp().Set(gf(m))
            xf.GetPrim().GetReferences().AddReference(f"./parts/{part.name}")
        for i, col in enumerate(link.findall("collision")):
            box = col.find("geometry/box")
            if box is None:
                continue
            size = np.array([float(v) for v in box.get("size").split()])
            cube = UsdGeom.Cube.Define(stage, f"{lp}/collision_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTransformOp().Set(gf(origin(col)))
            cube.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)  # not rendered; motion.py reads guides too
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            boxes += 1
    stage.GetRootLayer().Save()
    print(f"{rel}: {len(poses)} links, {boxes} collision boxes -> {out / 'model.usd'}", flush=True)


for model in args.models:
    import_model(model)
sys.stdout.flush()
app.close()
