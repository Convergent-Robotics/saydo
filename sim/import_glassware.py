"""The lab tasks' glassware, from Laboratory.glb ("Laboratory Equipment" by ozdemirbr on CGTrader,
https://www.cgtrader.com/items/4224377; not in the repository, download it from there). Run once on the instance, after setup_brev.sh:

    python sim/import_glassware.py ~/glassware-src/Laboratory.glb

The file is converted with Isaac's asset converter (external/glassware/Laboratory.usd), and each
vessel the tasks use is written to external/glassware/<name>.usd, metres and Z up, its origin at
the middle of its base, sized as the LabUtopia vessel it stands in for (VESSELS). Each has:

  mesh           the model's own mesh, as the cameras see it: thin-walled OmniGlass (GLASS),
                 double-sided. lab_001's glass beakers, emptied, showed a
                 star-shaped crack in their base at ablation 7 (the full run, 17 Nov 2026), which
                 models took for a cracked beaker. The mesh keeps the model's own Y-up frame,
                 scaled and turned upright by its transform, so sim/liquid.py's interiors are
                 measured in it as LabUtopia's are in theirs.
  spai_collider  unseen (purpose guide): a closed shell turned from the model's profile, WALL_LOCAL
                 thick about its surface, with a floor. The model is a single surface with no
                 thickness, which a convex decomposition would fill solid, and then a rim pinch has
                 no wall to close on. Bench._make_dynamic makes only this collide, and
                 Bench._profiles reads the vessel's shape off it (the model's spout left out).

Each mesh's interior (sim/liquid.py) is printed at the end, in the mesh's own units, for
liquid.GLASSWARE: the same for every size made from one model.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("glb", help="Laboratory.glb")
parser.add_argument("--force", action="store_true", help="convert the glb again")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()

import omni.kit.asset_converter as converter  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

def out(text):
    sys.__stdout__.write(text + "\n")  # past Isaac's hold on print()
    sys.__stdout__.flush()


OUT = Path(__file__).resolve().parent.parent / "external" / "glassware"
# name: (the model's node in Laboratory.glb, its height in metres, whether it has a spout), each
# standing in for one of lab_001's vessels at about its size (its bounding box, sim/ probe, 18 Nov 2026).
VESSELS = {
    # lab_001's beaker2 and beaker3, 7.5 cm across, 9.0 cm tall. Its wall 7.6 cm across, as wide as
    # theirs: under the fingers' 7.5 cm (GRIP_OPEN_M), pick would close round the body as round a
    # bottle's neck, not pinch the rim. The model is taller for its width than lab_001's, as real
    # Griffin beakers are.
    "beaker_400": ("Beaker", 0.109, True),
    # beaker1, 10.5 cm across, 12.7 cm tall: a 1 L Griffin beaker is 10.5 x 14.5 cm.
    "beaker_1000": ("Beaker", 0.144, True),
    # conical_bottle02, 9.3 cm across, 16.5 cm tall, and as tall (11 cm across): a 250 mL Erlenmeyer
    # flask is 8.5 x 14.5 cm, but at 14.5 cm the hand, holding it by the neck, came down too low
    # to put it into the sunk sink (3 scripted replays of 3, 18 Nov 2026).
    "erlenmeyer_250": ("Erlenmayer Flask", 0.165, False),
    # conical_bottle04, 14.1 x 23.4 cm: a 1 L flask is 13.1 x 23 cm.
    "erlenmeyer_1000": ("Erlenmayer Flask", 0.225, False),
    # conical_bottle01, 7.7 x 9.8 cm: a 100 mL flask is 6.4 x 10.5 cm.
    "erlenmeyer_100": ("Erlenmayer Flask", 0.105, False),
    # graduated_cylinder_03, 27.2 cm tall (11.7 cm across its foot), and as tall: a 100 mL cylinder is
    # ~25 cm, but at 26 cm the rinse task's pick found no way up 4 times in 5 (18 Nov 2026).
    "cylinder_100": ("Graduated Cylinder", 0.272, True),
}
WALL_LOCAL = 0.02   # the collider's wall, in the models' units: ~2-4 mm at the sizes above
SEGMENTS = 32       # round the collider
# lab_001's own glass is OmniGlass, solid: light through a closed wall is refracted as through a
# lens. Thin-walled, the model's one surface is a sheet of glass, as a vessel's wall is, and nothing
# is refracted through its base. A UsdPreviewSurface with opacity (World._relook's `glass`) rendered
# opaque white in the episodes' renderer (the full run's clean-down waste beaker, 17 Nov 2026).
GLASS = {"color": (0.97, 0.985, 1.0), "ior": 1.5}


async def convert(src, dst):
    ctx = converter.AssetConverterContext()
    ctx.keep_all_materials = True
    ctx.use_meter_as_world_unit = True
    task = converter.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    if not await task.wait_until_finished():
        raise RuntimeError(f"{src}: {task.get_status()} {task.get_error_message()}")


def run(coro):
    loop = asyncio.get_event_loop()
    fut = asyncio.ensure_future(coro)
    while not fut.done():
        app.update()
        loop.run_until_complete(asyncio.sleep(0))
    fut.result()


def node_mesh(stage, node):
    """The mesh the converter made of a glb node, and its points in the file's frame (Y up)."""
    want = node.replace(" ", "_").replace("-", "_")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and any(a.GetName() in (node, want) for a in [prim, prim.GetParent()]):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            mesh = UsdGeom.Mesh(prim)
            p = np.array(mesh.GetPointsAttr().Get(), float)
            return mesh, p @ m[:3, :3] + m[3, :3], m
    raise KeyError(f"no mesh for glb node {node!r}; meshes: "
                   f"{[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]}")


def profile(p, spout):
    """The model's surface as [(height, radius)] from its base up, about its axis, in its own frame
    (centred: axis at x = z = 0, base at y = 0). Each ring of points at one height gives the largest
    radius it reaches, the spout's sector left out (a beaker's spout is one side of its top rings)."""
    r = np.hypot(p[:, 0], p[:, 2])
    keep = np.ones(len(p), bool)
    if spout:
        tip = p[np.argmax(r)]
        a = np.arctan2(p[:, 2], p[:, 0]) - np.arctan2(tip[2], tip[0])
        keep = np.abs(np.angle(np.exp(1j * a))) > np.radians(50)
    ys = np.round(p[:, 1], 3)
    out = []
    for y in np.unique(ys):
        sel = (ys == y) & keep & (r > 1e-6)
        if sel.any():
            out.append((float(p[sel, 1].mean()), float(r[sel].max())))
    return out


def shell(prof, t):
    """A closed shell turned from `prof` ([(y, r)], y up from 0): its surface thickened t/2 either
    side, with a floor t thick. (points, face vertex counts, indices)."""
    a = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    ys, rs = np.array([y for y, _ in prof]), np.array([r for _, r in prof])
    outer = [(y, r + t / 2) for y, r in prof]
    inner = [(t, max(float(np.interp(t, ys, rs)) - t / 2, 0.2 * rs.max()))] + \
            [(y, max(r - t / 2, 0.2 * rs.max())) for y, r in prof if y > t * 1.5]
    pts, counts, idx = [[0.0, 0.0, 0.0]], [], []

    def ring(y, r):
        start = len(pts)
        pts.extend([[r * np.cos(x), y, r * np.sin(x)] for x in a])
        return start

    o = [ring(y, r) for y, r in outer]
    i = [ring(y, r) for y, r in inner]
    floor = len(pts)
    pts.append([0.0, t, 0.0])
    n = SEGMENTS
    for j in range(n):  # the base, facing down
        counts.append(3)
        idx += [0, o[0] + (j + 1) % n, o[0] + j]
    for k in range(len(o) - 1):  # the outside, up
        for j in range(n):
            counts.append(4)
            idx += [o[k] + j, o[k] + (j + 1) % n, o[k + 1] + (j + 1) % n, o[k + 1] + j]
    for j in range(n):  # the rim, from the outside's top ring to the inside's
        counts.append(4)
        idx += [o[-1] + j, o[-1] + (j + 1) % n, i[-1] + (j + 1) % n, i[-1] + j]
    for k in range(len(i) - 1, 0, -1):  # the inside, down
        for j in range(n):
            counts.append(4)
            idx += [i[k] + j, i[k] + (j + 1) % n, i[k - 1] + (j + 1) % n, i[k - 1] + j]
    for j in range(n):  # the floor, facing up
        counts.append(3)
        idx += [floor, i[0] + j, i[0] + (j + 1) % n]
    return np.array(pts), counts, idx, inner


def write(name, mesh, p, height, spout):
    """external/glassware/<name>.usd; returns the interior of its mesh, in the mesh's units."""
    stage = Usd.Stage.CreateNew(str(OUT / f"{name}.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Vessel")
    stage.SetDefaultPrim(root.GetPrim())
    s = height / float(p[:, 1].max())
    # Gf matrices act on row vectors: scale, then turn Y up to Z up.
    frame = Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s)) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))

    look = UsdGeom.Mesh.Define(stage, "/Vessel/mesh")
    look.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    look.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
    look.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
    normals = mesh.GetNormalsAttr().Get()
    if normals:
        look.CreateNormalsAttr(normals)
        look.SetNormalsInterpolation(mesh.GetNormalsInterpolation())
    else:
        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if pv.GetPrimvarName() == "normals":
                look.CreateNormalsAttr(pv.ComputeFlattened())
                look.SetNormalsInterpolation(pv.GetInterpolation())
    look.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    look.CreateDoubleSidedAttr(True)  # one surface, no thickness: seen from inside too
    # Thin-walled glass casts a full shadow in the episodes' renderer: the liquid inside lay in it,
    # black but for the patch the light reached through the mouth (render check, 18 Nov 2026).
    UsdGeom.PrimvarsAPI(look).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    look.AddTransformOp().Set(frame)

    prof = profile(p, spout)
    pts, counts, idx, inner = shell(prof, WALL_LOCAL)
    col = UsdGeom.Mesh.Define(stage, "/Vessel/spai_collider")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    col.CreateFaceVertexCountsAttr(counts)
    col.CreateFaceVertexIndicesAttr(idx)
    col.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    col.AddTransformOp().Set(frame)

    mat = UsdShade.Material.Define(stage, "/Vessel/Looks/glass")
    shader = UsdShade.Shader.Define(stage, "/Vessel/Looks/glass/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(GLASS["ior"])
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*GLASS["color"]))
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(look.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    lo, hi = p.min(0), p.max(0)
    out(f"{name}: {np.round((hi - lo) * s, 4).tolist()} m (x, height, z), wall {WALL_LOCAL * s * 1000:.1f} mm")
    return {"points": len(p), "min": np.round(lo, 6).tolist(), "radius": max(r for _, r in inner),
            "bottom": inner[0][0], "top": float(hi[1]), "lower_profile": [(round(y, 6), round(r, 6)) for y, r in inner]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    usd = OUT / "Laboratory.usd"
    if args.force or not usd.exists():
        run(convert(Path(args.glb).expanduser().resolve(), usd))
    src = Usd.Stage.Open(str(usd))
    if UsdGeom.GetStageUpAxis(src) != UsdGeom.Tokens.y:
        raise SystemExit(f"{usd}: expected the converter's Y-up file, got {UsdGeom.GetStageUpAxis(src)}")
    interiors = {}
    for name, (node, height, spout) in VESSELS.items():
        mesh, p, _ = node_mesh(src, node)
        # Centred: the axis (the middle of its body's rings, a tenth to nine tenths up, clear of a
        # spout's rings at the top and a foot's at the bottom) at x = z = 0, the base at y = 0.
        y0, y1 = p[:, 1].min(), p[:, 1].max()
        body = p[(p[:, 1] > y0 + 0.1 * (y1 - y0)) & (p[:, 1] < y0 + 0.9 * (y1 - y0))]
        axis = (body.min(0) + body.max(0)) / 2
        p = p - [axis[0], y0, axis[2]]
        interiors[node] = write(name, mesh, p, height, spout)
    out("\n# liquid.GLASSWARE: (point count, the mesh's min corner) -> interior, in the mesh's units")
    for node, it in interiors.items():
        out(f"{node!r}: {it}")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc(file=sys.__stderr__)  # app.close() exits before Python would print it
    raise
finally:
    app.close()
