"""Fixtures LabUtopia does not have, built from primitives: a sink (on the bench or sunk into it), a stripe on the
bench, a plain block, a centrifuge tube, an open plastic jar (a discard pot); and for the household tasks a ceramic hob, a monitor
showing a photo, a picture region on it, a tea towel on a rail, a roller blind, a café curtain on a rod, a rumpled tea towel, a dish rack, a jar candle, a lighter, a
saucepan, a toy box, a kitchen knife and a wooden spoon.

A task's config asks for one with `build:` in place of `prim:` (see sim/README.md). Each is built
under /World/spai/<name> before physics starts, with its base centre at the drawn position, so
`position: [x, y, bench height]` stands it on the bench (a towel's position is the top of its
rail, which it hangs from). Import only after SimulationApp exists.
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = "/World/spai"
STEEL = (0.62, 0.64, 0.66)
DARK = (0.08, 0.08, 0.08)
WALL = 0.015  # sink wall and floor thickness, metres
JAR_WALL = 0.003  # a moulded plastic jar's wall and floor thickness, metres
FLANGE = 0.025  # a sunk sink's steel rim on the bench round its opening: width, metres
FLANGE_T = 0.004  # and thickness
TAP_H = 0.34  # a sunk sink's gooseneck tap: its spout's height above the bench, metres
# Textures of fixtures without a label go here, a folder per process: overlapping episodes of one
# scene crashed when each launch rewrote the files its neighbour had loaded (sim/README.md, "Running
# several at once").
_TEXTURES = tempfile.TemporaryDirectory(prefix="spai_fixtures_")
TEXTURES = Path(_TEXTURES.name)


def build(stage, name, spec, position, label_png=None, root=ROOT, assets=None):
    """Build fixture `name` from its `build:` block under `root`. Returns its prim path. `assets`
    is where image files named in the block (a screen's `image`) are found."""
    path = f"{root}/{name}"
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    if spec.get("yaw"):
        xf.AddRotateZOp().Set(float(spec["yaw"]))
    kind = spec["kind"]
    size = np.array(spec["size"], float)
    out_dir = Path(label_png).parent if label_png else TEXTURES
    if kind in HOUSEHOLD:
        HOUSEHOLD[kind](stage, path, size, spec, out_dir / name, assets)
        return path
    if kind == "basin":
        (_sunk_basin if spec.get("sunk") else _basin)(stage, path, size)
    elif kind == "stripe":
        out = (Path(label_png).parent if label_png else TEXTURES) / f"{name}_stripe.png"
        png = _stripe_texture(size, label_png, out)
        _box(stage, f"{path}/stripe", [0, 0, size[2] / 2], size,
             material(stage, f"{path}/Looks/stripe", texture=png), collide=False, uv=True)
    elif kind == "block":
        if spec.get("look"):  # the box becomes the model's bounds, fitted into `size`
            size = _dress(stage, path, spec["look"], size, assets, spec.get("look_tilt"))
        _box(stage, f"{path}/body", [0, 0, size[2] / 2], size,
             material(stage, f"{path}/Looks/body", color=spec.get("color", (0.85, 0.85, 0.85)),
                      roughness=spec.get("roughness", 0.6), opacity=spec.get("opacity", 1.0)),
             collide=spec.get("collide", True))  # opacity 0: unseen (a numbered patch of bare worktop)
        if spec.get("lid"):  # a bin's lid: a slab 1.5 cm thick, 6 mm wider each way, a shade darker, on top
            c = np.array(spec.get("color", (0.85, 0.85, 0.85)), float)
            _box(stage, f"{path}/spai_lid", [0, 0, size[2] + 0.0075], [size[0] + 0.012, size[1] + 0.012, 0.015],
                 material(stage, f"{path}/Looks/lid", color=tuple(c * 0.8), roughness=0.45), collide=False)
        if spec.get("holes"):  # a tube rack: [rows, columns] of dark holes in its top, drawn, not cut
            _holes(stage, path, size, spec["holes"])
        if spec.get("dynamic"):
            _rigid(stage, path if spec.get("look") else f"{path}/body", spec.get("mass", 0.1))
    elif kind == "tube":
        # size: [diameter, diameter, height]. A self-standing conical tube (a 50 mL Falcon is 29 x
        # 115 mm), open, in frosted polypropylene: what it holds does not show, as in a real one.
        # Always dynamic: it is there to be picked, poured into and carried.
        if spec.get("look"):
            _dress(stage, path, spec["look"], size, assets, spec.get("look_tilt"))
        _prism(stage, f"{path}/body", [0, 0, size[2] / 2], size[0] / 2, size[2],
               material(stage, f"{path}/Looks/body", color=spec.get("color", (0.90, 0.91, 0.92)), roughness=0.35))
        if spec.get("pen"):
            _pen_dress(stage, path, size, spec["pen"])
        # Dressed (a look, a pen's parts beside the body), the whole object is the rigid body, so they move with it.
        _rigid(stage, path if spec.get("look") or spec.get("pen") else f"{path}/body", spec.get("mass", 0.03))
    else:
        raise ValueError(f"{name}: unknown fixture kind {kind!r}")
    if spec.get("look"):  # the plain shape stays as the collider and what pick reads, unseen
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/body")).CreatePurposeAttr(UsdGeom.Tokens.guide)
    return path


def _dress(stage, path, look, size, assets, tilt=None):
    """`look`: a model under external/ (external/home_assets/<name>.usd, made by
    sim/import_glb.py) shown in place of a block's or a tube's plain shape. It is turned Z up
    (`tilt` degrees about x; by default 90 for a Y-up file), scaled evenly to fit inside `size`,
    centred on the shape's axis and stood on its bottom, under `<path>/spai_look` (pick's shape
    profile leaves spai_ prims out). The rigid body moves to the object's root, so the model goes
    with it. Returns the fitted model's size."""
    url = Path(assets) / "external" / look
    if not url.exists():
        raise FileNotFoundError(f"{url}: run sim/import_glb.py on its glb (assets/home/)")
    xf = UsdGeom.Xform.Define(stage, f"{path}/spai_look")
    # The file on a child, so its root keeps its own turns: the converter's root undoes a turn
    # further down, and dropping it laid the duster on its side (28 Oct 2026).
    UsdGeom.Xform.Define(stage, f"{path}/spai_look/model").GetPrim().GetReferences().AddReference(str(url))
    move, scale = xf.AddTranslateOp(), xf.AddScaleOp()
    move.Set(Gf.Vec3d(0, 0, 0))
    scale.Set(Gf.Vec3d(1, 1, 1))
    if tilt is None:
        tilt = 90.0 if UsdGeom.GetStageUpAxis(Usd.Stage.Open(str(url))) == UsdGeom.Tokens.y else 0.0
    if tilt:
        xf.AddRotateXOp().Set(float(tilt))
    # Its bounds in the object's frame, from the meshes' points.
    to_root = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).GetInverse())
    pts = []
    for prim in Usd.PrimRange(xf.GetPrim(), Usd.TraverseInstanceProxies()):
        p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], float) if prim.IsA(UsdGeom.Mesh) else []
        if len(p):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ to_root
            pts.append(p @ m[:3, :3] + m[3, :3])
    pts = np.concatenate(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    s = float(np.min(np.asarray(size) / (hi - lo)))
    scale.Set(Gf.Vec3d(s, s, s))
    move.Set(Gf.Vec3d(*(-s * np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])).tolist()))
    return s * (hi - lo)


def _holes(stage, path, size, holes):
    """Dark discs on a block's top, `holes` = [rows along x, columns along y], evenly spaced."""
    rows, cols = holes
    dark = material(stage, f"{path}/Looks/hole", color=DARK)
    dx, dy = size[0] / rows, size[1] / cols
    for i in range(rows):
        for j in range(cols):
            hole = UsdGeom.Cylinder.Define(stage, f"{path}/spai_hole{i}_{j}")
            hole.CreateRadiusAttr(0.36 * min(dx, dy))
            hole.CreateHeightAttr(0.001)
            hole.AddTranslateOp().Set(Gf.Vec3d((i + 0.5) * dx - size[0] / 2, (j + 0.5) * dy - size[1] / 2, size[2] + 0.0005))
            UsdShade.MaterialBindingAPI.Apply(hole.GetPrim()).Bind(dark)


def _basin(stage, path, size):
    """An open steel basin (floor and four walls), a dark drain disc and a tap behind it on the
    side away from the robot (+x). Walls and floor collide, so a beaker set in it stands."""
    sx, sy, sz = size
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.8, roughness=0.35)
    dark = material(stage, f"{path}/Looks/drain", color=DARK)
    _box(stage, f"{path}/floor", [0, 0, WALL / 2], [sx, sy, WALL], steel)
    for i, (cx, cy, dx, dy) in enumerate([(-(sx - WALL) / 2, 0, WALL, sy), ((sx - WALL) / 2, 0, WALL, sy),
                                          (0, -(sy - WALL) / 2, sx, WALL), (0, (sy - WALL) / 2, sx, WALL)]):
        _box(stage, f"{path}/wall{i}", [cx, cy, sz / 2], [dx, dy, sz], steel)
    drain = UsdGeom.Cylinder.Define(stage, f"{path}/drain")
    drain.CreateRadiusAttr(0.025)
    drain.CreateHeightAttr(0.002)
    drain.AddTranslateOp().Set(Gf.Vec3d(0, 0, WALL + 0.001))
    UsdShade.MaterialBindingAPI.Apply(drain.GetPrim()).Bind(dark)
    # The tap: a riser behind the back wall and a spout reaching over the basin.
    riser_h = sz + 0.18
    riser = UsdGeom.Cylinder.Define(stage, f"{path}/tap_riser")
    riser.CreateRadiusAttr(0.012)
    riser.CreateHeightAttr(riser_h)
    riser.AddTranslateOp().Set(Gf.Vec3d(sx / 2 + 0.02, 0, riser_h / 2))
    spout_l = sx * 0.4
    spout = UsdGeom.Cylinder.Define(stage, f"{path}/tap_spout")
    spout.CreateRadiusAttr(0.01)
    spout.CreateHeightAttr(spout_l)
    spout.CreateAxisAttr("X")
    spout.AddTranslateOp().Set(Gf.Vec3d(sx / 2 + 0.02 - spout_l / 2, 0, riser_h - 0.01))
    for p in (riser, spout):
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(steel)


def basin_floor(spec, position):
    """Height of a built sink's floor, where things stand in it."""
    return float(position[2]) + (WALL - spec["size"][2] if spec.get("sunk") else WALL)


def basin_rim(spec, position):
    """Height of a built sink's rim: its walls' top, or a sunk sink's flange on the bench."""
    return float(position[2]) + (FLANGE_T if spec.get("sunk") else spec["size"][2])


def _sunk_basin(stage, path, size):
    """A steel sink set into the bench, as lab sinks are: `size` is its opening (x, y) and its
    depth, the position the bench top at its middle. Floor and walls hang below the bench (the
    bench is cut round the opening: Bench._cut_bench), a steel flange lies on the bench round
    the opening, a drain sits in the floor towards the back and a gooseneck tap stands behind the
    back flange, on the side away from the robot (+x), its spout over the drain. Floor, walls and
    flange collide; the tap does not, as the bench-top sink's does not."""
    sx, sy, sz = size
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.8, roughness=0.35)
    dark = material(stage, f"{path}/Looks/drain", color=DARK)
    _box(stage, f"{path}/floor", [0, 0, WALL / 2 - sz], [sx, sy, WALL], steel)
    for i, (cx, cy, dx, dy) in enumerate([(-(sx - WALL) / 2, 0, WALL, sy), ((sx - WALL) / 2, 0, WALL, sy),
                                          (0, -(sy - WALL) / 2, sx, WALL), (0, (sy - WALL) / 2, sx, WALL)]):
        _box(stage, f"{path}/wall{i}", [cx, cy, -sz / 2], [dx, dy, sz], steel)
    fx, fy = sx + 2 * FLANGE, sy + 2 * FLANGE
    for i, (cx, cy, dx, dy) in enumerate([(-(sx + FLANGE) / 2, 0, FLANGE, fy), ((sx + FLANGE) / 2, 0, FLANGE, fy),
                                          (0, -(sy + FLANGE) / 2, sx, FLANGE), (0, (sy + FLANGE) / 2, sx, FLANGE)]):
        _box(stage, f"{path}/flange{i}", [cx, cy, FLANGE_T / 2], [dx, dy, FLANGE_T], steel)
    # The drain, a third of the way from the back wall to the middle: a drain in the middle is
    # hidden behind the near wall from the overview camera's height.
    drain_x = sx / 6
    drain = UsdGeom.Cylinder.Define(stage, f"{path}/drain")
    drain.CreateRadiusAttr(0.03)
    drain.CreateHeightAttr(0.002)
    drain.AddTranslateOp().Set(Gf.Vec3d(drain_x, 0, WALL + 0.001 - sz))
    UsdShade.MaterialBindingAPI.Apply(drain.GetPrim()).Bind(dark)
    # The tap: a riser on the bench behind the back flange, a spout out over the drain and a
    # short nozzle pointing down.
    riser_x = fx / 2 + 0.03
    riser = UsdGeom.Cylinder.Define(stage, f"{path}/tap_riser")
    riser.CreateRadiusAttr(0.014)
    riser.CreateHeightAttr(TAP_H)
    riser.AddTranslateOp().Set(Gf.Vec3d(riser_x, 0, TAP_H / 2))
    spout_l = riser_x - drain_x
    spout = UsdGeom.Cylinder.Define(stage, f"{path}/tap_spout")
    spout.CreateRadiusAttr(0.011)
    spout.CreateHeightAttr(spout_l)
    spout.CreateAxisAttr("X")
    spout.AddTranslateOp().Set(Gf.Vec3d(riser_x - spout_l / 2, 0, TAP_H - 0.011))
    nozzle = UsdGeom.Cylinder.Define(stage, f"{path}/tap_nozzle")
    nozzle.CreateRadiusAttr(0.011)
    nozzle.CreateHeightAttr(0.04)
    nozzle.AddTranslateOp().Set(Gf.Vec3d(drain_x, 0, TAP_H - 0.03))
    base = UsdGeom.Cylinder.Define(stage, f"{path}/tap_base")
    base.CreateRadiusAttr(0.028)
    base.CreateHeightAttr(0.02)
    base.AddTranslateOp().Set(Gf.Vec3d(riser_x, 0, 0.01))
    for p in (riser, spout, nozzle, base):
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(steel)


def flat_shading(mesh, points, faces, radial=False):
    """Draw `mesh` as the faces it has, not a subdivision surface: USD's default scheme is
    Catmull-Clark, under which RTX rounded every built box into a pillow, its base shaded into a dark
    arch (the waste bins, the bread bin, 10 Nov 2026). Each face gets its own normal, so the edges
    stay sharp; with `radial`, faces whose normal is horizontal (a cylinder's side) take the radial
    direction at each corner instead, so a round side shades smoothly, as a tube's does."""
    pts = np.array([[p[0], p[1], p[2]] for p in points], float)
    normals = []
    for f in faces:
        c = pts[list(f)]
        n = np.zeros(3)
        for i in range(len(f)):  # Newell's normal: right for any planar polygon
            a, b = c[i], c[(i + 1) % len(f)]
            n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
        n /= np.linalg.norm(n) or 1.0
        for v in c:
            if radial and abs(n[2]) < 0.5 and np.hypot(v[0], v[1]) > 1e-9:
                r = np.array([v[0], v[1], 0.0])
                r *= np.sign(np.dot(r, n)) or 1.0  # an inner wall faces in
                normals.append(r / np.linalg.norm(r))
            else:
                normals.append(n)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, n)) for n in normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


def _box(stage, path, centre, size, material, collide=True, uv=False):
    """A box mesh (not UsdGeom.Cube, so the stripe can carry texture coordinates)."""
    hx, hy, hz = np.array(size, float) / 2
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
           (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)])
    flat_shading(mesh, pts, faces)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    if uv:
        # The top face shows the whole texture: u along -y (the robot's right, as seen from -x),
        # v along +x (away from the robot), so the text reads upright from the robot's side.
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        top = {4: (1, 0), 5: (1, 1), 6: (0, 1), 7: (0, 0)}
        st.Set([top.get(i, (0, 0)) if fi == 1 else (0, 0) for fi, f in enumerate(faces) for i in f])
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _prism(stage, path, centre, radius, height, material, sides=24, collide=True):
    """An upright cylinder as a mesh (a UsdGeom.Cylinder has no points, and pick's grasps are
    worked out from an object's mesh points). It collides unless told not to; a dynamic body takes
    its convex hull."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = [(radius * np.cos(t), radius * np.sin(t)) for t in a]
    zs = np.array([-height / 2, height / 2])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, float(z)) for z in zs for x, y in ring])
    top = (len(zs) - 1) * sides
    faces = [list(range(sides))[::-1], list(range(top, top + sides))]
    for k in range(len(zs) - 1):
        o = k * sides
        faces += [[o + i, o + (i + 1) % sides, o + sides + (i + 1) % sides, o + sides + i] for i in range(sides)]
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
    flat_shading(mesh, [(x, y, float(z)) for z in zs for x, y in ring], faces, radial=True)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def material(stage, path, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.6, texture=None, opacity=1.0,
             emissive=False, emission=None):
    """A UsdPreviewSurface. `opacity` below 1 shows through (glass) and 0 hides the surface, as
    the episodes run with fractional cutout opacity on; `emissive` makes it glow in its own
    colour or texture (a display), `emission` in another colour than its surface's."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if opacity == 0.0:  # cut out; without a threshold RTX renders it as dark glass
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    if texture:
        reader = UsdShade.Shader.Define(stage, f"{path}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        if emissive:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        if emissive or emission:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(emission or color)))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _stripe_texture(size, label_png, out):
    """Yellow-and-black hazard bands along the stripe, with the label card in the middle. The
    image's width runs along the stripe's y, its height along x. A stripe that runs along x is
    drawn along its length and turned, so its label reads from the robot's right."""
    px_per_m = 2000
    along_x = size[0] > size[1]
    length, width = (size[0], size[1]) if along_x else (size[1], size[0])
    w, h = max(int(length * px_per_m), 64), max(int(width * px_per_m), 16)
    img = Image.new("RGB", (w, h), (240, 200, 0))
    draw = ImageDraw.Draw(img)
    band = max(h, 20)
    for x0 in range(-h, w + h, 2 * band):
        draw.polygon([(x0, h), (x0 + band, h), (x0 + band + h, 0), (x0 + h, 0)], fill=(15, 15, 15))
    if label_png:
        card = Image.open(label_png).convert("RGB")
        ch = int(h * 0.9)
        card = card.resize((int(card.width * ch / card.height), ch))
        img.paste(card, ((w - card.width) // 2, (h - ch) // 2))
    if along_x:
        img = img.rotate(90, expand=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# ---- dressing an enclosure (world.py's `dress:`) ----

def box_on(stage, path, corners, to_local, material):
    """A box given by its 8 world corners, in _box's order (bottom face, then top), as a child of
    a moving body: `to_local` takes world points into the body's frame. No collision."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(to_local.Transform(Gf.Vec3d(*map(float, p)))) for p in corners])
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    flat_shading(mesh, [to_local.Transform(Gf.Vec3d(*map(float, p))) for p in corners], faces)
    mesh.CreateDoubleSidedAttr(True)  # the body's frame may be mirrored
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def box_corners(lo, hi):
    """The 8 corners of the axis-aligned box lo..hi, in _box's order."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


def quad_on(stage, path, corners, to_local, material):
    """A textured card through 4 world corners (texture's bottom left, bottom right, top right,
    top left) as a child of a moving body, as the label cards are."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(to_local.Transform(Gf.Vec3d(*map(float, p)))) for p in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    flat_shading(mesh, [to_local.Transform(Gf.Vec3d(*map(float, p))) for p in corners], [(0, 1, 2, 3)])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def display_texture(text, out, color=(255, 70, 40)):
    """A seven-segment-style readout: `text` lit on a near-black window, as on the controller of
    a lab fridge or freezer."""
    from labels import font
    w, h = 440, 200
    img = Image.new("RGB", (w, h), (12, 12, 14))
    draw = ImageDraw.Draw(img)
    f = font(100)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((w - box[0] - box[2]) / 2, (h - box[1] - box[3]) / 2), text, font=f, fill=color)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def bottle(stage, path, base, radius, height, color, cap, glass=False):
    """A reagent bottle standing at `base` (world x, y, z of its bottom): a body (amber glass or
    white plastic) and a screw cap. Visual only."""
    UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(*map(float, base)))
    body_h, cap_h = height * 0.78, height * 0.14
    parts = [("body", radius, body_h, body_h / 2, material(stage, f"{path}/Looks/body", color=color,
                                                           roughness=0.15 if glass else 0.5)),
             ("neck", radius * 0.45, height - body_h - cap_h, body_h + (height - body_h - cap_h) / 2,
              material(stage, f"{path}/Looks/neck", color=color, roughness=0.15 if glass else 0.5)),
             ("cap", radius * 0.55, cap_h, height - cap_h / 2, material(stage, f"{path}/Looks/cap", color=cap))]
    for name, r, h, z, mat in parts:
        cyl = UsdGeom.Cylinder.Define(stage, f"{path}/{name}")
        cyl.CreateRadiusAttr(float(r))
        cyl.CreateHeightAttr(float(h))
        cyl.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(z)))
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)


def carton(stage, path, base, size, color, yaw_deg=0.0):
    """A kit box standing at `base`, turned `yaw_deg` about the vertical. Visual only."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, base)))
    xf.AddRotateZOp().Set(float(yaw_deg))
    _box(stage, f"{path}/body", [0, 0, size[2] / 2], size, material(stage, f"{path}/Looks/body", color=color),
         collide=False)


# ---- household fixtures ----

def _show(stage, path, visible):
    img = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    img.MakeVisible() if visible else img.MakeInvisible()


def _flat_quad(stage, path, x0, x1, y, w, z, mat):
    """A card lying flat at height z, x0..x1 deep and w wide about y, reading upright from -x
    (the robot's side): the texture's bottom towards -x, its left towards +y."""
    return quad_on(stage, path, [(x0, y + w / 2, z), (x0, y - w / 2, z), (x1, y - w / 2, z), (x1, y + w / 2, z)],
                   Gf.Matrix4d(1), mat)


def _upright_quad(stage, path, x, y0, y1, z0, z1, mat):
    """A card standing in the plane x = const, facing -x: the texture's left at y1 (the viewer's
    left is +y), its bottom at z0."""
    return quad_on(stage, path, [(x, y1, z0), (x, y0, z0), (x, y0, z1), (x, y1, z1)], Gf.Matrix4d(1), mat)


def _rigid(stage, path, mass):
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateMassAttr(float(mass))


def _hob(stage, path, size, spec, out, assets):
    """A ceramic hob: a black glass top with its cooking zones and touch controls printed on, a
    cooking zone that glows red while it is on (`lit`: its index in `zones`), and over each zone's
    control a red readout: the power level while on, "H" while it is off but still hot (residual
    heat, as every ceramic hob shows). `size`: depth (x), width (y), glass thickness. `zones`:
    [x, y, radius] from the centre. hob_state switches the looks; the hob itself never moves."""
    sx, sy, sz = size
    zones = spec.get("zones", [[0.08, -0.13, 0.09], [0.08, 0.13, 0.075], [-0.1, -0.13, 0.075], [-0.1, 0.13, 0.09]])
    png = _hob_texture(size, zones, out.parent / f"{out.name}_glass.png")
    _box(stage, f"{path}/glass", [0, 0, sz / 2], size, material(stage, f"{path}/Looks/glass", texture=png, roughness=0.08),
         uv=True)
    x, y, r = zones[spec.get("lit", 0)]
    # Black glass lit from below in deep red. Lit by the kitchen's warm lights as well, a red
    # surface read as orange (28 Oct 2026).
    _prism(stage, f"{path}/glow", [x, y, sz + 0.0004], r * 0.9, 0.0006,
           material(stage, f"{path}/Looks/glow", color=(0.02, 0.0, 0.0), roughness=0.08,
                    emission=(0.55, 0.015, 0.0)), sides=48, collide=False)
    for state, text in (("on", str(spec.get("level", 6))), ("hot", "H")):
        tex = display_texture(text, out.parent / f"{out.name}_{state}.png", color=(255, 40, 30))
        _flat_quad(stage, f"{path}/display_{state}", -sx / 2 + 0.012, -sx / 2 + 0.042, y, 0.066, sz + 0.0006,
                   material(stage, f"{path}/Looks/display_{state}", texture=tex, emissive=True, roughness=0.2))
    # The touch controls' strip along the front edge: invisible, it carries the controls' number.
    _flat_quad(stage, f"{path}/controls", -sx / 2 + 0.008, -sx / 2 + 0.046, 0.0, sy * 0.8, sz + 0.0003,
               material(stage, f"{path}/Looks/controls", opacity=0.0))
    hob_state(stage, path, spec.get("state", "off"))


def hob_state(stage, path, state):
    """"on": the zone glows and its readout shows the level; "hot": "H"; "off": nothing lit."""
    _show(stage, f"{path}/glow", state == "on")
    _show(stage, f"{path}/display_on", state == "on")
    _show(stage, f"{path}/display_hot", state == "hot")


def _hob_texture(size, zones, out):
    """The glass's top: zone rings and the touch-control strip along the front edge, in the
    pale grey print of a ceramic hob. The image's top is the far edge (+x), its left +y."""
    sx, sy, _ = size
    px = 1600
    w, h = int(px * sy), int(px * sx)
    img = Image.new("RGB", (w, h), (10, 10, 12))
    d = ImageDraw.Draw(img)
    to_px = lambda x, y: ((sy / 2 - y) * px, (sx / 2 - x) * px)
    for x, y, r in zones:
        cx, cy = to_px(x, y)
        d.ellipse([cx - r * px, cy - r * px, cx + r * px, cy + r * px], outline=(95, 95, 100), width=3)
        d.ellipse([cx - 0.6 * r * px, cy - 0.6 * r * px, cx + 0.6 * r * px, cy + 0.6 * r * px], outline=(60, 60, 64), width=2)
    for x, y, _ in zones:  # a "-" and "+" beside each zone's readout
        cx, cy = to_px(-sx / 2 + 0.027, y)
        for dx, plus in ((-0.04, False), (0.04, True)):
            u = cx + dx * px
            d.line([u - 10, cy, u + 10, cy], fill=(150, 150, 155), width=3)
            if plus:
                d.line([u, cy - 10, u, cy + 10], fill=(150, 150, 155), width=3)
    cx, cy = to_px(-sx / 2 + 0.027, 0.0)  # the power key
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], outline=(150, 150, 155), width=3)
    d.line([cx, cy - 20, cx, cy - 2], fill=(150, 150, 155), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _screen(stage, path, size, spec, out, assets):
    """A monitor on a stand, its screen showing `image` (a file under `assets`), lit. `size`:
    the panel's depth (x), width (y) and height; `stand`: the panel's bottom above the base.
    screen_crack swaps in the same picture behind a shattered panel."""
    d, w, h = size
    stand = spec.get("stand", 0.10)
    bezel = spec.get("bezel", 0.012)
    black = material(stage, f"{path}/Looks/black", color=(0.03, 0.03, 0.035), roughness=0.35)
    _box(stage, f"{path}/foot", [0.03, 0, 0.006], [0.16, 0.22, 0.012], black)
    _box(stage, f"{path}/post", [d / 2 + 0.015, 0, (stand + h / 2) / 2], [0.025, 0.05, stand + h / 2], black)
    _box(stage, f"{path}/panel", [0, 0, stand + h / 2], [d, w, h], black)
    src = Image.open(Path(assets) / spec["image"]).convert("RGB")
    iw, ih = w - 2 * bezel, h - 2 * bezel
    src = _cover(src, iw / ih)
    whole = out.parent / f"{out.name}_image.png"
    whole.parent.mkdir(parents=True, exist_ok=True)
    src.save(whole)
    cracked = _crack(src, out.parent / f"{out.name}_cracked.png")
    x = -d / 2 - 0.001
    for name, png in (("image", whole), ("cracked", cracked)):
        _upright_quad(stage, f"{path}/{name}", x, -iw / 2, iw / 2, stand + bezel, stand + bezel + ih,
                      material(stage, f"{path}/Looks/{name}", texture=png, emissive=True, roughness=0.15))
    _show(stage, f"{path}/cracked", False)


def screen_crack(stage, path):
    _show(stage, f"{path}/image", False)
    _show(stage, f"{path}/cracked", True)


def _cover(img, aspect):
    """Crop to `aspect` (width / height) about the middle."""
    w, h = img.size
    if w / h > aspect:
        nw = int(h * aspect)
        return img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / aspect)
    return img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def _crack(img, out, seed=3):
    """The picture behind a shattered panel: a point of impact and cracks running from it."""
    rng = np.random.default_rng(seed)
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w * 0.5, h * 0.55
    for a in np.linspace(0, 2 * np.pi, 14, endpoint=False) + rng.uniform(0, 0.4, 14):
        x, y, step = cx, cy, w * 0.04
        for _ in range(int(rng.integers(6, 14))):
            a += rng.normal(0, 0.25)
            nx, ny = x + step * np.cos(a), y + step * np.sin(a)
            d.line([x, y, nx, ny], fill=(235, 235, 240), width=2)
            x, y = nx, ny
    for r in (0.04, 0.08):
        d.ellipse([cx - r * w, cy - r * w, cx + r * w, cy + r * w], outline=(220, 220, 225), width=2)
    img.save(out)
    return out


def _picture(stage, path, size, spec, out, assets):
    """Where a thing shown on a screen appears: an invisible card in the screen's face, `size`
    [0, width, height], its position the bottom middle. It carries the thing's number in the
    frames, as a detector would mark it, and nothing collides with it."""
    _, w, h = size
    _upright_quad(stage, f"{path}/region", 0.0, -w / 2, w / 2, 0.0, h,
                  material(stage, f"{path}/Looks/region", opacity=0.0))


def _towel(stage, path, size, spec, out, assets):
    """A tea towel hanging over a rail under a wall cabinet. The position is the rail's middle;
    the rail runs along y, the towel's front hangs `size[2]` below it, the back half as far.
    `size`: the cloth's thickness, width, drop. Static: the arm keeps clear of it."""
    t, w, drop = size
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.8, roughness=0.3)
    rail = UsdGeom.Cylinder.Define(stage, f"{path}/rail")
    rail.CreateRadiusAttr(0.007)
    rail.CreateHeightAttr(float(w + 0.10))
    rail.CreateAxisAttr("Y")
    UsdShade.MaterialBindingAPI.Apply(rail.GetPrim()).Bind(steel)
    for i, y in enumerate((-(w / 2 + 0.04), w / 2 + 0.04)):
        _box(stage, f"{path}/bracket{i}", [0.01, y, 0.012], [0.03, 0.008, 0.03], steel, collide=False)
    linen = material(stage, f"{path}/Looks/linen", color=(0.80, 0.76, 0.68), roughness=0.9)
    _box(stage, f"{path}/front", [-0.009, 0, 0.007 - drop / 2], [t, w, drop], linen)
    _box(stage, f"{path}/back", [0.009, 0, 0.007 - drop * 0.3], [t, w, drop * 0.6], linen)
    _box(stage, f"{path}/fold", [0, 0, 0.009], [0.018 + t, w, t], linen, collide=False)
    png = _linen_texture(w, drop, out.parent / f"{out.name}_linen.png")
    _upright_quad(stage, f"{path}/print", -0.009 - t / 2 - 0.0005, -w / 2, w / 2, 0.007 - drop, 0.007 + 0.002,
                  material(stage, f"{path}/Looks/print", texture=png, roughness=0.9))


def _blind(stage, path, size, spec, out, assets):
    """A fabric roller blind let down in a window: the position is the roller's middle, the roller
    runs along y, the cloth (`fabric`, what the candle task's predicates read) hangs `size[2]` below
    it on the roller's front (-x), with a weighted bar at the hem and a pull cord at its +y end.
    `size`: the cloth's thickness, width, drop. Static: the arm keeps clear of it."""
    t, w, drop = size
    r = 0.016
    white = material(stage, f"{path}/Looks/roller", color=(0.93, 0.93, 0.92), roughness=0.4)
    roller = UsdGeom.Cylinder.Define(stage, f"{path}/roller")
    roller.CreateRadiusAttr(r)
    roller.CreateHeightAttr(float(w + 0.02))
    roller.CreateAxisAttr("Y")
    UsdShade.MaterialBindingAPI.Apply(roller.GetPrim()).Bind(white)
    cloth = material(stage, f"{path}/Looks/fabric", color=tuple(spec.get("color", (0.80, 0.76, 0.66))), roughness=0.95)
    _box(stage, f"{path}/fabric", [-r, 0, -drop / 2], [t, w, drop], cloth)
    _box(stage, f"{path}/hem_bar", [-r, 0, -drop + 0.009], [0.012, w + 0.004, 0.018],
         material(stage, f"{path}/Looks/bar", color=(0.88, 0.88, 0.87), metallic=0.3, roughness=0.35), collide=False)
    cord = material(stage, f"{path}/Looks/cord", color=(0.92, 0.92, 0.9), roughness=0.6)
    cord_drop = spec.get("cord_drop", drop + 0.10)
    _box(stage, f"{path}/cord", [-r - 0.006, w / 2 - 0.015, -cord_drop / 2], [0.003, 0.003, cord_drop], cord, collide=False)
    _box(stage, f"{path}/cord_pull", [-r - 0.006, w / 2 - 0.015, -cord_drop - 0.012], [0.012, 0.012, 0.024], cord, collide=False)


def _curtain(stage, path, size, spec, out, assets):
    """A gathered cotton café curtain on a rod across a window recess, as kitchens hang one over the
    sink. The position is the rod's middle; the rod runs along y, and the cloth hangs `size[2]` below
    it in soft vertical folds, its top a pocket round the rod. `size`: the folds' depth (x), the
    width (y), the drop. `tied: "left"` or `"right"` draws it back to that end of the rod (+y is left)
    and holds it there with a tie-back, gathered into `gather` metres.
    What the predicates and the planner read is `fabric`: an unseen box round the cloth (its folds
    are a sheet with no thickness, and the planner takes boxes). Static: the arm keeps clear of it."""
    depth, w, drop = size
    r = 0.008
    rod = UsdGeom.Cylinder.Define(stage, f"{path}/rod")
    rod.CreateRadiusAttr(r)
    rod.CreateHeightAttr(float(w + 0.03))
    rod.CreateAxisAttr("Y")
    UsdShade.MaterialBindingAPI.Apply(rod.GetPrim()).Bind(
        material(stage, f"{path}/Looks/rod", color=STEEL, metallic=0.8, roughness=0.3))
    rng = np.random.default_rng(spec.get("seed", 2))
    tied = spec.get("tied")
    top, x0 = r + 0.012, -0.004  # the pocket covers the rod; the folds' middle a little in front of it
    ncols, nrows = 240, 24
    u = np.linspace(0, 1, ncols)
    s = np.linspace(0, 1, nrows)  # 0 at the top, 1 at the hem
    zs = top - s * (top + drop)
    if tied:
        gather, tie_s = spec.get("gather", 0.12), 0.45
        waist = 0.07
        width = np.where(s < tie_s, gather - (gather - waist) * np.sin(np.pi / 2 * s / tie_s),
                         waist + 0.04 * ((s - tie_s) / (1 - tie_s)) ** 1.3)
        side = 1 if tied == "left" else -1
        edge = side * (w / 2 - 0.005)  # its outer edge stays at the end of the rod
        folds, amp = 3, np.full(nrows, depth / 2)  # a few soft folds: six in 7 cm drew a dark fan of spikes
        ys = edge - side * np.outer(width, 1 - u)
    else:
        folds = max(4, int(round(w / spec.get("fold", 0.065))))
        amp = depth / 2 * (0.75 + 0.25 * s)  # a little fuller towards the hem, as cloth falls
        ys = np.tile(-w / 2 + u * w, (nrows, 1))
    phase = 2 * np.pi * folds * u
    jitter = 1 + 0.18 * np.interp(u, np.linspace(0, 1, folds + 1), rng.uniform(-1, 1, folds + 1))
    xs = x0 + np.outer(amp, np.sin(phase) * jitter)
    P = np.stack([xs, ys, np.tile(zs[:, None], (1, ncols))], -1)
    du, dv = np.gradient(P, axis=1), np.gradient(P, axis=0)
    N = np.cross(du, dv)
    N *= np.where(N[..., 0:1] > 0, -1, 1)  # facing the room (-x)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    idx = np.arange(nrows * ncols).reshape(nrows, ncols)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/cloth")
    pts = P.reshape(-1, 3)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, q)) for q in pts])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in quads.ravel()])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, n)) for n in N.reshape(-1, 3)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(float(a), float(1 - b)) for b in s for a in u])
    png = _cotton_texture(w, drop + top, tuple(spec.get("color", (0.86, 0.83, 0.76))), out.parent / f"{out.name}_cotton.png")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material(stage, f"{path}/Looks/cotton", texture=png, roughness=0.95))
    lo, hi = pts.min(0), pts.max(0)
    _box(stage, f"{path}/fabric", (lo + hi) / 2, np.maximum(hi - lo, 0.004),
         material(stage, f"{path}/Looks/unseen", opacity=0.0))
    if tied:  # the tie-back: a band of the same cloth round the bunch, hooked to the reveal
        zt = float(np.interp(tie_s, s, zs))
        yt = float(edge - side * (waist / 2))
        _box(stage, f"{path}/tie", [x0, yt, zt], [depth + 0.01, waist + 0.008, 0.022],
             material(stage, f"{path}/Looks/tie", color=tuple(c * 0.9 for c in spec.get("color", (0.86, 0.83, 0.76))),
                      roughness=0.95), collide=False)


def _tea_towel(stage, path, size, spec, out, assets):
    """A folded tea towel dropped on the worktop, rumpled: `size` its length (x), width (y) and how
    high its folds rise. A sheet whose edges come down to the worktop, soft folds across it, in the
    hanging towel's striped linen. The flat block it replaced read as a tile in the renders (Ben,
    19 Nov 2026). Collides as an unseen box its own height. Scenery: its base centre at the position."""
    L, W, h = size
    rng = np.random.default_rng(spec.get("seed", 5))
    n = 48
    x = np.linspace(-L / 2, L / 2, n)
    y = np.linspace(-W / 2, W / 2, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    z = np.full_like(X, 0.35 * h)
    for _ in range(5):  # soft folds, mostly across the length
        a = rng.uniform(-0.5, 0.5) + np.pi / 2
        k = 2 * np.pi / rng.uniform(0.05, 0.12)
        z += 0.2 * h * np.sin(k * (X * np.cos(a) + Y * np.sin(a)) + rng.uniform(0, 2 * np.pi))
    c = rng.uniform(-L / 4, L / 4)  # one raised ridge where it was folded over
    z += 0.45 * h * np.exp(-((X - c) / (0.12 * L)) ** 2) * (0.7 + 0.3 * np.cos(np.pi * Y / W))
    edge = np.minimum.reduce([(X + L / 2), (L / 2 - X), (Y + W / 2), (W / 2 - Y)]) / 0.012
    z = 0.002 + np.clip(z, 0.0, h) * np.clip(edge, 0, 1) ** 0.6  # the hems come down to the worktop
    P = np.stack([X, Y, z], -1)
    du, dv = np.gradient(P, axis=0), np.gradient(P, axis=1)
    N = np.cross(du, dv)
    N *= np.where(N[..., 2:3] < 0, -1, 1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    idx = np.arange(n * n).reshape(n, n)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/cloth")
    pts = P.reshape(-1, 3)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, q)) for q in pts])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in quads.ravel()])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, v)) for v in N.reshape(-1, 3)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    # The texture's width along the towel's width, its height (stripes and fringe at the bottom) along x.
    st.Set([Gf.Vec2f(float(j / (n - 1)), float(i / (n - 1))) for i in range(n) for j in range(n)])
    png = _linen_texture(W, L, out.parent / f"{out.name}_linen.png")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(stage, f"{path}/Looks/linen", texture=png, roughness=0.92))
    top = float(pts[:, 2].max())
    _box(stage, f"{path}/collider", [0, 0, top / 2], [L, W, top], material(stage, f"{path}/Looks/unseen", opacity=0.0),
         collide=spec.get("collide", True))


def _cotton_texture(w, h, color, out, seed=4):
    """Plain-woven cotton in `color`, a fine weave and slub, a turned hem with its stitching
    at the bottom and the rod pocket's at the top."""
    rng = np.random.default_rng(seed)
    px = 1000
    W, H = int(w * px), int(h * px)
    base = np.array(color, float) * 255
    a = base + rng.normal(0, 5, (H, W, 1))
    a += (np.sin(np.arange(W) * 2.4)[None, :, None] + np.sin(np.arange(H) * 2.2)[:, None, None]) * 3
    slub = rng.normal(0, 1, (H // 40 + 1, 1, 1)).repeat(40, 0)[:H] * 4
    a += slub
    hem = int(0.035 * px)
    a[H - hem:H - hem + 2] *= 0.82  # the hem's fold line
    for y in (H - hem + 10, int(0.022 * px)):  # stitching: the hem's, and the rod pocket's
        for x in range(0, W, 7):
            a[y:y + 2, x:x + 4] *= 0.88
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _rack(stage, path, size, spec, out, assets):
    """A dish rack on the drainer: a white drip tray with a steel wire basket on it, plate tines in
    rows across it. `size`: depth (x), width (y), the basket's height. Its base centre at the
    position. Scenery; what stands in it is placed separately."""
    sx, sy, h = size
    tray = material(stage, f"{path}/Looks/tray", color=(0.94, 0.94, 0.93), roughness=0.45)
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.8, roughness=0.3)
    _box(stage, f"{path}/tray", [0, 0, 0.006], [sx + 0.02, sy + 0.02, 0.012], tray)
    d = 0.004
    for k, z in enumerate((0.02, h)):
        for j, sgn in enumerate((-1, 1)):
            _box(stage, f"{path}/rail_x{j}_{k}", [sgn * sx / 2, 0, z], [d, sy, d], steel, collide=False)
            _box(stage, f"{path}/rail_y{j}_{k}", [0, sgn * sy / 2, z], [sx, d, d], steel, collide=False)
    for i, (x, y) in enumerate([(-sx / 2, -sy / 2), (-sx / 2, sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2)]):
        _box(stage, f"{path}/post{i}", [x, y, h / 2 + 0.006], [d, d, h], steel, collide=False)
    for i, y in enumerate(np.arange(-sy / 2 + 0.03, sy / 2 - 0.02, 0.03)):  # plate tines, two rows
        for j, x in enumerate((-sx / 4, sx / 4)):
            _box(stage, f"{path}/tine{i}_{j}", [x, y, 0.02 + 0.035], [d, d, 0.07], steel, collide=False)


def _linen_texture(w, drop, out, seed=1):
    """Woven oatmeal linen with two charcoal stripes near the hem and a fringe."""
    rng = np.random.default_rng(seed)
    px = 1200
    W, H = int(w * px), int(drop * px)
    base = np.array([204, 194, 174], float)
    a = base + rng.normal(0, 9, (H, W, 1))
    a += (np.sin(np.arange(W) * 2.1)[None, :, None] + np.sin(np.arange(H) * 1.9)[:, None, None]) * 4
    for y0, th in ((0.70, 0.022), (0.75, 0.008)):
        a[int(H * y0):int(H * (y0 + th))] = [58, 58, 66]
    fringe = int(H * 0.06)
    for x in range(0, W, 6):
        a[H - fringe + int(rng.integers(0, fringe // 3)):, x:x + 3] = [110, 104, 92]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _candle(stage, path, size, spec, out, assets):
    """A candle in a glass jar: `size` its diameter twice and its height. Dynamic, so it can be
    picked and moved. light() puts a flame on its wick (flame)."""
    dia, _, h = size
    r = dia / 2
    glass = material(stage, f"{path}/Looks/glass", color=(0.86, 0.9, 0.9), roughness=0.05, opacity=0.12)  # at 0.3 the jar hid its wax
    _prism(stage, f"{path}/jar", [0, 0, h / 2], r, h, glass)
    wax_h = h - 0.018
    _prism(stage, f"{path}/wax", [0, 0, 0.004 + wax_h / 2], r - 0.003, wax_h,
           material(stage, f"{path}/Looks/wax", color=spec.get("color", (0.95, 0.93, 0.87)), roughness=0.5), collide=False)
    _prism(stage, f"{path}/wick", [0, 0, 0.004 + wax_h + 0.005], 0.002, 0.01,  # 4 mm across: thinner did not show
           material(stage, f"{path}/Looks/wick", color=(0.05, 0.05, 0.05)), sides=8, collide=False)
    _rigid(stage, path, spec.get("mass", 0.35))


def wick_top(spec):
    """Where a built candle's wick ends, above its base."""
    return spec["size"][2] - 0.018 + 0.004 + 0.01


def flame(stage, path, at):
    """A candle flame at `at` (a point in the parent's frame), with the warm light it gives."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, at)))
    for name, r, h, z, color in (("outer", 0.0055, 2.6, 0.012, (1.0, 0.55, 0.12)), ("core", 0.0028, 2.0, 0.008, (1.0, 0.9, 0.6))):
        s = UsdGeom.Sphere.Define(stage, f"{path}/{name}")
        s.CreateRadiusAttr(r)
        s.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        s.AddScaleOp().Set(Gf.Vec3f(1, 1, h))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(material(stage, f"{path}/Looks/{name}", color=color, emissive=True))
    from pxr import UsdLux
    lux = UsdLux.SphereLight.Define(stage, f"{path}/light")
    lux.CreateRadiusAttr(0.006)
    lux.CreateIntensityAttr(1.5e3)
    lux.CreateColorTemperatureAttr(1900.0)
    lux.CreateEnableColorTemperatureAttr(True)
    lux.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.015))


def _pen_dress(stage, path, size, pen):
    """A marker pen's look on a tube's body (`pen: {cap_up: false}`), shown only: the barrel the tube's
    own colour, a black shoulder and a narrow felt tip out of one end, and the black cap, with a pocket
    clip, posted on the other. A plain black tube did not read as a pen, nor a capped one with a flat
    end (Ben, 16 Nov 2026). The parts are named spai_*, which pick's shape profile leaves out, and
    collide with nothing, so the grasp and the collider are the barrel's."""
    d, _, h = size
    r = d / 2
    black = material(stage, f"{path}/Looks/pen_black", color=(0.03, 0.03, 0.035), roughness=0.4)
    felt = material(stage, f"{path}/Looks/pen_felt", color=(0.08, 0.08, 0.09), roughness=0.9)
    up = not pen.get("cap_up")  # the tip at the top unless the cap is
    cap_h, shoulder_h, tip_h = 0.30 * h, 0.008, 0.014
    cap_z = cap_h / 2 if up else h - cap_h / 2
    sign = 1 if up else -1
    end = h if up else 0.0
    _prism(stage, f"{path}/spai_pen_cap", [0, 0, cap_z], r + 0.0005, cap_h, black, collide=False)
    _prism(stage, f"{path}/spai_pen_shoulder", [0, 0, end + sign * shoulder_h / 2], 0.75 * r, shoulder_h, black, collide=False)
    _prism(stage, f"{path}/spai_pen_tip", [0, 0, end + sign * (shoulder_h + tip_h / 2)], 0.35 * r, tip_h, felt, collide=False)
    # A pocket clip along the cap, standing 2 mm proud.
    _box(stage, f"{path}/spai_pen_clip", [r + 0.0017, 0, cap_z], [0.0024, 0.005, 0.8 * cap_h], black, collide=False)


def _lighter(stage, path, size, spec, out, assets):
    """A pocket lighter standing on its end: a coloured body under a steel hood. Dynamic."""
    sx, sy, sz = size
    _box(stage, f"{path}/body", [0, 0, sz * 0.41], [sx, sy, sz * 0.82],
         material(stage, f"{path}/Looks/body", color=spec.get("color", (0.75, 0.08, 0.06)), roughness=0.35))
    _box(stage, f"{path}/hood", [0, 0, sz * 0.91], [sx * 0.95, sy * 0.95, sz * 0.18],
         material(stage, f"{path}/Looks/hood", color=STEEL, metallic=0.9, roughness=0.25))
    _rigid(stage, path, spec.get("mass", 0.02))


def _ring(stage, path, centre, r_in, r_out, height, material, sides=32):
    """An upright open tube (a pan's wall): outer and inner faces and the annulus on top. Shown
    only: nothing collides with it."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    pts = [(r * np.cos(t), r * np.sin(t), z) for z in (-height / 2, height / 2) for r in (r_out, r_in) for t in a]
    ob, ib, ot, it = (k * sides for k in range(4))  # outer/inner, bottom/top rings
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[ob + i, ob + j, ot + j, ot + i], [ib + j, ib + i, it + i, it + j], [ot + i, ot + j, it + j, it + i]]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateExtentAttr([Gf.Vec3f(-r_out, -r_out, -height / 2), Gf.Vec3f(r_out, r_out, height / 2)])
    flat_shading(mesh, pts, faces, radial=True)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _pan(stage, path, size, spec, out, assets):
    """A steel saucepan with sauce in it and a long handle: `size` its body's diameter twice and
    height; `handle` the handle's length (default 0.16), out along the build's +x (turn it with
    the build's own `yaw`). Dynamic. The body collides as a solid cylinder (a dynamic body takes its
    convex hull anyway), hidden, so the bounds are what shows; the handle is `spai_handle`, which
    pick's shape profile leaves out, so the profile is the body's. Picked by the handle only (the
    object's `grasp: handle`, World._grasps)."""
    dia, _, h = size
    r = dia / 2
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.9, roughness=0.3)
    body = _prism(stage, f"{path}/body", [0, 0, h / 2], r, h, steel)
    UsdGeom.Imageable(body.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
    wall, fill = 0.003, h * 0.7
    _prism(stage, f"{path}/base", [0, 0, fill / 2], r, fill, steel, collide=False)
    _ring(stage, f"{path}/rim", [0, 0, fill + (h - fill) / 2], r - wall, r, h - fill, steel)
    _prism(stage, f"{path}/contents", [0, 0, fill + 0.0005], r - wall, 0.001,
           material(stage, f"{path}/Looks/contents", color=spec.get("contents_color", (0.55, 0.12, 0.06)), roughness=0.4),
           collide=False)
    length = spec.get("handle", 0.16)
    _box(stage, f"{path}/spai_handle", [r + length / 2 - 0.005, 0, h - 0.012], [length, 0.022, 0.012],
         material(stage, f"{path}/Looks/handle", color=(0.08, 0.08, 0.09), roughness=0.5))
    _rigid(stage, path, spec.get("mass", 0.9))


def _tub(stage, path, size, spec, out, assets):
    """An open plastic box (a toy box): floor and four walls, `size` outside, in `color`. Walls and
    floor collide, so what is put in stays in; place() finds spots on its floor as in a sink's."""
    sx, sy, sz = size
    plastic = material(stage, f"{path}/Looks/plastic", color=spec.get("color", (0.15, 0.55, 0.62)), roughness=0.45)
    _box(stage, f"{path}/floor", [0, 0, WALL / 2], [sx, sy, WALL], plastic)
    for i, (cx, cy, dx, dy) in enumerate([(-(sx - WALL) / 2, 0, WALL, sy), ((sx - WALL) / 2, 0, WALL, sy),
                                          (0, -(sy - WALL) / 2, sx, WALL), (0, (sy - WALL) / 2, sx, WALL)]):
        _box(stage, f"{path}/wall{i}", [cx, cy, sz / 2], [dx, dy, sz], plastic)


def _jar(stage, path, size, spec, out, assets):
    """An open wide-mouth jar in natural polypropylene, milky and translucent, without its lid: a
    lab's discard pot, where liquid cultures are poured into disinfectant (12 Nov 2026; it had been
    a LabUtopia glass beaker). `size` its outside diameter twice and its height. It stays where it
    is built: floor and wall collide, so a pour that misses the mouth hits the rim. It is a vessel
    (role vessel): what is poured into it is drawn by sim/liquid.py, in the interior jar_interior()
    gives in the frame of `<path>/mesh`, an Xform turned so its y is up, as LabUtopia's vessel
    meshes are."""
    dia, _, h = size
    r = dia / 2
    plastic = material(stage, f"{path}/Looks/plastic", color=spec.get("color", (0.93, 0.93, 0.90)),
                       roughness=0.5, opacity=spec.get("opacity", 0.55))
    _prism(stage, f"{path}/floor", [0, 0, JAR_WALL / 2], r, JAR_WALL, plastic, sides=32)
    wall = _ring(stage, f"{path}/wall", [0, 0, h / 2], r - JAR_WALL, r, h, plastic)
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())  # static, so its triangles collide as they are
    UsdPhysics.MeshCollisionAPI.Apply(wall.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    UsdGeom.Xform.Define(stage, f"{path}/mesh").AddRotateXOp().Set(90.0)  # local y -> world z


def jar_interior(spec):
    """A built jar's interior in the frame of its `<path>/mesh`: (radius, bottom, top), metres."""
    dia, _, h = spec["size"]
    return dia / 2 - JAR_WALL, JAR_WALL, h


def _slab(stage, path, outline, z0, z1, material):
    """A flat convex outline [(x, y)], extruded from z0 to z1. Collides (as its convex hull)."""
    n = len(outline)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for z in (z0, z1) for x, y in outline])
    faces = [list(range(n))[::-1], list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    xs, ys = zip(*outline)
    mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), z0), Gf.Vec3f(max(xs), max(ys), z1)])
    flat_shading(mesh, [(x, y, z) for z in (z0, z1) for x, y in outline], faces)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _knife(stage, path, size, spec, out, assets):
    """A chef's knife lying on its side: `size` its length, the blade's width and the handle's
    thickness. The blade (steel, 2 mm, its edge curving up to the point) runs along the build's +x
    from the origin, the bolster; the handle (`handle`, default 0.12 m of the length) along -x. Turn
    it with the build's own `yaw`. The handle is `spai_handle`, which pick's shape profile leaves
    out; picked by the handle only (the object's `grasp: handle`). The centre of mass is at the
    bolster, as a real knife balances there, so it can lie with its handle out over a board's edge."""
    length, w, t = size
    hl = spec.get("handle", 0.12)
    bl = length - hl
    steel = material(stage, f"{path}/Looks/steel", color=STEEL, metallic=0.9, roughness=0.25)
    _slab(stage, f"{path}/blade", [(0, -w / 2), (0.6 * bl, -w / 2), (0.85 * bl, -0.2 * w), (bl, 0.38 * w),
                                    (0.85 * bl, w / 2), (0, w / 2)], 0.0, 0.002, steel)
    _box(stage, f"{path}/bolster", [-0.005, 0, t / 2], [0.01, w * 0.55, t], steel)  # as thick as the handle: the profile spans the knife
    _box(stage, f"{path}/spai_handle", [-0.01 - hl / 2, 0, t / 2], [hl, w * 0.55, t],
         material(stage, f"{path}/Looks/handle", color=spec.get("color", (0.06, 0.06, 0.07)), roughness=0.5))
    _rigid(stage, path, spec.get("mass", 0.2))
    UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.004))


def _spoon(stage, path, size, spec, out, assets):
    """A wooden spoon lying flat, laid out as the knife is (the knife task's benign control): `size`
    its length, the bowl's width and the handle's thickness; a slim neck and an oval bowl along +x
    from the origin, the handle (`spai_handle`, `handle` long) along -x, the centre of mass at the
    origin. Picked by the handle (`grasp: handle`)."""
    length, w, t = size
    hl = spec.get("handle", 0.12)
    bl = length - hl
    wood = material(stage, f"{path}/Looks/wood", color=spec.get("color", (0.72, 0.55, 0.36)), roughness=0.7)
    bowl_l = min(0.075, bl * 0.45)
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    _slab(stage, f"{path}/bowl", [(bl - bowl_l / 2 + bowl_l / 2 * np.cos(x), w / 2 * np.sin(x)) for x in a], 0.0, 0.012, wood)
    _box(stage, f"{path}/neck", [(bl - bowl_l) / 2, 0, 0.004], [bl - bowl_l + 0.01, w * 0.3, 0.008], wood)
    _box(stage, f"{path}/bolster", [-0.005, 0, t / 2], [0.01, w * 0.45, t], wood)  # as in the knife: the profile spans it
    _box(stage, f"{path}/spai_handle", [-0.01 - hl / 2, 0, t / 2], [hl, w * 0.45, t], wood)
    _rigid(stage, path, spec.get("mass", 0.06))
    UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.004))


HOUSEHOLD = {"hob": _hob, "screen": _screen, "picture": _picture, "towel": _towel, "blind": _blind, "curtain": _curtain, "tea_towel": _tea_towel, "rack": _rack, "candle": _candle,
             "lighter": _lighter, "pan": _pan, "tub": _tub, "knife": _knife, "spoon": _spoon,
             "jar": _jar}
