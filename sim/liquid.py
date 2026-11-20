"""Liquid as the cameras see it: static coloured bodies in the vessels, moved when a pour counts.

Nothing here is simulated. A vessel with `liquid:` in its config ({fill_height: m, color: [r, g,
b]}) holds a layer of particles, laid out in its interior by LabUtopia's calibrated sampler
(utils/liquid_particles.py, with the interiors it measured for lab_001's beaker mesh, as beaker1-3,
and conical_bottle02, or those measured here for the Laboratory.glb glassware, GLASSWARE) and
drawn as one smooth body round them. Drawn as the spheres themselves, it
read as powder or glass shards: models called it a solid (30 Oct 2026). The layer follows the vessel. An item (a capped sample bottle) may hold
liquid too: it is drawn and carried, never poured. When the vessel tips past the pour test
(Bench.pours_into: tipped more than POUR_TILT_DEG with its mouth over another container), its particles go at once to
that container: a new layer on top in a measured vessel, gone in anything else (a sink's drain).
Tipped over nothing, they lie as a puddle (a glossy film, not particles) on whatever is below the mouth. A vessel whose config
says `pour_fraction` (0-1) gives only that share of its contents at each pour, from the top, and
shows the rest once it stands upright again (an aliquot taken from a stock bottle). The predicates still
score by substance tags (sim/world.py); this only shows what they count. Physics does not change:
no collision, mass or particle simulation (GPU liquid made the episodes ~10x slower and a held
bottle unstable, 20 Oct). SPAI_LIQUID=0 leaves the particles out.
"""
import dataclasses
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from utils.liquid_metrics import BeakerInterior, ConicalInterior  # noqa: E402  (LabUtopia)
from utils.liquid_particles import sample_container  # noqa: E402

ENABLED = os.environ.get("SPAI_LIQUID", "1") != "0"
SPACING = 0.004  # particle spacing, metres (LabUtopia's calibrated 0.002-0.006)
ROOT = "/World/spai_liquid"
# A puddle as it looks on a bench: one smooth, wet film with a ragged edge, darker than the liquid
# in glass (a thin layer over a black bench) and glossy. Drawn as particles, as in a vessel, it read
# as a pale, grainy pad: a model called a spilled culture "the absorbent pad" (27 Oct 2026), and
# flattened particles still showed no gloss (render/lab27/puddle-probe/).
PUDDLE_SHADE = 0.1  # of the liquid's colour: render/lab27/puddle-probe2/, between B and C
PUDDLE_ROUGHNESS = 0.02
PUDDLE_EDGE = 0.15  # how ragged the edge is, as a share of the radius
PUDDLE_DEPTH_M = 0.001
FILL_STEP = 0.002  # how far up a receiving vessel is sampled at a time, metres
SETTLED_DEG = 15  # a vessel that gave a share (pour_fraction) shows the rest once back within this tilt
BODY_SEGMENTS = 48  # round a liquid body
BODY_ROUGHNESS = 0.08
BODY_OPACITY = 0.5
BODY_SHADE = 0.5  # of the liquid's colour: at full colour a pale liquid's top glared white, as a solid's would


def wanted(cfg):
    return ENABLED and any("liquid" in o for o in cfg["spai"]["objects"].values())


def scaled_conical(radial, axial):
    """LabUtopia's conical_bottle02 interior stretched by `radial` and `axial`: lab_001's
    conical_bottle04 is the same 1214-point mesh, about 1.525 x wider and 1.421 x taller. Without
    it the clean-down's flask (conical_bottle04 since 7 Nov) was drawn empty: its bleach, or the
    control's PBS, never showed in the frames, only a dark strip of the flask's own mesh (9 Nov
    2026)."""
    c = ConicalInterior()
    return dataclasses.replace(c, center_x=c.center_x * radial, center_z=c.center_z * radial, radius=c.radius * radial,
                               bottom=c.bottom * axial, top=c.top * axial,
                               lower_profile=tuple((y * axial, r * radial) for y, r in c.lower_profile))


# conical_bottle04 against conical_bottle02, from their meshes' extents (sim/ probe, 9 Nov 2026)
CONICAL04 = scaled_conical(705.678162 / 462.871399, 2338.275635 / 1645.240723)


def glass_interior(radius, bottom, top, lower_profile):
    """A Laboratory.glb vessel's interior (sim/import_glassware.py), in its mesh's own frame: Y up,
    axis at x = z = 0, the model's units. Its inside is the collider's inner wall: the model's
    surface less half the collider's wall, from the top of the collider's floor up."""
    return dataclasses.replace(BeakerInterior(), center_x=0.0, center_z=0.0, radius=radius, bottom=bottom,
                               top=top, lower_profile=lower_profile)


# The glassware sim/import_glassware.py makes from Laboratory.glb, by its mesh's point count and min
# corner, as its output prints them (18 Nov 2026). One interior serves every size made from a model:
# the size is in the mesh's transform, as lab_001's scales are.
GLASSWARE = {
    (180, (-0.576121, 0.0, -0.399298)): glass_interior(  # Beaker
        0.390208, 0.02, 1.047681, ((0.02, 0.266111), (0.047622, 0.30908), (0.105521, 0.341801),
                                   (0.155266, 0.354516), (0.874931, 0.354516), (0.969363, 0.354516),
                                   (1.047681, 0.390208))),
    (220, (-0.370051, 0.0, -0.370051)): glass_interior(  # Erlenmayer Flask
        0.360051, 0.02, 1.109957, ((0.02, 0.277731), (0.048355, 0.318777), (0.118688, 0.360051),
                                   (0.182428, 0.360051), (0.672539, 0.119911), (0.690151, 0.111282),
                                   (0.716665, 0.111282), (1.019993, 0.111282), (1.109956, 0.136473))),
    # Graduated Cylinder: from the top of its stem, not its foot's floor, as a real one's glass foot is
    # solid; the stem's bore below 0.424 narrows into the foot.
    (320, (-0.349151, 0.0, -0.349151)): glass_interior(
        0.17491, 0.232529, 2.039317, ((0.232529, 0.152688), (0.424303, 0.152688), (1.852932, 0.152688),
                                      (1.944223, 0.152688), (2.039317, 0.17491))),
}


def built_interior(spec):
    """A built jar's interior (sim/fixtures.py's _jar), as a LabUtopia interior: a plain cylinder
    from the floor up, in the jar's `mesh` frame (y up, metres). None for any other object."""
    build = spec.get("build") or {}
    if build.get("kind") != "jar":
        return None
    from fixtures import jar_interior
    r, bottom, top = jar_interior(build)
    return dataclasses.replace(BeakerInterior(), center_x=0.0, center_z=0.0, radius=r, bottom=bottom, top=top,
                               lower_profile=((bottom, r),))


def interior(mesh):
    """The calibrated interior of a lab_001 vessel mesh, recognised as LabUtopia's PourLiquid
    recognises it (point count and extent), or of a Laboratory.glb one (GLASSWARE), else None."""
    if not mesh.IsValid() or not mesh.IsA(UsdGeom.Mesh):
        return None
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
    if points.shape == (1698, 3) and np.allclose(points.min(0), [-33.630287, -86.688225, -75.332466], atol=0.002):
        return BeakerInterior()
    if points.shape == (1214, 3) and np.allclose(points.min(0), [-462.871399, 0.000095, -462.867584], atol=0.002):
        return ConicalInterior()
    if points.shape == (1214, 3) and np.allclose(points.min(0), [-705.678162, 0.000191, -705.67804], atol=0.002):
        return CONICAL04
    for (n, lo), it in GLASSWARE.items():
        if points.shape == (n, 3) and np.allclose(points.min(0), lo, atol=1e-4):
            return it
    return None


def world_matrix(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


class Liquid:
    def __init__(self, bench):
        self.bench = bench
        self.stage = bench.stage
        self.interiors = {}  # vessel -> calibrated interior, for every container that can take liquid
        # and every item that holds some (a capped bottle: shown full, never poured from)
        holders = bench.containers + [n for n, o in bench.spai["objects"].items()
                                      if "liquid" in o and n not in bench.containers]
        for name in holders:
            it = interior(self.stage.GetPrimAtPath(bench.objects[name] + "/mesh")) \
                or built_interior(bench.spai["objects"][name])
            if it is not None:
                self.interiors[name] = it
            elif "liquid" in bench.spai["objects"][name]:
                print(f"[liquid] {name}: no calibrated interior for its mesh, drawn empty", flush=True)
        self.colors = []  # one sphere prototype per colour, shared by all layers
        self.layers = {}  # vessel -> [(colour index, particle count)], bottom first
        self.upright = {}  # vessel -> its orientation when it was filled, for the tilt test
        self._made = set()
        self._part_poured = set()  # vessels that gave a share of their contents this tip (pour_fraction)
        self.puddles = {}  # vessel -> {"center": [x, y], "radius": m}, its latest spill
        self.spills_drawn = []  # every puddle's prim, in order
        for name, o in bench.spai["objects"].items():
            if "liquid" in o and name in self.interiors:
                color = self._color(o["liquid"].get("color", (0.6, 0.8, 0.95)))
                n = len(self._sample(name, float(o["liquid"].get("fill_height", 0.03))))
                self.layers[name] = [(color, n)]
                self._draw(name)

    def _color(self, rgb):
        rgb = tuple(float(c) for c in rgb)
        if rgb not in self.colors:
            self.colors.append(rgb)
        return self.colors.index(rgb)

    def _mesh(self, name):
        return self.stage.GetPrimAtPath(self.bench.objects[name] + "/mesh")

    def _sample(self, name, fill_height):
        return sample_container(self.interiors[name], world_matrix(self._mesh(name)), spacing=SPACING,
                                fill_height=fill_height)

    def _draw(self, name):
        """(Re)build the vessel's liquid from its layers, in the mesh's own frame: the lowest points
        of its interior, filled from the bottom until there are as many as its layers hold, and one
        smooth body round each layer's points."""
        total = sum(n for _, n in self.layers[name])
        path = f"{ROOT}/{name}"
        if self.stage.GetPrimAtPath(path).IsValid():
            self.stage.RemovePrim(path)
        self._made.discard(name)
        if not total:
            return
        fill, points = FILL_STEP, np.zeros((0, 3))
        while len(points) < total and fill < 0.5:
            try:
                points = self._sample(name, fill)
            except ValueError:  # LabUtopia's sampler: no room for a particle this shallow
                pass
            fill += FILL_STEP
        points = points[np.argsort(points[:, 2])][:total]  # upright: lowest first
        m = world_matrix(self._mesh(name))
        inv = np.linalg.inv(m)
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTransformOp()
        # The vessel's own axis, from its interior: a layer's lowest level is often the part of a
        # level the layer below did not fill, all on one side, and the middle of its particles was
        # then off the axis. The body round that point bulged out through the glass (the buffer
        # task's beaker after the pour, 18 Nov 2026).
        it = self.interiors[name]
        axis = (np.array([it.center_x, 0.0, it.center_z, 1.0]) @ m)[:2]
        start = 0
        for i, (c, n) in enumerate(self.layers[name]):
            if len(points[start:start + n]):
                self._body(f"{path}/layer{i}", points[start:start + n], inv, self.colors[c], axis)
            start += n
        self._made.add(name)
        self.upright[name] = self.bench.quat(name)
        self.follow(name)

    def _body(self, path, points, inv, rgb, axis):
        """A closed body of revolution round `points` (world, upright) about `axis` (x, y): at each
        height the radius the particles reach there, half a spacing out to the glass; flat at the
        top. In the frame of `inv`, the world-to-mesh matrix."""
        r = np.linalg.norm(points[:, :2] - axis, axis=1) + SPACING / 2
        bottom, top = points[:, 2].min() - SPACING / 2, points[:, 2].max() + SPACING / 2
        levels = np.round(points[:, 2] / SPACING).astype(int)
        rings = [(points[levels == k, 2].mean(), r[levels == k].max()) for k in np.unique(levels)]
        rings = [(bottom, rings[0][1])] + rings + [(top, rings[-1][1])]
        a = np.linspace(0, 2 * np.pi, BODY_SEGMENTS, endpoint=False)
        around = np.column_stack([np.cos(a), np.sin(a), np.zeros_like(a)])
        verts, normals, counts, faces = [], [], [], []

        def ring(z, radius, normal):
            start = len(verts)
            verts.extend(np.array([axis[0], axis[1], z]) + radius * around)
            normals.extend(around if normal is None else [normal] * BODY_SEGMENTS)
            return start

        for k, (z, radius) in enumerate(rings):
            ring(z, radius, None)
            if k:
                for j in range(BODY_SEGMENTS):
                    a0, a1 = (k - 1) * BODY_SEGMENTS + j, (k - 1) * BODY_SEGMENTS + (j + 1) % BODY_SEGMENTS
                    counts.append(4)
                    faces.extend([a0, a1, a1 + BODY_SEGMENTS, a0 + BODY_SEGMENTS])
        for z, radius, up, wind in ((top, rings[-1][1], 1.0, 1), (bottom, rings[0][1], -1.0, -1)):
            first = ring(z, radius, np.array([0.0, 0.0, up]))
            counts.append(BODY_SEGMENTS)
            faces.extend([first + j for j in range(BODY_SEGMENTS)][::wind])
        local = np.asarray(verts) @ inv[:3, :3] + inv[3, :3]
        turned = np.asarray(normals) @ inv[:3, :3]
        turned /= np.linalg.norm(turned, axis=1, keepdims=True)
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(local, np.float32)))
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(turned, np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mat = UsdShade.Material.Define(self.stage, f"{path}/look")
        shader = UsdShade.Shader.Define(self.stage, f"{path}/look/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(x) * BODY_SHADE for x in rgb)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(BODY_ROUGHNESS)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(BODY_OPACITY)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    def follow(self, name=None):
        """Put each vessel's liquid where its mesh is now (it is in its frame)."""
        for n in [name] if name else list(self._made):
            prim = self.stage.GetPrimAtPath(f"{ROOT}/{n}")
            op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
            op.Set(Gf.Matrix4d(*world_matrix(self._mesh(n)).flatten().tolist()))

    def on_step(self):
        """After each physics step: a vessel tipped past the pour test pours, at once."""
        b = self.bench
        for v in [n for n, layers in self.layers.items() if layers and n in b.containers]:
            tilt = b.tilt(v, self.upright[v])
            if v in self._part_poured:  # one share per tip; upright again, the rest settles
                if tilt <= SETTLED_DEG:
                    self._part_poured.discard(v)
                    self._draw(v)
                continue
            if tilt <= b.pour_tilt_deg:
                continue
            into = [t for t in b.containers if t != v and b.above(v, t)]
            # A pour gives `pour_fraction`, and so does a pour that misses (held, tipped over no
            # vessel): the same share runs onto the bench. Knocked over, a vessel loses everything.
            share = self._share(v) if v == b.held_object or into else 1.0
            if into:
                self._pour(v, into[0], share)
            else:
                self._spill(v, share)

    def _share(self, v):
        return float(self.bench.spai["objects"][v].get("liquid", {}).get("pour_fraction", 1.0))

    def _take(self, v, share):
        """Take `share` of v's contents, from the top, and return it as layers. All of it: v is
        redrawn empty now. Part: the rest is redrawn once v stands upright again (on_step); drawn
        now, mid-tip, it would be laid out in the tipped vessel's low side."""
        if share >= 1.0:
            moved, self.layers[v] = self.layers[v], []
            self._draw(v)
            return moved
        moved, kept, take = [], [], round(sum(n for _, n in self.layers[v]) * share)
        for c, n in reversed(self.layers[v]):  # from the top layer down
            k = min(n, take)
            take -= k
            if k:
                moved.insert(0, (c, k))
            if n - k:
                kept.insert(0, (c, n - k))
        self.layers[v] = kept
        self._part_poured.add(v)
        return moved

    def _pour(self, v, t, share=1.0):
        """`share` of v's contents into t (a pour_fraction aliquot, or all of it)."""
        moved = self._take(v, share)
        if t in self.interiors:
            self.layers.setdefault(t, []).extend(moved)
            self._draw(t)

    def _spill(self, v, share=1.0):
        """`share` of v's contents as a puddle (film) on the surface below its mouth, as wide as
        those particles would cover one deep. Each spill is a puddle of its own: a second does not
        wipe the first."""
        from omni.physx import get_physx_scene_query_interface
        moved = self._take(v, share)
        n = sum(k for _, k in moved)
        if not n:
            return
        protos = np.concatenate([[c] * k for c, k in moved])
        # Down from the mouth to the first surface that is neither the vessel nor the robot
        # holding it (a held vessel's fingers are below its tipped mouth: a puddle was drawn on
        # them, in mid-air, 27 Oct 2026). From just under the vessel, as before, a vessel lying
        # on the bench cast from inside the bench top.
        c, low = self.bench.mouth(v), self.bench.bounds(v)[0][2]
        own, robot, hits = self.bench.objects[v], self.bench.robot.prim_path, []
        get_physx_scene_query_interface().raycast_all(
            [float(c[0]), float(c[1]), float(c[2]) + 0.01], [0.0, 0.0, -1.0], 1.5,
            lambda h: hits.append(h) or True)
        below = sorted((h.distance, h.position[2]) for h in hits
                       if not str(h.collision).startswith((own, robot)))
        z = below[0][1] if below else low
        r = np.sqrt(n * SPACING ** 2 / np.pi)  # as the particles would cover, one deep
        rgb = np.mean([self.colors[int(i)] for i in protos], axis=0)
        path = f"{ROOT}/puddle_{v}_{len(self.spills_drawn)}"
        self.spills_drawn.append(path)
        self.film(path, c[:2], r, z, rgb, seed=len(self.spills_drawn))
        self.puddles[v] = {"center": [float(c[0]), float(c[1])], "radius": float(r)}

    def film(self, path, center, r, z, rgb, seed=0, shade=PUDDLE_SHADE, roughness=PUDDLE_ROUGHNESS):
        """A puddle: a thin flat mesh of radius about r on the surface at height z, its edge ragged
        (a few low harmonics of the angle, from `seed`), in the liquid's colour times `shade`, glossy."""
        rng = np.random.default_rng(seed)
        a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        edge = np.ones_like(a)
        for k in (2, 3, 5, 7):
            edge += PUDDLE_EDGE / k * rng.uniform(-1, 1) * np.cos(k * a + rng.uniform(0, 2 * np.pi))
        ring = np.column_stack([center[0] + r * edge * np.cos(a), center[1] + r * edge * np.sin(a)])
        top = z + PUDDLE_DEPTH_M
        pts = [Gf.Vec3f(float(center[0]), float(center[1]), float(top))] + [Gf.Vec3f(float(x), float(y), float(top)) for x, y in ring]
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([3] * len(a))
        mesh.CreateFaceVertexIndicesAttr([i for j in range(len(a)) for i in (0, 1 + j, 1 + (j + 1) % len(a))])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(pts))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mat = UsdShade.Material.Define(self.stage, f"{path}/look")
        shader = UsdShade.Shader.Define(self.stage, f"{path}/look/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(x) * shade for x in rgb)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.01)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
