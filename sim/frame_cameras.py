"""Find camera views that show a task's labels: renders candidate overview cameras and candidate
arm poses for the wrist camera, and saves one contact sheet of each.

    python sim/frame_cameras.py tasks/lab/lab_02_bleach_acid/task.json --out runs/framing

Writes <out>/overview_sheet.png, <out>/wrist_sheet.png (each tile captioned with its candidate),
the single frames, and <out>/candidates.json. Copy the chosen overview camera into the task's
labutopia-config.yaml (cameras), the chosen arm pose into spai.look_pose and the wrist focal
length into the wrist camera's focal_length.

Arm poses are given as LabUtopia's controllers give them: an end-effector position and
xyz Euler angles in degrees, passed through scipy's as_quat() to the RMPflow controller. LabUtopia's
pick and place orientations are written this way, so they and these mean the same thing.
"""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--out", default="runs/framing")
args = parser.parse_args()
args.task = str(Path(args.task).resolve())
args.out = Path(args.out).resolve()

from isaacsim import SimulationApp  # noqa: E402

from common import CPU_THREADS, KIT_ARGS  # noqa: E402  (no Isaac imports)

app = SimulationApp({"headless": True, "extra_args": KIT_ARGS, **CPU_THREADS})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from common import load_task  # noqa: E402
from labels import cards, font, styles, write_labels  # noqa: E402
from world import Bench, look_at  # noqa: E402

BENCH_CENTRE = [0.28, 0.0, 0.85]

# (eye, target, focal length in Isaac's units). The robot stands at x = -0.4 facing +x; the labels
# face it, so cameras behind it and to one side see them.
OVERVIEWS = [
    ([-0.60, -0.55, 1.45], BENCH_CENTRE, 5.0),
    ([-0.65, -0.45, 1.30], BENCH_CENTRE, 5.0),
    ([-0.60, -0.55, 1.45], BENCH_CENTRE, 4.5),
    ([-0.60, 0.55, 1.45], BENCH_CENTRE, 5.0),
]

# (end-effector position, xyz Euler degrees in LabUtopia's convention). [0, 90, 0] points the
# wrist camera forward along the bench; more pitch tilts it down.
ARM_POSES = [(pos, eul) for pos in ([-0.05, 0.0, 1.10], [-0.10, 0.0, 1.10])
             for eul in ([0, 100, 0], [0, 105, 0])]
WRIST_FOCALS = [2.0, 2.5]  # LabUtopia's is 1.0: very wide


def sheet(tiles, path, cols=4, size=384):
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * size, rows * (size + 40)), "white")
    draw = ImageDraw.Draw(out)
    f = font(16)
    for i, (img, caption) in enumerate(tiles):
        x, y = (i % cols) * size, (i // cols) * (size + 40)
        out.paste(Image.fromarray(img).resize((size, size)), (x, y + 40))
        draw.text((x + 6, y + 4), caption, font=f, fill="black")
    out.save(path)


def main():
    from isaacsim.sensors.camera import Camera

    spec, cfg, pred = load_task(args.task)
    args.out.mkdir(parents=True, exist_ok=True)
    bench = Bench(spec, cfg, pred, write_labels(cards(spec), args.out / "labels", styles(cfg)), 0)
    record = {"overview": [], "arm": [], "labels": {}}

    # Where each label card was meant to go, and where it is after the scene settled.
    from pxr import Usd, UsdGeom
    for name, (body, want) in bench.label_targets.items():
        card = bench.stage.GetPrimAtPath(body + "/spai_label")
        m = UsdGeom.Xformable(card).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        got = [list(map(float, m.Transform(p))) for p in UsdGeom.Mesh(card).GetPointsAttr().Get()]
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            bench.stage.GetPrimAtPath(bench.objects[name])).ComputeAlignedRange()
        record["labels"][name] = {"body": body, "want": want, "got": got,
                                  "object_box": [list(box.GetMin()), list(box.GetMax())]}
        print(f"{name}: body {body}\n  want {np.round(want, 3).tolist()}\n  got  {np.round(got, 3).tolist()}"
              f"\n  box  {np.round(list(box.GetMin()), 3).tolist()} .. {np.round(list(box.GetMax()), 3).tolist()}")

    cams = []
    for i, (eye, target, focal) in enumerate(OVERVIEWS):
        cam = Camera(prim_path=f"/World/spai_probe_{i}", name=f"probe_{i}", resolution=(1024, 1024),
                     position=np.array(eye), orientation=look_at(eye, target))
        cam.set_focal_length(focal)
        cam.set_clipping_range(near_distance=0.05, far_distance=10.0)
        cam.initialize()
        cams.append(cam)
    bench.settle(30, render=True)
    tiles = []
    for i, (cam, (eye, target, focal)) in enumerate(zip(cams, OVERVIEWS)):
        img = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
        Image.fromarray(img).save(args.out / f"overview_{i}.png")
        tiles.append((img, f"{i}: eye {eye} f {focal}"))
        record["overview"].append({"i": i, "eye": eye, "look_at": target, "focal_length": focal})
    sheet(tiles, args.out / "overview_sheet.png")

    tiles = []
    for i, (pos, eul) in enumerate(ARM_POSES):
        bench.go_to(pos, eul)
        reached = bench.robot.get_gripper_position()
        for focal in WRIST_FOCALS:
            bench.cameras["wrist"].set_focal_length(focal)
            bench.settle(5, render=True)
            img = bench.frame("wrist")
            Image.fromarray(img).save(args.out / f"wrist_{i}_f{focal}.png")
            tiles.append((img, f"{i}: {pos} {eul} f {focal}"))
        record["arm"].append({"i": i, "position": pos, "euler_deg": eul, "reached": [float(v) for v in reached]})
    sheet(tiles, args.out / "wrist_sheet.png")

    (args.out / "candidates.json").write_text(json.dumps(record, indent=1))
    print("sheets:", args.out / "overview_sheet.png", args.out / "wrist_sheet.png")


try:
    main()
finally:
    app.close()
