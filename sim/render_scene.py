"""Wide renders of a task's whole scene, robot included, in the episode's start pose: for
figures and for checking a scene by eye.

    python sim/render_scene.py tasks/lab/lab_02_bleach_acid/task.json --out runs/scene

Writes <out>/scene_<view>.png at 1920x1080.
"""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--out", default="runs/scene")
args = parser.parse_args()
args.task = str(Path(args.task).resolve())
args.out = Path(args.out).resolve()

from isaacsim import SimulationApp  # noqa: E402

from common import CPU_THREADS, KIT_ARGS  # noqa: E402  (no Isaac imports)

app = SimulationApp({"headless": True, "extra_args": KIT_ARGS, **CPU_THREADS})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import load_task  # noqa: E402
from labels import cards, styles, write_labels  # noqa: E402
from world import Bench, look_at  # noqa: E402

# Between the robot's base (x = -0.4) and the objects (x ~ 0.28), at bench height.
TARGET = [-0.05, 0.0, 0.95]
# name: (eye, focal length in Isaac's units)
VIEWS = {
    "three_quarter": ([1.05, -1.25, 1.75], 2.4),   # in front of the bench, to the robot's right
    "side": ([0.0, -1.9, 1.25], 2.4),              # square to the bench, from the robot's right
    "behind": ([-1.45, -0.9, 1.9], 2.4),           # behind and above the robot, as the operator stands
}


def main():
    from isaacsim.sensors.camera import Camera

    spec, cfg, pred = load_task(args.task)
    args.out.mkdir(parents=True, exist_ok=True)
    bench = Bench(spec, cfg, pred, write_labels(cards(spec), args.out / "labels", styles(cfg)), 0)
    cams = {}
    for name, (eye, focal) in VIEWS.items():
        cam = Camera(prim_path=f"/World/spai_scene_{name}", name=f"scene_{name}", resolution=(1920, 1080),
                     position=np.array(eye), orientation=look_at(eye, TARGET))
        cam.set_focal_length(focal)
        cam.set_clipping_range(near_distance=0.05, far_distance=20.0)
        cam.initialize()
        cams[name] = cam
    bench.settle(30, render=True)
    for name, cam in cams.items():
        path = args.out / f"scene_{name}.png"
        Image.fromarray(np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)).save(path)
        print(path)


try:
    main()
finally:
    app.close()
