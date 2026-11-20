"""A task's scene, live in Isaac Sim's full window, streamed over WebRTC: for trying skills by hand.

    python sim/live.py tasks/lab/lab_02_bleach_acid/task.json

Connect the Isaac Sim WebRTC Streaming Client to the instance's public IP (the script prints it).
The instance must expose TCP 49100 and UDP 47998 to your IP (see sim/README.md).

In the streamed window, open Window > Script Editor and queue calls on `spai`. They run in this
script's loop, one after the other, and the stream shows the arm move:

    spai.do("pick", "bottle_A")
    spai.do("pour", "bottle_A", "beaker_B")
    spai.do("place", "bottle_A", "bench")
    spai.do("go_to", [0.28, 0.0, 1.0], [0, 90, 25])
    spai.bench                                   # the Bench, for anything else (read-only use)

Each result goes to the terminal running this script, to /tmp/spai-live/log.txt and to `spai.log`.
Physics keeps running between calls, but substance tags only move during one. To start over, stop
the script and start it again: rigid bodies cannot simply be put back.
"""
import argparse
import builtins
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("task", help="task.json with a _base_config for LabUtopia")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--public-ip", help="default: asked of ifconfig.me")
args = parser.parse_args()
args.task = str(Path(args.task).resolve())
ip = args.public_ip or subprocess.run(["curl", "-s", "ifconfig.me"], capture_output=True, text=True).stdout.strip()

from isaacsim import SimulationApp  # noqa: E402
from common import KIT_ARGS  # noqa: E402  (no Isaac imports)

# As Isaac Sim's own livestream example: headless, but with the UI, which the stream carries.
app = SimulationApp({"headless": True, "hide_ui": False, "width": 1280, "height": 720,
                     "window_width": 1920, "window_height": 1080,
                     "extra_args": [*KIT_ARGS,
                                    f"--/app/livestream/publicEndpointAddress={ip}",
                                    "--/app/livestream/port=49100"]})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.kit.livestream.webrtc")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from common import load_task  # noqa: E402
from labels import cards, styles, write_labels  # noqa: E402
from world import Bench  # noqa: E402


OUT = Path("/tmp/spai-live")
OUT.mkdir(parents=True, exist_ok=True)


def say(line):
    """Kit takes over print() once the app runs; write past it, and to a file."""
    sys.__stdout__.write(line + "\n")
    sys.__stdout__.flush()
    with open(OUT / "log.txt", "a") as f:
        f.write(line + "\n")


class Live:
    """The queue the Script Editor writes to and the main loop drains."""

    def __init__(self, bench):
        self.bench = bench
        self.queue = deque()
        self.log = []

    def do(self, method, *a, **kw):
        self.queue.append((method, a, kw))
        return f"queued {method}{a}{kw or ''}"

    def run_next(self):
        method, a, kw = self.queue.popleft()
        try:
            result = getattr(self.bench, method)(*a, **kw)
        except Exception:
            result = traceback.format_exc()
        positions = {n: np.round(self.bench.center(n), 3).tolist() for n in self.bench.objects}
        line = f"{method}{a}{kw or ''} -> {result} | held {self.bench.held_object} | {positions}"
        self.log.append(line)
        say(line)


spec, cfg, pred = load_task(args.task)
bench = Bench(spec, cfg, pred, write_labels(cards(spec), OUT / "labels", styles(cfg)), args.seed)
builtins.spai = Live(bench)
say(f"LIVE: connect the WebRTC Streaming Client to {ip}. Script Editor: spai.do(\"pick\", \"<object>\")")

while app.is_running():
    if builtins.spai.queue:
        builtins.spai.run_next()
    else:
        app.update()
app.close()
