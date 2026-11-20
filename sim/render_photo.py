"""Render a scene config's overview camera, once settled, to a PNG: the photo a household task's
monitor shows (tasks/home/home_mug_screen/photo). setup_brev.sh runs it.

    python sim/render_photo.py tasks/home/home_mug_screen/photo/task.json external/home_assets/mug_photo.png
"""
import sys
from pathlib import Path

task, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()

from isaacsim import SimulationApp  # noqa: E402

from common import KIT_ARGS  # noqa: E402  (no Isaac imports)

app = SimulationApp({"headless": True, "extra_args": KIT_ARGS})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

from common import load_task  # noqa: E402
from world import Bench  # noqa: E402

try:
    spec, cfg, pred = load_task(task)
    bench = Bench(spec, cfg, pred, {}, 0)
    bench.settle(150, render=True)  # materials and lights settle over the first frames
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bench.frame("overview")).save(out)
    print(out, flush=True)
except Exception:  # app.close() ends the process before Python would print it
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
