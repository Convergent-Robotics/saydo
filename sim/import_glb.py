"""Turn single-file glTF props (assets/home/*.glb) into USD for the household tasks. Run once on the
instance, after setup_brev.sh:

    python sim/import_glb.py assets/home/compressed_air_can.glb

Written to external/home_assets/<name>.usd as Isaac's asset converter leaves it (Y up, the model's
own units): fixtures.py fits it to the object it dresses (`look:` on a tube).
"""
import argparse
import asyncio
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("files", nargs="+", help="glb files")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()

import omni.kit.asset_converter as converter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "external" / "home_assets"


async def convert(src, dst):
    ctx = converter.AssetConverterContext()
    ctx.keep_all_materials = True
    ctx.use_meter_as_world_unit = True
    ctx.embed_textures = True
    task = converter.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    if not await task.wait_until_finished():
        raise RuntimeError(f"{src}: {task.get_status()} {task.get_error_message()}")


OUT.mkdir(parents=True, exist_ok=True)
loop = asyncio.get_event_loop()
for f in args.files:
    dst = OUT / (Path(f).stem + ".usd")
    fut = asyncio.ensure_future(convert(Path(f).resolve(), dst))
    while not fut.done():
        app.update()
        loop.run_until_complete(asyncio.sleep(0))
    fut.result()
    print(f"{f} -> {dst}", flush=True)
app.close()
