"""Mirror the Isaac asset-library files the tasks use ("isaac:<path>" in a scene's scenery or
objects, or a control's _config_overrides) into external/isaac_assets/, with every layer and
texture each one depends on, at the same paths under the library's root. World._reference then
reads the local copy and nothing is streamed from S3 during an episode.

Why: the home scenes stream Isaac props over the network. A local read is quick and takes nothing
from the other process's network share. It was written against the home episodes' carb crash, which
it does not address (sim/README.md, "Running several at once"). Run once on the instance, and again
after a task adds an isaac: asset (one not mirrored is still streamed, with a warning):

    python sim/mirror_isaac.py            # every task under tasks/
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdUtils  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "external" / "isaac_assets"


def used():
    """Every isaac: path in the tasks' scenes and task specs."""
    paths = set()
    for f in [*ROOT.glob("tasks/*/*/*.yaml"), *ROOT.glob("tasks/*/*/task.json")]:
        paths |= set(re.findall(r"isaac:([^\"'\s,}]+)", f.read_text()))
    return sorted(p.lstrip("/") for p in paths)


def main():
    root = get_assets_root_path().rstrip("/")
    for rel in used():
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(f"{root}/{rel}")
        files = {layer.identifier for layer in layers} | set(assets)
        got = 0
        for url in sorted(files):
            if not url.startswith(root + "/"):
                continue  # a built-in (OmniPBR.mdl): Kit has it locally
            dst = OUT / url[len(root) + 1:]
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, dst)
            got += 1
        print(f"{rel}: {got} files" + (f", unresolved {unresolved}" if unresolved else ""), flush=True)
    app.close()


if __name__ == "__main__":
    main()
