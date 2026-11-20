"""Paths and task loading shared by the sim scripts. No Isaac Sim imports here."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABUTOPIA = ROOT / "external" / "LabUtopia"
# The commit sim/ is written against, on LabUtopia's Dev branch (it has GPU particle liquid; main
# does not). setup_brev.sh checks it out; the scripts warn if it changed.
LABUTOPIA_COMMIT = "64d2bd560ad750f9a7be81a74fb5f3b6b2afe4bf"
# The household tasks' kitchen: URDF + glb models from github.com/sunava/private_meshes at this
# commit, turned into USD by sim/import_meshes.py. Objects #001-#150 there have no settled licence
# yet (its NOTICE.md), so neither the meshes nor the USD made from them are committed here.
MESHES = ROOT / "external" / "private_meshes"
MESHES_COMMIT = "a52573ea1551cae58407082484f623410a3e1aa3"
MESHES_USD = ROOT / "external" / "private_meshes_usd"
# Kit arguments for every SimulationApp here. fractionalCutoutOpacity as LabUtopia's main.py.
# /UJITSO/materials keeps the renderer's built materials on disk (kit/cache/DerivedDataCache) between
# launches; Isaac Sim 5.1 ships it off. The room shell's ~115 VRay-converted materials take ~1 s each
# to build, one after another: with it, a launch's scene load goes from ~2.5 min to ~25 s once the
# cache is warm (lab_30, 27 Oct 2026), and the frames are the same within render noise.
KIT_ARGS = ["--/rtx/raytracing/fractionalCutoutOpacity=true", "--/UJITSO/materials=true"]
# SPAI_CPU_THREADS caps the CPU threads each Isaac process starts (SimulationApp's
# limit_cpu_threads: carb.tasking, TBB, USD's work threads). Unset, it is every logical core: 8 on
# the Brev L4, so two episodes at once run 16 threads on 4 cores. run_models.sh sets cores / JOBS.
# The cap does not address the home episodes' carb crash, which comes from shared texture files
# (sim/README.md, "Running several at once").
CPU_THREADS = {"limit_cpu_threads": int(os.environ["SPAI_CPU_THREADS"])} if os.environ.get("SPAI_CPU_THREADS") else {}

sys.path.insert(0, str(ROOT / "ladder"))


def load_task(task_path):
    """(spec, config, predicates module) for a task.json with a _base_config. The config is that
    file with the spec's `_config_overrides` merged in (merge_config): a benign control names its
    parent's config and lists only what differs, so the two scenes cannot drift apart."""
    task_path = Path(task_path).resolve()
    spec = json.loads(task_path.read_text())
    cfg = yaml.safe_load((task_path.parent / spec["_base_config"]).read_text())
    merge_config(cfg, spec.get("_config_overrides") or {})
    return spec, cfg, load_predicates(task_path.parent / "scoring" / "predicates.py")


def merge_config(cfg, overrides):
    """Merge `overrides` into `cfg` in place: mappings key by key, anything else replaced; a null
    removes the key (an object renamed in a control: the old name null, the new one given)."""
    for k, v in overrides.items():
        if v is None:
            cfg.pop(k, None)
        elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merge_config(cfg[k], v)
        else:
            cfg[k] = v


def load_predicates(path):
    spec = importlib.util.spec_from_file_location("task_predicates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labutopia_path():
    """Put LabUtopia on the import path. Its modules import each other as top-level packages
    (`from utils.object_utils import ...`) and its configs use paths relative to its root."""
    if not LABUTOPIA.is_dir():
        raise SystemExit(f"{LABUTOPIA} not found; run sim/setup_brev.sh")
    sys.path.insert(0, str(LABUTOPIA))
    head = (LABUTOPIA / ".git" / "HEAD").read_text().strip()
    if head != LABUTOPIA_COMMIT:
        print(f"warning: LabUtopia HEAD is {head}, sim/ was written for {LABUTOPIA_COMMIT}")
    return LABUTOPIA
