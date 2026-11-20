"""Validate task specs against schema/task.schema.json.

    python scripts/check_tasks.py tasks/lab/*/task.json tasks/home/*/task.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = Path(__file__).resolve().parent.parent / "schema" / "task.schema.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    # Patterns are expanded here too, for launchers that don't use a shell (VS Code debug configs).
    args.paths = [Path(p) for g in args.paths for p in (sorted(glob.glob(str(g))) or [g])]

    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
    failed = 0
    for path in args.paths:
        task = json.loads(path.read_text())
        errors = sorted(validator.iter_errors(task), key=lambda e: list(e.path))
        for e in errors:
            print(f"{path}: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}")
        failed += bool(errors)
    print(f"{len(args.paths) - failed}/{len(args.paths)} valid")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
