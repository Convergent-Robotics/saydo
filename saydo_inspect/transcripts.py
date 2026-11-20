"""Write each trial's conversation from an Inspect Robots run's eval logs next to its episode.jsonl.

episode.py saves the conversation as sent at every model turn (conversation/turn<k>.json). The
Inspect harness does not: the policy's whole conversation is in the eval log instead
(samples[0].policy_transcripts[0].messages, images as {"type": "image", "frame": ...}). This writes
it to <episodes>/<model slug>/<scene_id>/transcript.json, which ladder/noticed.py reads where
conversation/ is missing (the messages before the k-th assistant turn are what turn k was sent).

Usage:  python saydo_inspect/transcripts.py runs/ir-full-1117/inspect-logs runs/ir-full-1117/sim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", type=Path, help="the run's inspect-logs directory")
    ap.add_argument("episodes", type=Path, help="the episodes directory: <model slug>/<task>/<ablation>/seed<n>")
    args = ap.parse_args()
    written, missing, empty = 0, [], 0
    # A re-run trial (irfill) has the failed attempt's log too: oldest first, so the last one wins.
    logs = sorted((json.loads(f.read_text()) for f in args.logs.glob("*.json")), key=lambda d: d["eval"]["created"])
    for log in logs:
        sample = log["samples"][0]
        transcripts = sample.get("policy_transcripts") or []
        if not transcripts:
            empty += 1
            continue
        t = transcripts[0]
        out = args.episodes / t["model"].replace("/", "_") / sample["scene_id"]
        if not out.is_dir():
            missing.append(str(out))
            continue
        (out / "transcript.json").write_text(json.dumps(t["messages"], ensure_ascii=False) + "\n")
        written += 1
    print(f"wrote {written} transcripts; {empty} logs without one; {len(missing)} with no episode directory")
    for m in missing[:10]:
        print("  no episode directory:", m)


if __name__ == "__main__":
    main()
