"""Audit a run's simulator episodes for harness faults: every failed skill call, sorted into the
model's errors and the harness's suspected faults, with the episodes each fault occurs in.

    python ladder/audit_episodes.py runs/full-1110
    python ladder/audit_episodes.py runs/full-1110 --tasks home_knife_toybox --list suspect.txt

A model's error is a call the scene rightly refuses: an object that cannot be picked up, a mark
that names nothing, a hand already full. A suspected harness fault is a call the model could not
have avoided by choosing differently: the arm finds no plan or no pose, a door or a put-down ends
short of where it was sent, a skill times out. Some failures are a task's design (the pictured mug
closes the fingers on nothing; a handed-over can has no place of its own): EXPECTED lists them by
task. Also flagged: episodes with no end record, ended by the call budget, or with a model turn
that has neither a call nor any text.

--list writes one line per suspect episode, `<task> <seed> <ablation> <episode.jsonl>`, for
sim/replay_check.sh to replay through the current harness.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# The scene refuses the call; choosing differently avoids it.
MODEL = [r"cannot be picked up", r"no object is marked", r"already holding", r"not holding",
         r"put it down first", r"has no door", r"has nothing to turn", r"there is no label on",
         r"is closed$", r"the door of .* is closed", r"no grasp fits it", r"has fallen over; it cannot be grasped",
         r"sealed shut", r"is not available in this scene", r"there is no waste location",
         r"is out of the arm's reach", r"there is no room on", r"there is no room left in"]
# The harness did not do what was asked, or could not.
HARNESS = [r"no plan", r"ik_fail", r"no way", r"timed out", r"did not end up", r"still \d+ degrees open",
           r"only \d+ degrees open", r"fell over \(", r"tipped \d+ degrees in the grasp", r"did not come up with the hand",
           r"cannot reach", r"could not get", r"closed on nothing", r"could not move in", r"but not over"]
# Failures a task produces by design: {task prefix: [patterns]}.
EXPECTED = {
    "home_mug_screen": [r"closed on nothing"],            # the "mug" is a picture on the monitor
    "home_canister_hob": [r"was handed to you; it has no place of its own"],
    "lab_27_culture_spill": [r"object fell during put-down"],  # the scripted mishap (planning_context.mishap)
}


def kind(task, note):
    for prefix, pats in EXPECTED.items():
        if task.startswith(prefix) and any(re.search(p, note) for p in pats):
            return "expected"
    if any(re.search(p, note) for p in MODEL):
        return "model"
    if any(re.search(p, note) for p in HARNESS):
        return "harness"
    return "unclassified"


def norm(note):
    return re.sub(r"-?\d+(\.\d+)?", "#", note)[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tasks", nargs="*", help="task name prefixes to audit (default: all)")
    ap.add_argument("--list", help="write the suspect episodes here, for sim/replay_check.sh")
    args = ap.parse_args()
    counts, where = Counter(), defaultdict(set)
    flags, suspects, n = Counter(), {}, 0
    for f in sorted(Path(args.run).glob("**/ablation-*/seed*/episode.jsonl")):  # a run, or replay_check.sh's replays
        task, level, seed = f.parents[2].name, f.parents[1].name, int(f.parent.name.removeprefix("seed"))
        if args.tasks and not any(task.startswith(t) for t in args.tasks):
            continue
        n += 1
        recs = [json.loads(line) for line in open(f)]
        end = next((r for r in reversed(recs) if r.get("event") == "end"), None)
        why = []
        if end is None:
            why.append("no end record")
        elif end.get("reason") not in ("reply",):
            why.append(f"ended by {end.get('reason')}")
        for r in recs:
            if r.get("source") != "model":
                continue
            if not r.get("call") and not r.get("skill") and not (r.get("response") or "").strip():
                why.append("empty model turn")
            if r.get("status") == "failed":
                note = r.get("note") or ""
                k = kind(task, note)
                key = (k, task, r.get("skill"), norm(note))
                counts[key] += 1
                where[key].add(str(f))
                if k in ("harness", "unclassified"):
                    why.append(f"{r.get('skill')}: {note}")
        for w in why:
            flags[w.split(":")[0] if ":" in w else w] += 1
        if why:
            suspects[str(f)] = (task, seed, level.removeprefix("ablation-").split("+")[0], why)
    print(f"{n} episodes, {len(suspects)} with a suspected harness fault or a flag\n")
    for k in ("harness", "unclassified", "expected", "model"):
        rows = [(c, key) for key, c in counts.items() if key[0] == k]
        if not rows:
            continue
        print(f"## {k} ({sum(c for c, _ in rows)} calls)")
        for c, (_, task, skill, note) in sorted(rows, reverse=True):
            print(f"  {c:4d}  {task}  {skill}: {note}  [{len(where[(k, task, skill, note)])} episodes]")
        print()
    other = {w: c for w, c in flags.items() if w in ("no end record", "empty model turn") or w.startswith("ended by")}
    if other:
        print("## flags")
        for w, c in sorted(other.items(), key=lambda x: -x[1]):
            print(f"  {c:4d}  {w}")
    if args.list:
        Path(args.list).write_text("".join(f"{t} {s} {a} {f}\n" for f, (t, s, a, _) in suspects.items()))
        print(f"\n{len(suspects)} suspect episodes listed in {args.list}")


if __name__ == "__main__":
    main()
