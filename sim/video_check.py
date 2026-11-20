"""
Watch a run's episode videos and count skill and simulator errors, call by call.

Every physical call (pick, place, pour, open, ...; not say, done, wait or lookup) is pre-classified
from episode.jsonl:
  model     the call asked for something the scene does not allow ("cannot be picked up",
            "has nothing to turn on", "not holding ..."): the model's mistake, not the harness's
  mishap    a scripted mishap (the task's own, e.g. the culture spill's fall during put-down)
  execution any other failed call: no grasp, cannot reach, timed out, no plan. The suspects
  ok        the call reported success
Video time is sim time from the episode's first call (the video runs in real time from there), so
each call has a segment of the video, up to the next call.

  serve    a local page (http://localhost:8766): one episode at a time, its video and its calls;
           a click plays a call's segment. Each failed call gets a verdict (harness fault, fair
           physical failure, model's fault, can't tell); any ok call can be flagged as a silent error;
           each episode, whether the video agrees with its scored outcome. Labels append to
           <run>/video-check/labels.jsonl; the last per episode counts.
  report   counts per kind and verdict, harness faults per call and per episode with intervals,
           by skill and by task, and a list of every fault with its video and time.

Usage (videos must be local: rsync the run back from the instance first):
    python sim/video_check.py serve runs/vid-1103
    python sim/video_check.py report runs/vid-1103
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NOT_PHYSICAL = {"say", "done", "wait", "lookup"}
MODEL_FAULT = re.compile(r"cannot be picked up|has nothing to turn on|not holding|already holding|"
                         r"is not available in this scene|has no door|pick it up first")
VERDICTS = ("harness", "fair", "model", "unclear")


def kind(r):
    if r.get("mishap") in (True, "True"):
        return "mishap"
    if r.get("status") != "failed":
        return "ok"
    return "model" if MODEL_FAULT.search(str(r.get("note"))) else "execution"


def episodes(run):
    """Every episode with a video, as the page shows it."""
    out = []
    for path in sorted(run.glob("sim/*/*/ablation-*/seed*/episode.jsonl")):
        d = path.parent
        if not (d / "video.mp4").exists():
            continue
        rs = [json.loads(l) for l in open(path)]
        timed = [r for r in rs if r.get("t_sim") is not None]
        if not timed:
            continue
        t0 = float(timed[0]["t_sim"])
        end = next((r for r in rs if r.get("event") == "end"), {})
        calls = []
        for n, r in enumerate(timed):
            if r.get("source") != "model" or not r.get("skill") or r["skill"] in NOT_PHYSICAL:
                continue
            nxt = next((float(x["t_sim"]) for x in timed[n + 1:] if float(x["t_sim"]) > float(r["t_sim"])), None)
            calls.append({"step": r["step"], "raw": r.get("raw"), "skill": r["skill"], "status": r.get("status"),
                          "note": r.get("note"), "kind": kind(r), "reasoning": r.get("reasoning"),
                          "trigger": r.get("trigger"), "start": float(r["t_sim"]) - t0,
                          "end": None if nxt is None else nxt - t0})
        model, task, ablation, seed = d.relative_to(run / "sim").parts
        out.append({"id": f"{model}/{task}/{ablation}/{seed}", "model": model, "task": task,
                    "ablation": ablation, "seed": seed, "video": str((d / "video.mp4").relative_to(run)),
                    "outcome": end.get("outcome"), "execution_goal": end.get("execution_goal"),
                    "end_reason": end.get("reason"), "calls": calls})
    return out


def check_dir(run):
    d = run / "video-check"
    d.mkdir(exist_ok=True)
    return d


def latest_labels(path):
    return {l["id"]: l for l in map(json.loads, open(path))} if path.exists() else {}


# ---------- serve ----------

def serve(args):
    run = args.run.resolve()
    labels_path = check_dir(run) / "labels.jsonl"
    page = Path(__file__).with_name("video_check.html").read_bytes()
    eps = episodes(run)

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, ctype, status=200, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/episodes":
                self.send(json.dumps({"episodes": eps, "labels": latest_labels(labels_path)}).encode(),
                          "application/json")
            elif self.path.startswith("/video/"):
                path = (run / self.path.removeprefix("/video/")).resolve()
                if not (path.is_relative_to(run) and path.name == "video.mp4" and path.exists()):
                    return self.send_error(404)
                self.send_range(path)
            else:
                self.send(page, "text/html; charset=utf-8")

        def send_range(self, path):
            # Byte ranges, so the browser can seek to a call's segment.
            size = path.stat().st_size
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if not m:
                return self.send(path.read_bytes(), "video/mp4", extra=[("Accept-Ranges", "bytes")])
            start = int(m[1]) if m[1] else size - int(m[2])
            end = int(m[2]) if m[1] and m[2] else size - 1
            with open(path, "rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            self.send(body, "video/mp4", 206, [("Accept-Ranges", "bytes"),
                                               ("Content-Range", f"bytes {start}-{end}/{size}")])

        def do_POST(self):
            label = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(labels_path, "a") as f:
                f.write(json.dumps(label) + "\n")
            self.send(b"{}", "application/json")

        def log_message(self, *a):
            pass

    print(f"http://localhost:{args.port}  ({len(eps)} episodes with video; labels -> {labels_path})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


# ---------- report ----------

def wilson(k, n, z=1.96):
    if not n:
        return "—"
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return f"{k}/{n} ({100 * p:.0f}%, 95% CI {100 * (m - c):.0f}–{100 * (m + c):.0f}%)"


def report(args):
    run = args.run
    eps = {e["id"]: e for e in episodes(run)}
    labels = latest_labels(check_dir(run) / "labels.jsonl")
    done = [(eps[i], l) for i, l in labels.items() if i in eps]
    calls = [(e, c, l["calls"].get(str(c["step"]), {})) for e, l in done for c in e["calls"]]

    def is_fault(c, cl):
        return cl.get("verdict") == "harness" or bool(cl.get("silent"))

    out = [f"# Video check: {run.name}", "",
           f"{len(done)} of {len(eps)} episodes watched, {len(calls)} physical calls. A **harness fault** "
           "is a failed call judged the harness's or simulator's fault, or an ok call flagged as a silent "
           "error.", "",
           "## Harness faults", "",
           f"- Per physical call: {wilson(sum(is_fault(c, cl) for _, c, cl in calls), len(calls))}",
           f"- Episodes with at least one: {wilson(sum(any(is_fault(c, l['calls'].get(str(c['step']), {})) for c in e['calls']) for e, l in done), len(done))}",
           f"- Silent errors (call reported ok): {sum(bool(cl.get('silent')) for _, _, cl in calls)}",
           f"- Video disagrees with the scored outcome: "
           f"{wilson(sum(l.get('outcome_agrees') == 'no' for _, l in done), sum(l.get('outcome_agrees') in ('yes', 'no') for _, l in done))}",
           "", "## Failed calls: pre-classified kind against verdict", "",
           "| Kind | " + " | ".join(VERDICTS) + " | unlabelled |", "|---|" + "---|" * (len(VERDICTS) + 1)]
    table = defaultdict(Counter)
    for _, c, cl in calls:
        if c["kind"] != "ok":
            table[c["kind"]][cl.get("verdict") or "unlabelled"] += 1
    for k in ("execution", "model", "mishap"):
        out.append(f"| {k} | " + " | ".join(str(table[k][v]) for v in VERDICTS + ("unlabelled",)) + " |")

    for title, key in (("By skill", lambda e, c: c["skill"]), ("By task", lambda e, c: e["task"])):
        out.extend(["", f"## {title}", "", "| | calls | failed | harness faults |", "|---|---|---|---|"])
        groups = defaultdict(list)
        for e, c, cl in calls:
            groups[key(e, c)].append((c, cl))
        for g, xs in sorted(groups.items(), key=lambda kv: -sum(is_fault(c, cl) for c, cl in kv[1])):
            out.append(f"| {g} | {len(xs)} | {sum(c['status'] == 'failed' for c, _ in xs)} "
                       f"| {sum(is_fault(c, cl) for c, cl in xs)} |")

    out.extend(["", "## Every harness fault", ""])
    for e, c, cl in calls:
        if is_fault(c, cl):
            what = "silent error" if cl.get("silent") else f"failed: {c['note']}"
            out.append(f"- `{e['id']}` at {c['start']:.0f}s, `{c['raw']}`: {what}. {cl.get('comment') or ''}")
    out.extend(["", "## Episodes where the video disagrees with the score", ""])
    for e, l in done:
        if l.get("outcome_agrees") == "no":
            out.append(f"- `{e['id']}`: scored {e['outcome']}. {l.get('note') or ''}")

    text = "\n".join(out)
    (check_dir(run) / "report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("run", type=Path)
    p.add_argument("--port", type=int, default=8766)
    p = sub.add_parser("report")
    p.add_argument("run", type=Path)
    args = parser.parse_args()
    {"serve": serve, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
