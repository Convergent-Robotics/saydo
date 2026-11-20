"""Parse the model's call and resolve its object arguments to task objects. No Isaac imports.

At ablation 7 the model sees images, not object ids, so an argument is whatever it calls the
object: "bottle_A", "bottle A", "the bottle labelled sodium hypochlorite", "the beaker on the
right". The resolver matches it against what the model can know about each object (its name in
the command, its kind, its label, where it stands), and fails on a tie rather than guess: a
wrong guess would make the robot act on an object the model did not mean. Every resolution is
logged with the episode.
"""
import ast
import re

ARG_KEYS = {"object", "source", "target", "location"}  # arguments that name an object
STOPWORDS = {"the", "a", "an", "of", "on", "in", "to", "with", "and", "labelled", "labeled",
             "label", "that", "which", "is", "says", "reads", "one", "from", "at", "it"}
KINDS = ("bottle", "beaker", "flask", "container", "plate")  # plus each object's `kind` from the config


class CallError(ValueError):
    pass


def parse_call(text, skills):
    """The first line of `text` that is a call to a known skill, as (raw, skill, args).

    `skills` maps each skill to its argument names, for positional arguments.
    Raises CallError with a message for the model if no line is a call.
    """
    text = re.sub(r"```\w*", "", text)
    for line in text.splitlines():
        line = line.strip().rstrip(";")
        m = re.match(r"([a-z_]+)\s*\(.*\)$", line)
        if not m or m.group(1) not in skills:
            continue
        try:
            call = ast.parse(line, mode="eval").body
        except SyntaxError:
            raise CallError(f"could not parse {line!r}")
        names = skills[m.group(1)]
        if len(call.args) > len(names):
            raise CallError(f"{m.group(1)} takes {len(names)} arguments: {', '.join(names)}")
        try:
            args = {names[i]: ast.literal_eval(a) for i, a in enumerate(call.args)}
            args.update({k.arg: ast.literal_eval(k.value) for k in call.keywords})
        except ValueError:
            raise CallError(f"arguments must be quoted strings or numbers: {line!r}")
        unknown = set(args) - set(names)
        if unknown:
            raise CallError(f"{m.group(1)} has no argument {', '.join(sorted(unknown))}")
        return line, m.group(1), args
    raise CallError("no call to a skill from the API list found")


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if w not in STOPWORDS}


def describe(name, label, side, kind=None):
    """The words that can pick out one object: its name, kind, label and side."""
    w = words(name) | words(label or "") | words(kind or "") | {side}
    if side == "middle":
        w |= {"centre", "center"}
    return w


def sides(positions):
    """left/middle/right for each object from its y (robot frame: +y is the robot's left): the
    outer thirds of the objects' y span are left and right, the inner third is the middle. Two
    objects in one third share a side, and the resolver then needs another word to tell them apart."""
    ys = {n: p[1] for n, p in positions.items()}
    if len(ys) < 2:
        return {n: "middle" for n in ys}
    lo, hi = min(ys.values()), max(ys.values())
    third = (hi - lo) / 3
    return {n: "left" if y > hi - third else "right" if y < lo + third else "middle" for n, y in ys.items()}


def resolve(arg, objects, extra=()):
    """Resolve one argument to a name in `objects` ({name: {label, side}}) or in `extra`
    (places that are not objects, e.g. "bench", "waste").

    Returns (name, how). Raises CallError on no match or a tie.
    """
    key = " ".join(sorted(words(arg)))
    for name in list(objects) + list(extra):
        if key == " ".join(sorted(words(name))):
            return name, "name"
    all_kinds = set(KINDS) | {w for o in objects.values() for w in words(o.get("kind") or "")}
    # A place word counts only when no object is named with it ("waste" vs waste_beaker)
    named = set().union(*(words(n) for n in objects)) if objects else set()
    for place in extra:
        if place in words(arg) and place not in named and not words(arg) & all_kinds:
            return place, "place word"
    asked = words(arg)
    kinds = asked & all_kinds
    scores = {}
    for name, o in objects.items():
        have = describe(name, o.get("label"), o.get("side", ""), o.get("kind"))
        if kinds and not kinds & have:
            continue  # "the beaker ..." never resolves to a bottle
        scores[name] = len(asked & have)
    best = max(scores.values(), default=0)
    hits = [n for n, s in scores.items() if s == best and s > 0]
    if len(hits) == 1:
        return hits[0], f"description ({best} words)"
    if not hits:
        raise CallError(f"no object matches {arg!r}")
    raise CallError(f"{arg!r} could be any of {', '.join(sorted(hits))}; be more specific")
