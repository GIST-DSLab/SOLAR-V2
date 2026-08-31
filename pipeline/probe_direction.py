#!/usr/bin/env python3
"""Does the trajectory go the way the verifier says the task goes?

The verifier is the task rewritten in RE-ARC's DSL, so the functions it calls
name the concept: `hmirror` is a reflection, `gravitate` a fall, `upscale` a
replication. A trajectory can land on the right grid without any of that — paint
the differing cells one region at a time and the answer is correct while the
rule has vanished from what the policy did.

This does not ask for the same operations. Following the DSL step for step is
often wrong for us: the verifier has to say everything in one expression, so it
writes "fall downwards" as rot90 - shift - rot270, where a trajectory just moves
the object down. A transform that appears together with its own inverse is that
kind of scaffolding and is ignored here. What is left is the direction, and the
question is only whether the trajectory travels in it at all.

This is a diagnostic and no longer a demand. `critique_to_feedback.py` drops
the finding rather than passing it on: asking for the operation got routes that
contain it and perform nothing, and a route that carries the concept on some
fraction of its instances turned out to be the surest sign of that -- every
maker in the release scoring between 0 and 0.5 has a group of operations that
composes back to where it started. Use it to decide what to look at.

No LLM calls. Reads recorded episodes, so it costs a rollout and nothing else.

    python pipeline/probe_direction.py --root $SOLAR_DATA_ROOT/draw/whole \\
        --concepts reflection --out probe_direction.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
from pathlib import Path

SOLAR_ROOT = Path(__file__).resolve().parents[1]

# a DSL idea, and the ARCLE operations that carry it
CONCEPTS = {
    "reflection":  ({"hmirror", "vmirror", "dmirror", "cmirror"},
                    {"FlipH", "FlipV", "Rotate90", "Rotate270"}),
    "rotation":    ({"rot90", "rot180", "rot270"},
                    {"Rotate90", "Rotate270", "FlipH", "FlipV"}),
    "translation": ({"move", "gravitate"},
                    {"MoveU", "MoveD", "MoveL", "MoveR"}),
    # upscale is deliberately absent. Enlarging a grid has no operation in
    # ARCLE, so a trajectory that scales something up has to draw the result --
    # painting is the route here, not a way of avoiding it.
    #
    # `occurrences` is absent for a different reason: it is not a replication.
    # It answers "where does this pattern appear", and what the rule does at
    # the places it finds is almost always a recolor or a fill -- read the
    # verifiers and the call after it is `paint`, `fill` or `recolor`, not a
    # copy of anything. Asking for a Paste here asks for an op that carries no
    # part of the rule.
    "replication": ({"repeat", "hperiod", "vperiod"},
                    {"CopyI", "CopyO", "Paste"}),
    "reframing":   ({"crop", "subgrid", "trim", "compress", "downscale"},
                    {"ResizeGrid", "CopyInput"}),
}

# Concepts worth naming but not worth forcing. ARCLE has Move, so a fall that
# is painted could have been moved and it is better when it is -- but a rule
# that ends up placing cells is not wrong for placing them, and a trajectory
# should not acquire a Move to satisfy this line.
SOFT = {"translation"}

FINDING_CODES = {
    "CONCEPT_DIRECTION_MISSING":
        "the verifier's rule is geometric and the trajectory contains nothing "
        "geometric — the grid is right, the concept is not visible in the route",
}


# functions that reduce a grid to a measurement: whatever is transformed to feed
# one of these was transformed in order to be measured, and the measurement is a
# number, not the answer
REDUCERS = {"size", "width", "height", "colorcount", "numcolors", "mostcolor",
            "leastcolor", "palette", "hperiod", "vperiod", "ulcorner", "lrcorner",
            "index", "dedupe", "shape", "portrait", "square", "even",
            # arithmetic on a measurement is still a measurement. 3eda0437
            # mirrors the grid to read the tallest run off it and take a
            # maximum; without these the chain escapes into `maximum` and the
            # mirror reads as the rule.
            "maximum", "minimum", "increment", "decrement", "interval",
            "valmax", "valmin"}


def _test_only(fn: ast.FunctionDef) -> set[str]:
    """DSL names the verifier only ever looks through, never applies.

    RE-ARC's verifiers are single-assignment, so a name's role can be read off
    the chain it feeds. `matcher` turns a function into a boolean, and nothing
    downstream of a boolean can be the grid that comes back -- so a transform
    reaching the return only through one is a test the verifier runs to decide
    something, not the transformation the task performs. d8c310e9 asks whether
    the grid deduplicates under `cmirror` in order to find which axis its
    pattern repeats along; it is a period-and-tile task, and reading that
    cmirror as the rule had every regeneration told to put in a flip the task
    does not contain.
    """
    assigns, consumers = {}, collections.defaultdict(set)
    for st in fn.body:
        if not (isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)):
            continue
        tgt = st.targets[0].id
        assigns[tgt] = st.value
        for n in ast.walk(st.value):
            if isinstance(n, ast.Name):
                consumers[n.id].add(tgt)

    def base(node) -> bool:
        """A boolean, or a measurement: nothing past here is the grid."""
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and (node.func.id == "matcher" or node.func.id in REDUCERS))

    test = {k for k, v in assigns.items() if base(v)}
    changed = True
    while changed:                       # a node is a test if every use of it is
        changed = False                  # -- and an unused one is the return value
        for k in assigns:
            if k in test or not consumers[k]:
                continue
            if consumers[k] <= test:
                test.add(k)
                changed = True

    names = collections.defaultdict(set)
    for k, v in assigns.items():
        for n in ast.walk(v):
            if isinstance(n, ast.Name) and n.id not in assigns:
                names[n.id].add(k)
    # the reducer's own name still says what the task is -- 29ec7d0e transposes
    # only to read a period off the result, and the period is the rule
    return {nm for nm, used_in in names.items()
            if used_in and used_in <= test and nm not in REDUCERS}


def _cycled(fn: ast.FunctionDef) -> set[str]:
    """Transforms the verifier applies in a cycle that returns where it began.

    `power(compose(rot90, f), FOUR)` is not a rotation the task performs -- four
    quarter turns are the identity. It is how the DSL says "and the same in
    every orientation", and 150deff5 builds its whole rule that way. A route
    that turned the grid four times to satisfy it would be turning it for show.
    """
    assigns, consumers = {}, collections.defaultdict(set)
    for st in fn.body:
        if not (isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)):
            continue
        assigns[st.targets[0].id] = st.value
        for n in ast.walk(st.value):
            if isinstance(n, ast.Name):
                consumers[n.id].add(st.targets[0].id)

    cyc = set()
    for k, v in assigns.items():
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "power":
            seen, stack = set(), [a.id for a in v.args if isinstance(a, ast.Name)]
            while stack:                 # everything folded into the repeated step
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                node = assigns.get(cur)
                if node is None:
                    cyc.add(cur)
                    continue
                stack += [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
    return {n for n in cyc if n in TURNS}


MEASURES = REDUCERS | {"color"}

TURNS = {"rot90", "rot180", "rot270", "hmirror", "vmirror", "dmirror", "cmirror"}


def _ssa(fn: ast.FunctionDef) -> dict[str, ast.AST]:
    return {st.targets[0].id: st.value for st in fn.body
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)}


def _from_input(name: str, assigns: dict, seen=None) -> bool:
    """Does this value descend from I, or was it built out of constants?

    A measurement taken from I does not carry the grid forward: 98cf29f8 reads
    one colour off the input and assembles a three-cell pattern around it, and
    that pattern is a thing the verifier made, not a piece of the grid. So the
    walk stops at anything that turns a grid into a number, a colour or a shape.
    """
    if name == "I":
        return True
    seen = seen or set()
    if name in seen or name not in assigns:
        return False
    seen.add(name)

    def reached(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in MEASURES:
            return False
        if isinstance(node, ast.Name):
            return _from_input(node.id, assigns, seen)
        return any(reached(c) for c in ast.iter_child_nodes(node))

    return reached(assigns[name])


def _not_of_the_grid(fn: ast.FunctionDef) -> set[str]:
    """Turns applied to a shape the verifier built, not to anything from I.

    98cf29f8 mirrors a three-cell pattern it assembles out of constants so it
    can search for both orientations of it. The grid is never turned; what the
    rule does at the places found is a fill. A route that flipped the grid to
    account for this would be flipping it for show.
    """
    assigns = _ssa(fn)
    out = set()
    for v in assigns.values():
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)):
            continue
        if v.func.id not in TURNS:
            continue
        args = [a.id for a in v.args if isinstance(a, ast.Name)]
        if args and not any(_from_input(a, assigns) for a in args):
            out.add(v.func.id)
    return out


def _padded(fn: ast.FunctionDef) -> set[str]:
    """`trim` that only undoes a border the verifier added itself.

    canvas(add(TWO, shape(I))) ... trim(...) is a one-cell margin taken so that
    neighbourhood logic has somewhere to look, and then given back. The grid
    that comes out is the size the grid that went in was, so nothing about the
    task is a reframing.
    """
    assigns = _ssa(fn)
    grew = any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
               and v.func.id == "add"
               and any(isinstance(a, ast.Name) and a.id == "TWO" for a in v.args)
               and any(isinstance(a, ast.Name) and isinstance(assigns.get(a.id), ast.Call)
                       and assigns[a.id].func.id == "shape" for a in v.args)
               for v in assigns.values())
    return {"trim"} if grew else set()


def verifier_concepts(rearc_root: Path) -> dict[str, set[str]]:
    """Per task, the DSL names it calls, with scaffolding and tests dropped."""
    tree = ast.parse((rearc_root / "verifiers.py").read_text(encoding="utf-8"))
    out = {}
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("verify_")):
            continue
        names = collections.Counter(n.id for n in ast.walk(fn) if isinstance(n, ast.Name))
        scaffold = set()
        if names["rot90"] and names["rot270"]:
            scaffold |= {"rot90", "rot270"}
        if names["rot180"] >= 2:
            scaffold.add("rot180")
        for m in ("hmirror", "vmirror", "dmirror", "cmirror"):
            if names[m] >= 2:
                scaffold.add(m)
        scaffold |= _cycled(fn) | _not_of_the_grid(fn) | _padded(fn)
        out[fn.name[len("verify_"):]] = set(names) - scaffold - _test_only(fn)
    return out


def ops_used(task_dir: str, limit: int) -> collections.Counter:
    c = collections.Counter()
    for f in sorted(glob.glob(f"{task_dir}/*.json"))[:limit]:
        c.update(json.loads(Path(f).read_text())["operation_name"])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a rollout directory, e.g. <draw>/whole")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--concepts", nargs="+", default=sorted(CONCEPTS),
                    choices=sorted(CONCEPTS))
    # A defect here is "no episode carries the concept", so the sample size is
    # the claim's strength. Read the whole draw.
    ap.add_argument("--episodes", type=int, default=25,
                    help="episodes read per task")
    ap.add_argument("--rearc_root", default=str(SOLAR_ROOT / "re-arc"))
    ap.add_argument("--out", default="probe_direction.json")
    args = ap.parse_args()

    concepts = verifier_concepts(Path(args.rearc_root))
    recs, flagged = [], 0
    for d in sorted(glob.glob(f"{args.root}/test.*")):
        task = d.split("test.")[1].split(".")[0]
        if args.tasks and task not in args.tasks:
            continue
        if task not in concepts:
            continue
        ops = ops_used(d, args.episodes)
        if not ops:
            continue
        rec = {"task_id": task, "verdict": "PASS", "findings": []}
        # One is enough. A verifier that calls both dmirror and repeat is not
        # asking for a flip *and* a paste; it is one rule described in the terms
        # the DSL had. 8e1813be flips on every instance it solves and was still
        # being called defective for the repeat, and the round that produced
        # came back with a maker that flips on none of them.
        carried = any(concepts[task] & dsl and set(ops) & arcle
                      for dsl, arcle in CONCEPTS.values())
        for name in args.concepts:
            dsl, arcle = CONCEPTS[name]
            named = concepts[task] & dsl
            if not named or carried:
                continue
            top = ", ".join(f"{k} x{v}" for k, v in ops.most_common(6))
            rec["verdict"] = "REVISE"
            rec["findings"].append({
                "code": "CONCEPT_DIRECTION_MISSING",
                "severity": "low" if name in SOFT else "medium",
                "evidence": (
                    f"This task's RE-ARC verifier calls {', '.join(sorted(named))}, "
                    f"so the rule is a {name}. A trajectory carrying that out would "
                    f"use one of {', '.join(sorted(arcle))}; yours uses none of them. "
                    f"What it does use: {top}.\n\n"
                    f"The grid you produce is correct — this is about the route. "
                    + (f"If the rule moves something, moving it is the better route "
                       f"and it is worth trying. But a rule that ends up placing "
                       f"cells is not wrong for placing them: keep what you have "
                       f"rather than add a Move to satisfy this line, and say so by "
                       f"leaving the route as it is."
                       if name in SOFT else
                       f"Re-derive so the {name} is something the trajectory "
                       f"performs, not something the final grid merely reflects.")),
            })
        recs.append(rec)
        flagged += rec["verdict"] != "PASS"

    Path(args.out).write_text(json.dumps(recs, indent=1))
    print(f"tasks read {len(recs)} | direction not carried out {flagged}")
    per = collections.Counter(f["evidence"].split("so the rule is a ")[1].split(".")[0]
                              for r in recs for f in r["findings"])
    for k, v in per.most_common():
        print(f"  {v:4d}  {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
