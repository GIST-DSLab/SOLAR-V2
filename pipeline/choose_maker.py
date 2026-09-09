#!/usr/bin/env python3
"""Given several versions of a maker, keep the one that holds up.

A feedback round produces a candidate; the question that follows is whether to
take it. That question was being answered by hand, and by hand it was answered
badly: a round that put the concept back into the route was promoted because the
concept was there, and nobody measured that the route now read its parameters
off O, or that an ancestor had done the same job without them.

So the candidates are scored, not read, and the ancestor is a candidate too --
`arc-best` competes with the newest regeneration and wins where it deserves to.
Every candidate sees the same instances, drawn the way the rollout draws them,
and each gives up its rates:

    solve   the ops, replayed on I, reach O
    copy    handed a different instance's output, the ops reach it anyway
    dep     the ops change when the answer is disturbed
    idle    some group of its turns or moves composes to the identity
    spare   the route reaches O with one of its own operations left out
    route   the ops contain the operation family the verifier's concept names

Only the first five decide anything. `route` is reported, and was once the last
filter: six routes were promoted by it alone and taken back out by hand in
628fda7, among them a transpose spelled with two flips that beat a crop. What
an operation family can be scored on, it can be satisfied by, so it says which
candidate carries the concept and never which one to keep.

`route` is `probe_direction` per instance rather than per maker, and that is the
point of measuring it here: a maker that flips on three quarters of its
instances and paints the rest passes the binary check and loses to an ancestor
that flips on all of them.

    python pipeline/choose_maker.py --candidates arc-agi-1 redo-copy arc-best \\
        --tasks 3345333e --out choose.json
    python pipeline/choose_maker.py --candidates arc-agi-1 redo-copy arc-best \\
        --out choose.json --apply
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import importlib.util
import inspect
import json
import random
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from arcle.loaders import Loader

SOLAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils  # noqa: E402
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

from probe_direction import CONCEPTS, verifier_concepts  # noqa: E402

MAX_GRID_DIM = (30, 30)
SPARE_MIN = 0.5
OP_ID = {name: i for i, name in enumerate(solar_utils.action_names)}

_REARC = {}


def _rearc():
    """re-arc's generators and verifiers, with its own utils under them."""
    if _REARC:
        return _REARC["G"], _REARC["V"]
    rs = str(SOLAR_ROOT / "re-arc")
    while rs in sys.path:
        sys.path.remove(rs)
    sys.path.insert(0, rs)
    for name in ("utils", "dsl", "generators", "verifiers"):
        sys.modules.pop(name, None)
    import generators as G
    import verifiers as V
    _REARC.update(G=G, V=V)
    return G, V


def grid_str(g) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def totuple(g):
    return tuple(tuple(int(x) for x in row) for row in np.asarray(g))


class _One(Loader):
    def __init__(self, I, O):
        self.pair = (I, O)
        self._pathlist = [""]
        self.data = self.parse()

    def get_path(self, **k):
        return [""]

    def parse(self, **k):
        I, O = self.pair
        a, b = np.asarray(I, np.uint8), np.asarray(O, np.uint8)
        return [([a], [b], [a], [b], {"id": "0"})]


def episodes_from_draw(root: str, task: str, limit: int = 10):
    """Episodes as the rollout wrote them: three demonstrations and a test.

    Read rather than rebuilt. A maker may now read its episode's
    demonstrations, so it has to be judged on demonstrations that share the
    episode's palette -- and the rollout already builds those, holding one
    colour assignment across the four pairs. Rebuilding them here meant
    reproducing that hold, and it kept coming apart: every maker deletes
    `generators` from sys.modules as it imports, so the module object the hold
    patched was not the one the draw went on to call. Four makers were marked
    down to a third of their coverage by demonstrations that shared nothing
    with their test, and all four solve every episode in a real draw.

    The instances are the generator's own and do not depend on which maker
    produced the file, so a draw made with one maker set scores another.
    """
    import glob as _glob
    # Both spellings of the path. The flag's own help says "a rollout
    # directory", and every caller so far typed the directory the rollout
    # printed, which ends in /whole; the mismatch returned None and the score
    # fell back to demonstrations built out of other drawn pairs, without
    # saying so. Judgements were made on that fallback for a week.
    base = str(root)[:-len("/whole")] if str(root).rstrip("/").endswith("/whole") \
        else str(root)
    hits = _glob.glob(f"{base}/whole/test.{task}.*")
    if not hits:
        return None
    out = []
    for f in sorted(_glob.glob(hits[0] + "/*.json"))[:limit]:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        def crop(g, dim):
            h, w = dim
            return [r[:w] for r in g[:h]]
        ex = [(crop(d["ex_in"][i], d["ex_in_grid_dim"][i]),
               crop(d["ex_out"][i], d["ex_out_grid_dim"][i]))
              for i in range(len(d["ex_in"]))]
        I = np.asarray(crop(d["in_grid"], d["grid_dim"][0]), int)
        O = np.asarray(crop(d["out_grid"], d["grid_dim"][-1]), int)
        out.append(([(np.asarray(a, int), np.asarray(b, int)) for a, b in ex], (I, O)))
    return out or None


def draw(task: str, n: int, seeds) -> list | None:
    """Instances as the rollout draws them: generator, size cap, verifier.

    Several seeds, pooled: one seed's sample is a narrow view of what the
    generator makes, and a maker that misses a whole shape of instance can look
    like it missed one by chance.

    One shape of instance is worth insisting on. Paste, CopyI and CopyO treat 0
    as nothing there, so a maker that replicates a region loses whatever role
    the palette happened to assign 0 to -- and the palette assigns it only
    sometimes, so the miss hides. Seven released makers solve every instance
    whose target holds no 0 and as little as none of the rest. The pool is
    topped up from further seeds until targets carrying a 0 are a quarter of
    it, when the generator makes any at all; a generator that never does is
    left alone.
    """
    out = []
    for sd in seeds:
        got = _draw_one(task, n, sd)
        if got is None:
            return None
        out += got
    want_zero = max(1, len(out) // 4)
    has_zero = sum(1 for _, O in out if 0 in np.unique(O))
    sd = max(seeds) + 1 if seeds else 1
    while has_zero < want_zero and sd <= max(seeds, default=0) + 12:
        got = _draw_one(task, n, sd) or []
        sd += 1
        keep = [(I, O) for I, O in got if 0 in np.unique(O)]
        if not keep:
            continue
        out += keep[:want_zero - has_zero]
        has_zero += len(keep[:want_zero - has_zero])
    return out


def _draw_one(task: str, n: int, seed: int):
    G, V = _rearc()
    gen = getattr(G, f"generate_{task}", None)
    ver = getattr(V, f"verify_{task}", None)
    if gen is None or ver is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    out = []
    for _ in range(n * 80):
        if len(out) >= n:
            break
        lb = random.random() * 0.8
        try:
            d = gen(lb, min(1.0, lb + 0.3))
        except Exception:
            continue
        I, O = np.asarray(d["input"], int), np.asarray(d["output"], int)
        if max(I.shape) > MAX_GRID_DIM[0] or max(O.shape) > MAX_GRID_DIM[1]:
            continue
        try:
            if totuple(ver(totuple(I))) != totuple(O):
                continue
        except Exception:
            continue
        out.append((I, O))
    return out


def load_derive(maker_path: Path):
    sys.path.insert(0, str(maker_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            f"gm_{maker_path.parent.parent.name}_{maker_path.parent.name}", maker_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "derive_operations", None)
    except Exception:
        return None



def call_derive(derive, I, O, examples=None):
    """derive_operations, given the episode's demonstrations when it takes them.

    Mirrors gen_rearc_trajectories_v2.call_derive: a maker that reads the rule's
    colour convention off three demonstrations is doing what a solver does, and
    a maker handed only the test pair had to read it off the answer instead.
    Two-argument makers are called exactly as before.
    """
    try:
        n = len(inspect.signature(derive).parameters)
    except (TypeError, ValueError):
        n = 2
    if n >= 3 and examples is not None:
        return derive(I, O, examples)
    return derive(I, O)


def replay(derive, I, O_shown, examples=None):
    """The ops the maker returns for (I, O_shown), and the grid they produce."""
    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=_One(I, O_shown), max_grid_size=MAX_GRID_DIM,
                   colors=10, max_episode_steps=None, max_trial=1)
    try:
        ops, sels = call_derive(derive, I.tolist(), O_shown.tolist(), examples)
        obs, _ = env.reset(options={"prob_index": 0, "adaptation": False})
        for op, sel in zip(ops, sels):
            obs, _, _, _, _ = env.step(
                {"selection": solar_utils.to_sel_mask(sel, MAX_GRID_DIM).astype(bool),
                 "operation": int(op)})
        h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
        return np.asarray(obs["grid"])[:h, :w].astype(int), list(ops), list(sels)
    except Exception:
        return None, None, None
    finally:
        env.close()


TERMINAL = {OP_ID.get("Submit", 34)}


def _run(I, O_shown, ops, sels):
    """The grid a given route produces, or None if the environment refuses it."""
    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=_One(I, O_shown), max_grid_size=MAX_GRID_DIM,
                   colors=10, max_episode_steps=None, max_trial=1)
    try:
        obs, _ = env.reset(options={"prob_index": 0, "adaptation": False})
        for op, sel in zip(ops, sels):
            obs, _, _, _, _ = env.step(
                {"selection": solar_utils.to_sel_mask(sel, MAX_GRID_DIM).astype(bool),
                 "operation": int(op)})
        h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
        return np.asarray(obs["grid"])[:h, :w].astype(int)
    except Exception:
        return None
    finally:
        env.close()


def spare_ops(I, O, ops, sels) -> list:
    """The operations this route reaches O without, named.

    No judgement is involved and there are no false positives: the route minus
    that operation is replayed and it still lands on the answer, so the
    operation was not paid for. `cancels` cannot see this one -- half the routes
    it catches break under either deletion -- and this cannot see that one, so
    both are asked. What neither sees is an operation that is needed for the
    wrong reason, and no measurement here will.
    """
    out = []
    for i, op in enumerate(ops):
        if int(op) in TERMINAL:
            continue
        g = _run(I, O, ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:])
        if g is not None and g.shape == O.shape and bool((g == O).all()):
            out.append((i, NAME.get(int(op), str(int(op)))))
    return out


TURN = {"FlipH": ((1, 0), (0, -1)), "FlipV": ((-1, 0), (0, 1)),
        "Rotate90": ((0, -1), (1, 0)), "Rotate270": ((0, 1), (-1, 0))}
MOVE = {"MoveU": (-1, 0), "MoveD": (1, 0), "MoveL": (0, -1), "MoveR": (0, 1)}
I2 = ((1, 0), (0, 1))
NAME = {i: nm for nm, i in OP_ID.items()}


def _mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def cancels(ops, sels) -> bool:
    """Does some group of geometric ops on one region compose to nothing?

    Deleting one of a mirror pair breaks the route, and deleting both breaks it
    too -- what sits between them is addressed in the mirrored frame -- so
    neither deletion test can see it, and counting which cells survive cannot
    either, because a mirror swaps cells in pairs and half of them coincide with
    the target whether the mirror is real or undone. The algebra can: turns on a
    region are elements of D4, moves are displacements, and a group that
    composes to the identity performs no reflection, rotation or translation at
    all. The same edits could have been made in the original frame.
    """
    turns, moves = collections.defaultdict(list), collections.defaultdict(list)
    for op, sel in zip(ops, sels):
        nm = NAME.get(int(op))
        key = tuple(sel) if isinstance(sel, (list, tuple)) else str(sel)
        if nm in TURN:
            turns[key].append(nm)
        elif nm in MOVE:
            moves[key].append(nm)
    for seq in turns.values():
        m = I2
        for nm in seq:
            m = _mul(TURN[nm], m)
        if m == I2:
            return True
    for seq in moves.values():
        if sum(MOVE[nm][0] for nm in seq) == 0 and sum(MOVE[nm][1] for nm in seq) == 0:
            return True
    return False


def concept_ops(task: str, vconcepts: dict) -> tuple[str | None, set[int]]:
    """The concepts the verifier names, and the ARCLE ops that would carry one.

    All of them, not the first: a verifier that calls both dmirror and repeat
    describes one rule in the terms the DSL had, and a route that flips is
    carrying it whichever of the two the table happens to list first.
    """
    called = vconcepts.get(task, set())
    names, ops = [], set()
    for name, (dsl, arcle) in CONCEPTS.items():
        if called & dsl:
            names.append(name)
            ops |= {OP_ID[o] for o in arcle if o in OP_ID}
    return ("/".join(names) if names else None), ops


def scrambles(O, rng):
    """The same answer with its arrangement, and then its colours, disturbed.

    Two questions, because one of them alone is answerable by accident. Moving
    O's cells leaves its colour multiset untouched, so a route that reads *which
    colours are in the answer* -- fill with the one that is in O and not in I --
    survives that and looks independent. Recolouring leaves the arrangement
    untouched, so a route that reads where something sits survives the other.
    A maker that returns the same operations under both took neither.
    """
    flat = O.reshape(-1).copy()
    rng.shuffle(flat)
    moved = flat.reshape(O.shape)
    perm = rng.permutation(10)
    recoloured = perm[O]
    return moved, recoloured


def depends_on_O(derive, pairs, rng):
    """The share of pairs whose route changes when the *test's* answer is disturbed.

    Only the test's. A maker that takes the episode's demonstrations reads their
    outputs legitimately -- that is where the rule's colour convention lives, and
    reading it there is what a solver does. What it must not do is take anything
    from the answer to the pair it is deriving, so that is the grid disturbed and
    the demonstrations are handed over untouched.
    """
    tried = changed = 0
    for i, (I, O) in enumerate(pairs):
        shown = [(a.tolist(), b.tolist())
                 for j, (a, b) in enumerate(pairs) if j != i][:3]
        try:
            base = call_derive(derive, I.tolist(), O.tolist(), shown)
        except Exception:
            continue
        tried += 1
        for alt in scrambles(O, rng):
            try:
                if call_derive(derive, I.tolist(), alt.tolist(), shown) != base:
                    changed += 1
                    break
            except Exception:
                changed += 1
                break
    return changed / tried if tried else 0.0


def copy_pairs(pairs):
    """(I, O_other) pairs where landing on O_other would be decisive.

    Only two instances of the same shapes can be one: a route that reads O has
    to be *able* to reach the other output for reaching it to mean anything.
    """
    buckets = collections.defaultdict(list)
    for I, O in pairs:
        buckets[(I.shape, O.shape)].append((I, O))
    out = []
    for group in buckets.values():
        for k in range(0, len(group) - 1, 2):
            (I1, O1), (_, O2) = group[k], group[k + 1]
            if not np.array_equal(O1, O2):
                out.append((I1, O2))
    return out


def zero_rate(derive, pairs):
    """Solved, over the drawn instances whose target carries a 0.

    Paste, CopyI and CopyO treat 0 as nothing there, so a maker that replicates
    a region drops whatever role the palette gave 0 to. The palette gives it
    only sometimes, which is why the miss survived a release: seven makers
    solve every instance without a 0 in the target and as little as none of the
    ones with. Measured on drawn instances rather than on episodes, because a
    rollout keeps only what solved.
    """
    solved = seen = 0
    for i, (I, O) in enumerate(pairs):
        if 0 not in np.unique(O):
            continue
        seen += 1
        shown = [(a.tolist(), b.tolist())
                 for j, (a, b) in enumerate(pairs) if j != i][:3]
        g, _, _ = replay(derive, I, O, shown)
        solved += g is not None and g.shape == O.shape and bool((g == O).all())
    return solved, seen


def score(task: str, maker_path: Path, pairs, cpairs, want: set[int],
          eps=None, ablate: int = 3, ablate_max_ops: int = 60) -> dict:
    """solve / route / copy for one candidate on a fixed set of instances.

    `eps` are episodes -- (demonstrations, test) -- and when they are given the
    test of each is what gets solved, with its own demonstrations handed over.
    Falling back to loose pairs keeps the older measures working, but a maker
    that reads the demonstrations has to be judged on real ones.
    """
    derive = load_derive(maker_path)
    if derive is None:
        return {"loaded": False}
    # Held before `eps` replaces them. Episodes come out of a rollout, which
    # dropped whatever failed, so an instance a maker cannot solve is missing
    # from them by construction -- exactly the instances the zero rate is for.
    drawn = list(pairs)
    if eps:
        pairs = [t for _, t in eps]
        shows = [[(a.tolist(), b.tolist()) for a, b in ex] for ex, _ in eps]
    else:
        shows = None
    n = len(pairs)
    solved = routed = idle = 0
    # Ablation costs one replay per operation, so it is asked of the first few
    # solved instances rather than all of them, and of routes short enough that
    # the bill stays flat. A route past the cap reports None: unmeasured, not
    # clean.
    spare_hits = spare_seen = 0
    spare_which = None
    lens = []
    missed = []
    for i, (I, O) in enumerate(pairs):
        shown = shows[i] if shows else [
            (a.tolist(), b.tolist()) for j, (a, b) in enumerate(pairs) if j != i][:3]
        g, ops, sels = replay(derive, I, O, shown)
        if g is not None and g.shape == O.shape and bool((g == O).all()):
            solved += 1
            lens.append(len(ops))
            if want and set(ops) & want:
                routed += 1
            if cancels(ops, sels):
                idle += 1
            if spare_seen < ablate and len(ops) <= ablate_max_ops:
                spare_seen += 1
                found = spare_ops(I, O, ops, sels)
                spare_hits += int(bool(found))
                if found and spare_which is None:
                    spare_which = [f"the {n} at position {i} of {len(ops)}"
                                   for i, n in found]
        else:
            missed.append(i)
    zsolved, zseen = zero_rate(derive, drawn)
    dep = depends_on_O(derive, pairs[:8], np.random.default_rng(0))
    copied = 0
    for I, P in cpairs:
        g, _, _ = replay(derive, I, P)
        if g is not None and g.shape == P.shape and bool((g == P).all()):
            copied += 1
    return {"loaded": True, "n": n, "_derive": derive,
            "solve": solved / n if n else 0.0,
            "route": routed / solved if solved else 0.0,
            "copy": copied / len(cpairs) if cpairs else 0.0,
            "idle": idle / solved if solved else 0.0,
            "spare": spare_hits / spare_seen if spare_seen else None,
            "zero": zsolved / zseen if zseen else None, "zero_seen": zseen,
            "spare_seen": spare_seen, "spare_which": spare_which,
            # Reported, never a condition. What a route costs is the operations
            # in it that do nothing, which `idle` and `spare` name directly; a
            # length made into a bar is a length to write towards.
            "ops": sum(lens) / len(lens) if lens else None,
            "dep": dep,
            "solved": solved, "routed": routed, "idled": idle, "missed": missed,
            "copied": copied, "trials": len(cpairs),
            "measured_route": bool(want)}


def regenerate_finding(task: str, concept: str | None, cur: dict,
                       scores: dict | None = None, pairs=None) -> dict:
    """What to tell the generator when nothing on hand is good enough.

    Both facts in one finding on purpose. Sent separately they were answered
    separately: told the direction was missing, the next version put a Flip in
    and took its parameters off O; told it copied, the version after that
    stopped copying by going back to painting.

    Where the incumbent copies, its own numbers describe none of the trouble --
    it solves everything, by reading. The instance that actually defeated the
    honest attempts is in the best of them, so that is what gets shown.
    """
    spare_text = (
        f"On {cur['spare']:.0%} of the instances looked at, the grid your "
        f"operations produce is exactly the same when one of them is left out — "
        + "; ".join(cur.get("spare_which") or []) + ". Nothing in the answer "
        f"depends on it: it is written, and then the route arrives where it "
        f"would have arrived anyway. Every operation has to be one the answer "
        f"needs.") if (cur.get("spare") or 0.0) >= SPARE_MIN else ""
    if spare_text and not cur["copy"] and not cur.get("dep") and cur["solve"] >= 1:
        # Nothing else is wrong with this one. Sending the whole lecture with it
        # asks a maker that already derives from I to stop reading O, and it
        # will find something to change; one fault, one instruction, and the
        # rest of the route is to come back as it was.
        return {"code": "SPARE_OPERATION", "severity": "medium",
                "evidence": spare_text + " Removing it is the whole change: the "
                            "rest of the route is right and should come back "
                            "unaltered."}
    ev = [f"Of {cur['n']} instances drawn from this task's own generator and "
          f"vouched for by its verifier, your derive_operations solves "
          f"{cur['solved']}."]
    if cur["copy"] > 0:
        ev.append(
            f"On {cur['copied']} of {cur['trials']} pairs where you were handed "
            f"another instance's output, your operations drew it exactly. That "
            f"output is not the answer to the input you were given, so the route "
            f"cannot have measured anything from the input: it transcribed O.")
    ev.append(
        "One thing is being asked for: take every parameter of the route — which "
        "region, which colour, which axis, how far — from I. O is what you check "
        "against at the end, not where the plan comes from. A fallback branch "
        "that paints O when the rule does not fit is the transcription, however "
        "rarely it runs. Nothing is being asked about which operations you use: "
        "whatever route derives from I is the right one, and a route that paints "
        "cells is not worse than one that turns them.")
    if spare_text:
        ev.append(spare_text)
    ev += _honest_attempt_evidence(scores, pairs)
    return {"code": "ANSWER_COPIED_FROM_O", "severity": "high",
            "evidence": "\n\n".join(ev)}


def _honest_attempt_evidence(scores, pairs) -> list:
    """An instance that beat the best version which does not read O."""
    if not scores or not pairs:
        return []
    clean = [(k, v) for k, v in scores.items()
             if v.get("loaded") and v["copy"] <= 1e-9 and v["missed"]]
    if not clean:
        return []
    _, best = max(clean, key=lambda kv: kv[1]["solve"])
    I, O = pairs[best["missed"][0]]
    derive = best.get("_derive")
    got = None
    if derive is not None:
        got, _, _ = replay(derive, I, O)
    out = ["A version of this maker that does not read O has already been "
           "written, and it fails on instances like the one below — which is "
           "the part still to be solved, not the copying.",
           "the input:", grid_str(I), "the answer:", grid_str(O)]
    if got is not None:
        out += ["what that version produced instead:", grid_str(got)]
    return out


def pick(scores: dict, incumbent: str, order: list,
         spare_min: float = SPARE_MIN) -> tuple[str | None, str]:
    """The candidate that carries the concept without reading the answer.

    An earlier draft ordered these as two tie-breaks -- fewest copies, then most
    route -- and on the first fifteen tasks it reverted eleven of them to the
    ancestor, giving back every bit of direction the round had won, because the
    ancestor copied nothing by virtue of doing nothing. The two are not ranked
    against each other. Both are required, and where no candidate has both the
    honest answer is that none of them is good enough yet: say so, and let the
    finding go back to the generator.
    """
    def spare(v):
        """The spare rate, on the scale where it counts as a fault.

        Deleting an operation and still landing on the answer is decisive about
        the instance and not about the route: across the released draw, 38 of
        400 tasks have some operation their route reaches the answer without,
        and in all but four of them it is one or two episodes in five -- a
        colour laid on cells that instance already had that colour, needed by
        every other instance. What a majority means is that the route emits it
        whatever it is given, which is the maker's doing. Below that it is the
        draw's, and rejecting on it would hold a tenth of the set for repairs
        that are not there.
        """
        r = v.get("spare")
        return 0.0 if r is None else float(r)

    live = {k: v for k, v in scores.items() if v.get("loaded")}
    if not live:
        return None, "no candidate loaded"
    base = live.get(incumbent)
    r0 = base["route"] if base else 0.0
    s0 = base["solve"] if base else 0.0
    # No slack here. An earlier draft allowed a candidate one unsolved instance
    # of the best, on the reasoning that one instance in thirty is noise, and it
    # promoted an ancestor that had 29/30 at the seed it was measured on and 6/8
    # at the next seed tried: the instance it missed was not noise, it was the
    # part of the task that version never handled. Noise is answered by drawing
    # more instances from more seeds, which `--rand_seed` now takes, not by
    # forgiving a miss.
    if base and (base["copy"] > 0 or base["idle"] > 0):
        # The incumbent's coverage is not a bar the honest candidates have to
        # clear. a48eeaf7 solved 45 of 45 and two of those it solved by drawing
        # the answer: every version that derives from I alone misses the same
        # two. Measuring them against 45 rejects them for the very instances the
        # incumbent cannot do either. The same holds when the incumbent's
        # geometry cancels: 6855a6e4 reaches every target and moves an object
        # down and back up again to do it, and the version that does not solves
        # nine per cent fewer. Coverage bought that way is not coverage to
        # defend.
        clean = [v["solve"] for v in live.values()
                 if v["copy"] <= 1e-9 and v["idle"] <= 1e-9
                 and spare(v) < spare_min]
        if clean:
            s0 = max(clean)
    ok = {k: v for k, v in live.items() if v["solve"] >= s0 - 1e-9}
    # `route` is not a bar to clear. A mirror and its undo scores 1.00 on it
    # while performing nothing, and this rule used to require route to go up:
    # four of the six routes that cancel out came from the two rounds that
    # asked for it. Counting whether the operation appears is satisfiable
    # without doing anything, so route is a preference among candidates that
    # have already passed, never a condition on its own.
    idle0 = base["idle"] if base else 0.0
    dep0 = base["dep"] if base else 1.0
    # `spare` is what `idle` cannot see. A mirror and its undo compose to the
    # identity and neither one can be deleted, which is why the algebra above
    # exists; an operation the route reaches O without is the other shape of the
    # same fault, and only deleting it shows that. Neither test subsumes the
    # other, so both are conditions. Where the route was too long to ablate the
    # rate is None, and an unmeasured axis rejects nobody.
    good = {k: v for k, v in ok.items()
            if v["copy"] <= 1e-9 and v["idle"] <= idle0 + 1e-9 and v["idle"] <= 1e-9
            and spare(v) < spare_min
            and v["dep"] <= dep0 + 1e-9}
    if not good:
        # A candidate whose geometry cancels is not an improvement even when
        # the incumbent's does too; if nothing is clean, say so.
        good = {k: v for k, v in ok.items()
                if v["copy"] <= 1e-9 and v["idle"] <= idle0 - 1e-9
                and spare(v) < spare_min}
    if not good:
        return None, ("no candidate keeps its parameters off O and performs the "
                      "geometry it contains")
    # Coverage first among the candidates that passed: solving more of what the
    # generator draws is a property of the maker, where route only counts
    # whether an operation appears. 0a938d79 was kept over a version that
    # solves every instance because it scored higher on route, which is the
    # wrong way round.
    # A route that does not read the answer is preferred to one that does, and
    # ahead of coverage: thirty makers were regenerated to derive from the input
    # alone and every one of them had to be applied by hand, because nothing in
    # this rule could see the difference. Their operations got shorter doing it
    # -- d364b489 from 374 to 5 -- so it is not a cost being traded away.
    dp = min(v["dep"] for v in good.values())
    good = {k: v for k, v in good.items() if v["dep"] <= dp + 1e-9}
    # Coverage of the instances where 0 is a colour of the picture, ahead of
    # coverage in general because `solve` cannot see it. With --episodes_root
    # the solved set is the rollout's, and the rollout kept only what solved:
    # 2bcee788 stands at solve 1.00 there and answers none of the instances
    # whose target holds a 0. Only a real gap counts -- a rate is a handful of
    # instances and a few points of difference is noise -- and a candidate that
    # has no such instances to answer neither wins nor loses by them. The margin
    # is wide on purpose: e21d9049's repair went from answering 14% of them to
    # 26%, which on twenty-odd instances is three more, and swapping the release
    # for that is churn. It stays on the list to be regenerated instead.
    zs = [v["zero"] for v in good.values() if v.get("zero") is not None]
    if zs:
        zv = max(zs)
        if base is None or base.get("zero") is None or zv > base["zero"] + 0.15:
            good = {k: v for k, v in good.items()
                    if v.get("zero") is None or v["zero"] >= zv - 1e-9}
    sv = max(v["solve"] for v in good.values())
    good = {k: v for k, v in good.items() if v["solve"] >= sv - 1e-9}
    # Nothing after coverage. `route` used to be the last filter, and on the six
    # tasks repinned in 628fda7 it was the only one that separated anything: the
    # candidate tied its incumbent on every measured axis and contained the
    # operation family the verifier's concept names, so a transpose spelled with
    # two flips beat a crop, and a version that had to insert geometry to score
    # here replaced one that did not need it. Counting whether an operation
    # appears is satisfiable by appearing. It is reported and it decides nothing.
    # Candidates that reach here are indistinguishable on everything measurable,
    # and where the incumbent is among them it stays: a promotion is earned on an
    # axis or it does not happen. Route length is not a tie-break either -- made
    # one, the shortest route becomes the thing to write towards.
    top = good
    # A spare operation in the incumbent is a repair, not a reason to swap.
    # 995c5fa3 grows its canvas before painting and then resizes it to the
    # answer, so the first resize can go -- and the only candidate without that
    # fault was the one taken out in 628fda7 for spelling a transpose with two
    # flips. Cleanliness on the single axis the incumbent fails does not make a
    # candidate better; what it is worse at is what nothing here measures. So
    # where the incumbent would have been kept but for its spare operations,
    # nothing is promoted and the task goes back to be regenerated.
    if (incumbent not in top and base is not None
            and spare(base) >= spare_min
            and base["copy"] <= 1e-9 and base["idle"] <= 1e-9
            and base["dep"] <= dp + 1e-9 and base["solve"] >= sv - 1e-9):
        # Here `route` earns its keep, as a veto rather than a prize. A
        # candidate that drops the spare operation and picks up nothing else is
        # the repair the finding asked for -- 995c5fa3's regeneration sizes its
        # canvas once and stands at the incumbent's route. A candidate that
        # wins this axis while acquiring the operation family the incumbent
        # never had is the shape of 628fda7, and being clean of one marginal
        # fault does not buy it. Preferring a lower route is not something a
        # maker can be written towards; the ones that could win by it are the
        # ones already keeping the route they had.
        repair = {k: v for k, v in top.items()
                  if v["route"] <= base["route"] + 1e-9}
        if not repair:
            return None, ("the incumbent reaches the answer without one of its "
                          "own operations, and the only candidates without that "
                          "fault take on operations it never had; this is a "
                          "repair, not a swap")
        top = repair
    if incumbent in top:
        return incumbent, "already the best of the candidates; kept"
    # Several candidates can be indistinguishable on all of it. Break it by the
    # order they were named on the command line rather than by their names,
    # so which lineage wins a tie is something the caller states.
    k = next(c for c in order if c in top)
    why = []
    if base is None:
        why.append("nothing was in place")
    else:
        if base["copy"] > 1e-9:
            why.append(f"the incumbent draws another instance's output on "
                       f"{base['copy']:.0%} of the pairs it was handed")
        if base["dep"] > 1e-9:
            why.append(f"the incumbent's route changes on {base['dep']:.0%} of pairs "
                       f"when the answer is disturbed, so it is reading it")
        if base["idle"] > 1e-9:
            why.append(f"the incumbent's geometry cancels out on "
                       f"{base['idle']:.0%} of its solutions")
        if spare(base) >= spare_min:
            why.append(f"the incumbent reaches the answer without one of its "
                       f"own operations on {spare(base):.0%} of the instances "
                       f"looked at")
        if base["solve"] < s0 - 1e-9:
            why.append(f"the incumbent solves {base['solve']:.0%} against {s0:.0%}")
        bz, gz = base.get("zero"), top[k].get("zero")
        if bz is not None and gz is not None and gz > bz + 1e-9:
            why.append(f"the incumbent answers {bz:.0%} of the instances where 0 "
                       f"is a colour of the picture, against {gz:.0%}")
    lead = "; ".join(why) or "it is preferred on the concept"
    got = top[k]
    return k, (f"{lead}. This one solves, never draws another instance's output, "
               f"performs the geometry it contains, keeps no operation the route "
               f"reaches the answer without, and does not change its route when "
               f"the answer is disturbed ({got['dep']:.0%})"
               + (f"; it carries the concept on {got['route']:.0%} of its "
                  f"solutions, which is reported and was not why it won"
                  if got.get("measured_route") else ""))


def keep(root: Path, incumbent: str, task: str, rec: dict, stamp: str):
    """Put the version about to be overwritten somewhere it can be read back.

    Six routes were replaced by worse ones and the replacement was a `rmtree`,
    so what had been there was recoverable only from a draw made before it. The
    copy costs a few kilobytes and the ledger beside it says, per task, what was
    swapped for what and on which numbers.
    """
    dst = root / incumbent / task
    if not dst.exists():
        return None
    held = root / ".promoted" / stamp / task
    held.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dst, held, dirs_exist_ok=True)
    return str(held.relative_to(root.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="maker set names under maker/, incumbent first")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--rand_seed", type=int, nargs="+", default=[0, 4242, 90210],
                    help="instances are pooled over these seeds")
    ap.add_argument("--rearc_root", default=str(SOLAR_ROOT / "re-arc"))
    ap.add_argument("--episodes_root", default=None,
                    help="a rollout directory whose episodes supply each test's "
                         "demonstrations; needed to judge a maker that reads them")
    ap.add_argument("--out", default="choose_maker.json")
    ap.add_argument("--findings", default=None,
                    help="write the tasks with no acceptable candidate as "
                         "critique records, for critique_to_feedback.py")
    ap.add_argument("--spare_min", type=float, default=SPARE_MIN,
                    help="share of ablated instances that must give up an "
                         "operation before it counts against a route")
    ap.add_argument("--ablate", type=int, default=3,
                    help="solved instances per candidate to delete operations "
                         "from; 0 turns the check off")
    ap.add_argument("--ablate_max_ops", type=int, default=60,
                    help="routes longer than this are reported unmeasured")
    ap.add_argument("--apply", action="store_true",
                    help="copy each winner into the incumbent set")
    args = ap.parse_args()

    incumbent = args.candidates[0]
    root = SOLAR_ROOT / "maker"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    vconcepts = verifier_concepts(Path(args.rearc_root))

    tasks = args.tasks
    if tasks is None:
        tasks = sorted(p.name for p in (root / incumbent).iterdir()
                       if (p / "grid_maker.py").exists())

    recs = []
    for t in tasks:
        pairs = draw(t, args.num_samples, args.rand_seed)
        if not pairs:
            print(f"{t}  no verifier-approved instances", flush=True)
            continue
        cname, want = concept_ops(t, vconcepts)
        cpairs = copy_pairs(pairs)
        scores = {}
        for cand in args.candidates:
            p = root / cand / t / "grid_maker.py"
            if not p.exists():
                scores[cand] = {"loaded": False}
                continue
            # Episodes are drawn per candidate: the palette a maker asks for is
            # its own, and its demonstrations have to be the ones it would be
            # shown rather than another maker's.
            eps = (episodes_from_draw(args.episodes_root, t)
                   if args.episodes_root else None)
            if args.episodes_root and eps is None:
                raise SystemExit(
                    f"--episodes_root {args.episodes_root} holds no episodes for "
                    f"{t}. Judging would fall back to demonstrations built from "
                    f"other drawn pairs, which is what this flag exists to avoid.")
            scores[cand] = score(t, p, pairs, cpairs, want, eps=eps,
                                 ablate=args.ablate,
                                 ablate_max_ops=args.ablate_max_ops)
        win, why = pick(scores, incumbent, args.candidates, args.spare_min)
        rec = {"task_id": t, "concept": cname, "instances": len(pairs),
               "winner": win, "reason": why,
               "changed": win is not None and win != incumbent,
               "verdict": "PASS" if win else "REVISE", "findings": [],
               "scores": scores}
        if win is None and scores.get(incumbent, {}).get("loaded"):
            rec["findings"].append(
                regenerate_finding(t, cname, scores[incumbent], scores, pairs))
        recs.append(rec)
        s = "  ".join(
            f"{c}:{scores[c]['solve']:.2f}/{scores[c]['route']:.2f}/"
            f"{scores[c]['copy']:.2f}/{scores[c]['idle']:.2f}/{scores[c]['dep']:.2f}/"
            + ("-" if scores[c].get("spare") is None else f"{scores[c]['spare']:.2f}")
            + "/" + ("-" if scores[c].get("zero") is None else f"{scores[c]['zero']:.2f}")
            if scores[c].get("loaded") else f"{c}:-" for c in args.candidates)
        print(f"{t}  {s}   -> {win or 'none of them'}", flush=True)
        if args.apply and win is not None and win != incumbent:
            rec["replaced"] = keep(root, incumbent, t, rec, stamp)
            src, dst = root / win / t, root / incumbent / t
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)

    for r in recs:
        for v in r["scores"].values():
            v.pop("_derive", None)
    if args.apply:
        moved = [r for r in recs if r.get("replaced")]
        if moved:
            led = root / ".promoted" / stamp / "promotions.json"
            led.parent.mkdir(parents=True, exist_ok=True)
            axes = ('solve', 'route', 'copy', 'idle', 'spare', 'dep', 'ops')
            led.write_text(json.dumps(
                [{"task_id": r["task_id"], "into": incumbent,
                  "from": r["winner"], "reason": r["reason"],
                  "held": r["replaced"],
                  "before": {a: r["scores"][incumbent].get(a) for a in axes},
                  "after": {a: r["scores"][r["winner"]].get(a) for a in axes}}
                 for r in moved], indent=1))
            print(f"  {len(moved)} replaced versions held under {led.parent}")
    Path(args.out).write_text(json.dumps(recs, indent=1))
    ch = [r for r in recs if r["changed"]]
    rg = [r for r in recs if r["winner"] is None]
    print(f"\n{len(recs)} tasks | {len(ch)} would change"
          + (" (applied)" if args.apply else "")
          + f" | {len(rg)} have no acceptable candidate")
    for r in ch:
        print(f"  {r['task_id']}  -> {r['winner']}   {r['reason']}")
    if args.findings:
        Path(args.findings).write_text(json.dumps(
            [{k: r[k] for k in ("task_id", "verdict", "findings")}
             for r in recs if r["findings"]], indent=1))
        print(f"  findings for {len(rg)} tasks -> {args.findings}")


if __name__ == "__main__":
    main()
