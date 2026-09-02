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
and each gives up three rates:

    solve   the ops, replayed on I, reach O
    route   the ops contain the operation family the verifier's concept names
    copy    handed a different instance's output, the ops reach it anyway

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
    hits = _glob.glob(f"{root}/whole/test.{task}.*")
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
    """
    out = []
    for sd in seeds:
        got = _draw_one(task, n, sd)
        if got is None:
            return None
        out += got
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


def score(task: str, maker_path: Path, pairs, cpairs, want: set[int],
          eps=None) -> dict:
    """solve / route / copy for one candidate on a fixed set of instances.

    `eps` are episodes -- (demonstrations, test) -- and when they are given the
    test of each is what gets solved, with its own demonstrations handed over.
    Falling back to loose pairs keeps the older measures working, but a maker
    that reads the demonstrations has to be judged on real ones.
    """
    derive = load_derive(maker_path)
    if derive is None:
        return {"loaded": False}
    if eps:
        pairs = [t for _, t in eps]
        shows = [[(a.tolist(), b.tolist()) for a, b in ex] for ex, _ in eps]
    else:
        shows = None
    n = len(pairs)
    solved = routed = idle = 0
    missed = []
    for i, (I, O) in enumerate(pairs):
        shown = shows[i] if shows else [
            (a.tolist(), b.tolist()) for j, (a, b) in enumerate(pairs) if j != i][:3]
        g, ops, sels = replay(derive, I, O, shown)
        if g is not None and g.shape == O.shape and bool((g == O).all()):
            solved += 1
            if want and set(ops) & want:
                routed += 1
            if cancels(ops, sels):
                idle += 1
        else:
            missed.append(i)
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


def pick(scores: dict, incumbent: str, order: list) -> tuple[str | None, str]:
    """The candidate that carries the concept without reading the answer.

    An earlier draft ordered these as two tie-breaks -- fewest copies, then most
    route -- and on the first fifteen tasks it reverted eleven of them to the
    ancestor, giving back every bit of direction the round had won, because the
    ancestor copied nothing by virtue of doing nothing. The two are not ranked
    against each other. Both are required, and where no candidate has both the
    honest answer is that none of them is good enough yet: say so, and let the
    finding go back to the generator.
    """
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
                 if v["copy"] <= 1e-9 and v["idle"] <= 1e-9]
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
    good = {k: v for k, v in ok.items()
            if v["copy"] <= 1e-9 and v["idle"] <= idle0 + 1e-9 and v["idle"] <= 1e-9
            and v["dep"] <= dep0 + 1e-9}
    if not good:
        # A candidate whose geometry cancels is not an improvement even when
        # the incumbent's does too; if nothing is clean, say so.
        good = {k: v for k, v in ok.items()
                if v["copy"] <= 1e-9 and v["idle"] <= idle0 - 1e-9}
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
    sv = max(v["solve"] for v in good.values())
    good = {k: v for k, v in good.items() if v["solve"] >= sv - 1e-9}
    hi = max(v["route"] for v in good.values())
    top = {k: v for k, v in good.items() if v["route"] >= hi - 1e-9}
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
        if base["solve"] < s0 - 1e-9:
            why.append(f"the incumbent solves {base['solve']:.0%} against {s0:.0%}")
    lead = "; ".join(why) or "it is preferred on the concept"
    got = top[k]
    return k, (f"{lead}. This one solves, never draws another instance's output, "
               f"performs the geometry it contains, does not change its route "
               f"when the answer is disturbed ({got['dep']:.0%}), and carries the "
               f"concept on {hi:.0%} of its solutions")


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
    ap.add_argument("--apply", action="store_true",
                    help="copy each winner into the incumbent set")
    args = ap.parse_args()

    incumbent = args.candidates[0]
    root = SOLAR_ROOT / "maker"
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
            scores[cand] = score(t, p, pairs, cpairs, want, eps=eps)
        win, why = pick(scores, incumbent, args.candidates)
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
            f"{scores[c]['copy']:.2f}/{scores[c]['idle']:.2f}/{scores[c]['dep']:.2f}"
            if scores[c].get("loaded") else f"{c}:-" for c in args.candidates)
        print(f"{t}  {s}   -> {win or 'none of them'}", flush=True)
        if args.apply and win is not None and win != incumbent:
            src, dst = root / win / t, root / incumbent / t
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)

    for r in recs:
        for v in r["scores"].values():
            v.pop("_derive", None)
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
