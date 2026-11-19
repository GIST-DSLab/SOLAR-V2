"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the two colour roles the rule depends on are fixed per episode:
    # the background and the "anchor dot" colour every fragment carries.
    # The fragment colours themselves are free (the rule only aligns dots).
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dotc is None:
        dotc = choice(remove(bgc, cols))
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
    remcols = remove(dotc, remove(bgc, cols))
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    lo = min(h, w) * 2
    if lo > max_h or lo > max_w:
        raise ValueError("grid too small")
    gih = unifint(diff_lb, diff_ub, (lo, max_h))
    giw = unifint(diff_lb, diff_ub, (lo, max_w))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    for _attempt in range(200):
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            if oh > gih or ow > giw:
                succ = False
                break
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            return {'input': gi, 'output': go}
        maxtr = int(maxtr * 1.5)
        gih = randint(gih, max_h)
        giw = randint(giw, max_w)
    raise ValueError("could not place fragments")


# ── reading the input ─────────────────────────────────────────────────────────
# Each fragment is one colour's cells together with ONE anchor dot (the dot
# colour).  The answer stacks every fragment so that all anchor dots coincide;
# the stacked fragments tile an h x w frame exactly.  (The golf entry reads the
# task the same way: "a window around each dot, all windows overlaid"; it only
# differs in never keeping any fragment in place — here one fragment is the
# meeting point and the others slide onto it, a frame-sized window around a dot
# being exactly the region the stacked fragments fill.)

def _bg(I, examples):
    tot = Counter(np.asarray(I, dtype=int).flatten().tolist())
    for e in (examples or []):
        tot.update(np.asarray(e[0], dtype=int).flatten().tolist())
    return tot.most_common(1)[0][0]


def _dot_candidates(g, bgc):
    # a dot colour appears exactly once per fragment, i.e. as often as there
    # are other foreground colours
    others = [c for c in np.unique(g).tolist() if c != bgc]
    return [c for c in others if int((g == c).sum()) == len(others) - 1]


def _fragments(I, bgc, dotc):
    dots = [tuple(p) for p in np.argwhere(I == dotc).tolist()]
    colors = [c for c in np.unique(I).tolist() if c not in (bgc, dotc)]
    cells = {c: [tuple(p) for p in np.argwhere(I == c).tolist()] for c in colors}

    def dist(c, d):
        return min(abs(r - d[0]) + abs(q - d[1]) for r, q in cells[c])
    prefs = {c: sorted(range(len(dots)), key=lambda i: (dist(c, dots[i]), i)) for c in colors}
    order = sorted(colors, key=lambda c: dist(c, dots[prefs[c][0]]))

    def frame_ok(assign):
        rel = set()
        for c, i in assign.items():
            d = dots[i]
            for r, q in cells[c]:
                o = (r - d[0], q - d[1])
                if o in rel or o == (0, 0):
                    return False
                rel.add(o)
        rel.add((0, 0))
        rs = [o[0] for o in rel]; qs = [o[1] for o in rel]
        hh = max(rs) - min(rs) + 1; ww = max(qs) - min(qs) + 1
        return hh <= 4 and ww <= 4 and len(rel) == hh * ww

    best = {}

    def bt(k, assign, used):
        if k == len(order):
            if frame_ok(assign):
                best.update(assign)
                return True
            return False
        c = order[k]
        for i in prefs[c]:
            if i in used:
                continue
            assign[c] = i
            if bt(k + 1, assign, used | {i}):
                return True
            del assign[c]
        return False

    if not (len(colors) == len(dots) and bt(0, {}, frozenset())):
        return None
    frags = []
    for c in colors:
        d = dots[best[c]]
        frags.append({"color": c, "dot": d, "body": cells[c],
                      "rel": [(r - d[0], q - d[1]) for r, q in cells[c]]})
    return frags


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc = _bg(I, examples)
    # the dot colour is the one whose dots let the other colours tile one frame;
    # it is fixed for the whole episode, so the demonstrations break any tie
    readings = []
    for c in _dot_candidates(I, bgc):
        fr = _fragments(I, bgc, c)
        if fr is None:
            continue
        votes = 0
        for e in (examples or []):
            g = np.asarray(e[0], dtype=int)
            if c in _dot_candidates(g, bgc) and _fragments(g, bgc, c) is not None:
                votes += 1
        readings.append((-votes, c, fr))
    if not readings:
        raise ValueError("137eaa0f: could not read the fragments out of the input")
    readings.sort(key=lambda t: (t[0], t[1]))
    dotc, frags = readings[0][1], readings[0][2]

    # the frame every fragment fits into, relative to its anchor dot
    rel_all = [(0, 0)] + [o for f in frags for o in f["rel"]]
    r0 = min(o[0] for o in rel_all); r1 = max(o[0] for o in rel_all)
    c0 = min(o[1] for o in rel_all); c1 = max(o[1] for o in rel_all)
    th, tw = r1 - r0 + 1, c1 - c0 + 1

    own = {}
    for k, f in enumerate(frags):
        for p in f["body"] + [f["dot"]]:
            own[p] = k

    def clean_meeting(P, keep):
        # frame around P must fit and hold nothing but fragment `keep`
        if not (0 <= P[0] + r0 and P[0] + r1 < H and 0 <= P[1] + c0 and P[1] + c1 < W):
            return False
        for r in range(P[0] + r0, P[0] + r1 + 1):
            for q in range(P[1] + c0, P[1] + c1 + 1):
                if I[r, q] != bgc and own.get((r, q), -1) != keep:
                    return False
        return True

    def cost(P):
        return sum(abs(f["dot"][0] - P[0]) + abs(f["dot"][1] - P[1]) for f in frags)

    meet, stay = None, None
    dot_cands = [(cost(f["dot"]), k) for k, f in enumerate(frags) if clean_meeting(f["dot"], k)]
    if dot_cands:
        stay = min(dot_cands)[1]
        meet = frags[stay]["dot"]
    else:
        free = [(cost((r, q)), (r, q)) for r in range(H) for q in range(W)
                if clean_meeting((r, q), -2)]
        meet = min(free)[1] if free else frags[0]["dot"]
        if not free:
            stay = 0

    used = set(np.unique(I).tolist())
    spare = next((c for c in range(1, 10) if c not in used), 1 if dotc != 1 else 2)

    ops, sels = [], []
    movers = [k for k in range(len(frags)) if k != stay]
    movers.sort(key=lambda k: (abs(frags[k]["dot"][0] - meet[0]) + abs(frags[k]["dot"][1] - meet[1]), k))
    for k in movers:
        f = frags[k]
        dr, dc = meet[0] - f["dot"][0], meet[1] - f["dot"][1]
        if dr == 0 and dc == 0:
            continue
        if f["color"] == 0:
            # 0 is "empty" to ARCLE's object ops: tint the fragment so it can be lifted
            ops.append(spare); sels.append(sel_of(f["body"]))
        grab = [p for p in f["body"] + [f["dot"]] if (I[p] != 0 or f["color"] == 0 and p != f["dot"])]
        steps = [20 if dr < 0 else 21] * abs(dr) + [22 if dc > 0 else 23] * abs(dc)
        ops.append(steps[0]); sels.append(sel_of(grab))     # grab the fragment once
        for st in steps[1:]:
            ops.append(st); sels.append(sel_of([]))          # keep sliding the same fragment
        if f["color"] == 0:
            ops.append(0); sels.append(sel_of([(r + meet[0], q + meet[1]) for r, q in f["rel"]]))

    # with a 0-coloured dot and no fragment left at the meeting point, the
    # meeting cell still shows the background: the stacked dots are 0 there
    if dotc == 0 and stay is None and bgc != 0:
        ops.append(0); sels.append(sel_of([meet]))

    # the stacked fragments fill the frame around the meeting point: that frame is the answer
    tr, tc = meet[0] + r0, meet[1] + c0
    ops.append(33); sels.append([tr, tc, th - 1, tw - 1])
    ops.append(34); sels.append([0, 0, th - 1, tw - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task 137eaa0f"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
