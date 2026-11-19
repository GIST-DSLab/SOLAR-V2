"""
ARC Task: 2dd70a9a (RE-ARC) — LLM-generated grid_maker
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
import random
import heapq
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    if choice((True, False)):
        oh = unifint(diff_lb, diff_ub, (5, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        hli = randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = connect((loci + oh - 1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj + ow - 1))
        ln3 = connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = unifint(diff_lb, diff_ub, (3, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        if choice((True, False)):
            sp1j = randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))
    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    # Noise can open a second, equally legal route. Keep only instances whose
    # path is unique under the walking rule, so the episode can be learned.
    for _ in range(200):
        g = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, fgc)
        gi = np.array(g['input']); go = np.array(g['output'])
        runs = _find_runs(gi, bgc, fgc)
        if runs is None:
            continue
        test = gi.copy()
        for run in runs:
            for r, c in run:
                test[r, c] = 3
        if (test == go).all():
            return g
    return g


def _find_runs(I, bgc, fgc):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    green = [tuple(map(int, p)) for p in np.argwhere(I == 3)]
    red = {tuple(map(int, p)) for p in np.argwhere(I == 2)}
    green_vertical = len({r for r, _ in green}) > 1
    red_vertical = len({r for r, _ in red}) > 1

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w

    dirs0 = [(1, 0), (-1, 0)] if green_vertical else [(0, 1), (0, -1)]
    starts = [(r, c, d) for (r, c) in green for d in dirs0]

    def is_goal(s):
        r, c, (dr, dc) = s
        if (dr != 0) != red_vertical:          # must hit red head-on, along its axis
            return False
        return inside(r + dr, c + dc) and (r + dr, c + dc) in red

    def succ(s):
        r, c, (dr, dc) = s
        nr, nc = r + dr, c + dc
        if not inside(nr, nc):
            return
        if I[nr, nc] != fgc:
            yield (nr, nc, (dr, dc))
        else:                                   # barrier ahead -> right-angle turns
            yield (r, c, (dc, dr))
            yield (r, c, (-dc, -dr))

    # uniform-cost search: fewest turns first, then shortest walk
    best = {st: (0, 0) for st in starts}
    heap = [(0, 0, st) for st in starts]
    heapq.heapify(heap)
    parent = {}
    goal = None
    while heap:
        tn, ln, s = heapq.heappop(heap)
        if best.get(s) != (tn, ln):
            continue
        if is_goal(s):
            goal = s
            break
        for t in succ(s):
            cost = (tn + (t[2] != s[2]), ln + 1)
            if t not in best or cost < best[t]:
                best[t] = cost
                parent[t] = s
                heapq.heappush(heap, (cost[0], cost[1], t))

    if goal is None:
        return None
    path = []
    s = goal
    while s in parent:
        path.append(s)
        s = parent[s]
    path.reverse()
    # split the walk into straight runs at each turn
    runs, cur, curd = [], [], None
    for (r, c, d) in path:
        if d != curd and cur:
            runs.append(cur)
            cur = []
        curd = d
        if I[r, c] == bgc and (r, c) not in cur:
            cur.append((r, c))
    if cur:
        runs.append(cur)
    return runs


def derive_operations(I, O, examples=None):
    """
    Path finding (reference reading): start on the green (3) domino, heading
    along its own axis. Walk straight over open cells. You may turn 90 degrees
    only when the next cell is a barrier pixel. Stop when the next cell is
    red (2), reached head-on along the red domino's own axis.
    A search over (cell, heading), fewest turns first and then shortest,
    finds this path. Each straight run is painted green with one Color3 op,
    in walking order.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background = the colour the rule paints green (read from the demonstrations)
    bgc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                m = (ei != eo) & (eo == 3)
                cnt.update(ei[m].tolist())
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    present = set(I.flatten().tolist())
    if bgc is None or bgc not in present:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    others = [c for c in present if c not in (bgc, 2, 3)]
    fgc = others[0] if others else -1          # barrier colour

    runs = _find_runs(I, bgc, fgc)

    ops, sels = [], []
    if runs is not None:
        painted = set()
        for run in runs:
            run = [p for p in run if p not in painted]
            if run:
                ops.append(3)                   # paint this straight leg green
                sels.append(sel_of(run))
                painted.update(run)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
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
                                f"for task 2dd70a9a"
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
                    f"Failed to build a complete episode for task 2dd70a9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2dd70a9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
