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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

# Reading of the task (it matches the maker's reading). The code-golf entry reads
# it the same way: a turtle leaves the free end of the 3-domino along the
# domino's axis. It walks over background until a wall stops it, turns 90
# degrees either way, and walks again. A route is good if it ends inside the
# 2-domino after at most two turns. Each straight leg runs from one wall to the
# next, so each leg is painted as one Color(3) op.


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, fgc=1, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    if random.choice((True, False)):
        oh = _unifint(diff_lb, diff_ub, (5, h - 2))
        ow = _unifint(diff_lb, diff_ub, (3, w - 2))
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        hli = random.randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = _connect((loci + oh - 1, locj), (hli, locj))
        ln2 = _connect((hli, locj), (hli, locj + ow - 1))
        ln3 = _connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = _unifint(diff_lb, diff_ub, (3, h - 2))
        ow = _unifint(diff_lb, diff_ub, (3, w - 2))
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        if random.choice((True, False)):
            sp1j = random.randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = random.randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = _connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = _connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = _connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))
    for p in sp:
        gi[p] = 3
    for p in ep:
        gi[p] = 2
    for p in (bp1, bp2):
        gi[p] = fgc
    go = gi.copy()
    for p in ln1 | ln2 | ln3:
        go[p] = 3
    inds = [tuple(p) for p in np.argwhere(go == bgc)]
    namt = _unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    for p in random.sample(inds, namt):
        gi[p] = fgc
        go[p] = fgc
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: g[:, ::-1],
        lambda g: g[::-1, :],
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    # Background: the generator caps noise at half the background cells, so
    # background is always the most common colour in I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    m3 = [tuple(map(int, p)) for p in np.argwhere(I == 3)]
    set2 = {tuple(map(int, p)) for p in np.argwhere(I == 2)}

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w

    def leg(start, d):
        """Walk from start (exclusive) over background; return cells and stopper."""
        cells = []
        r, c = start[0] + d[0], start[1] + d[1]
        while inside(r, c) and I[r, c] == bgc:
            cells.append((r, c))
            r += d[0]
            c += d[1]
        return cells, ((r, c) if inside(r, c) else None)

    def walk(pos, d, turns):
        """Turtle search: run straight; on a wall turn either way (<=2 turns)."""
        cells, stop = leg(pos, d)
        if stop is None:
            return None
        if stop in set2:
            return [cells]
        if turns >= 2 or not cells or I[stop] == 3:
            return None
        end = cells[-1]
        for e in ((d[1], d[0]), (-d[1], -d[0])):
            rest = walk(end, e, turns + 1)
            if rest is not None:
                return [cells] + rest
        return None

    ops, sels = [], []
    route = None
    if len(m3) >= 2:
        vertical = len({r for r, _ in m3}) > 1
        axis = (1, 0) if vertical else (0, 1)
        for d in (axis, (-axis[0], -axis[1])):
            head = max(m3, key=lambda p: p[0] * d[0] + p[1] * d[1])
            route = walk(head, d, 0)
            if route is not None:
                break

    if route is not None:
        for cells in route:          # one Color(3) per straight leg, in walking order
            if cells:
                ops.append(3)
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
