"""
ARC Task: 05f2a901 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


# ── episode-level colours ─────────────────────────────────────────────────────
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc, destc = random.sample(cols, 3)
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


# ── generator ─────────────────────────────────────────────────────────────────
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, fgc=1, destc=2) -> dict:
    H = max(8, min(30, max_h))
    W = max(8, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (8, H))
    w = _unifint(diff_lb, diff_ub, (8, W))
    objh = _unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = _unifint(diff_lb, diff_ub, (objh, w // 2))
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = random.choice(sorted(bb))
    obj = {sp}
    bb.discard(sp)
    ncells = _unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for _ in range(ncells - 1):
        nb = set()
        for (r, c) in obj:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb.add((r + dr, c + dc))
        cand = sorted((bb - obj) & nb)
        if not cand:
            break
        obj.add(random.choice(cand))
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(obj):
        obj.discard(random.choice(sorted(obj)))
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    r0, c0 = min(rs), min(cs)
    obj = {(r - r0, c - c0) for r, c in obj}
    objh = max(rs) - r0 + 1
    objw = max(cs) - c0 + 1
    loci = _unifint(diff_lb, diff_ub, (3, h - objh))
    locj = _unifint(diff_lb, diff_ub, (0, w - objw))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    obj = {(r + loci, c + locj) for r, c in obj}
    for r, c in obj:
        gi[r, c] = fgc
    sqd = random.randint(1, min(w, loci - 1))
    locisq = random.randint(0, loci - sqd - 1)
    locjsq = random.randint(locj - sqd + 1, locj + objw - 1)
    sq = {(r, c) for r in range(locisq, locisq + sqd)
          for c in range(locjsq, locjsq + sqd) if 0 <= r < h and 0 <= c < w}
    for r, c in sq:
        gi[r, c] = destc
        go[r, c] = destc
    guard = 0
    while len(obj & sq) == 0 and guard < 100:
        obj = {(r - 1, c) for r, c in obj}
        guard += 1
    obj = {(r + 1, c) for r, c in obj}
    for r, c in obj:
        if 0 <= r < h and 0 <= c < w:
            go[r, c] = fgc
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


# ── operations ────────────────────────────────────────────────────────────────
# Reading check against the code-golf entry: it rotates the grid and deletes the
# blank lines between the anchor and the shape, so everything past the gap
# slides toward the anchor. Here, the only thing past the gap is the ragged
# shape, so that reading still comes down to "move the shape toward the solid
# rectangle". It treats the same thing as one unit, works in the same order and
# holds the same quantity fixed (the anchor stays put), so it agrees with the
# maker. Its "delete the gap lines" step is wrong for RE-ARC, though. A shape
# can slide past the anchor's bounding edge until its cells touch (example 3).
# Moving a full-width band would also paste background over the anchor.
# So the shape is grabbed by its own cells and slid until the next step would hit.

def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    by_col = {}
    for r in range(hi):
        for c in range(wi):
            by_col.setdefault(int(I[r, c]), set()).add((r, c))

    # anchor = the colour that exactly fills its bounding box (the solid square)
    solid, ragged = [], []
    for col, cells in by_col.items():
        r0, r1, c0, c1 = _bbox(cells)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(cells) == area and area < hi * wi:
            solid.append((col, cells))
        else:
            ragged.append((col, cells))
    if len(solid) != 1 or len(ragged) != 2:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    rect = solid[0][1]

    # background spans the grid; the shape fits in a half-size box
    def bbarea(cells):
        r0, r1, c0, c1 = _bbox(cells)
        return (r1 - r0 + 1) * (c1 - c0 + 1)
    ragged.sort(key=lambda t: bbarea(t[1]))
    obj_col, obj = ragged[0]
    bgc = ragged[1][0]

    orr0, orr1, occ0, occ1 = _bbox(obj)
    rr0, rr1, rc0, rc1 = _bbox(rect)
    if orr0 > rr1:
        dr, dc = -1, 0
    elif orr1 < rr0:
        dr, dc = 1, 0
    elif occ0 > rc1:
        dr, dc = 0, -1
    elif occ1 < rc0:
        dr, dc = 0, 1
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # slide until the next step would put a shape cell onto the rectangle
    cur = set(obj)
    steps = 0
    while steps < 60:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & rect:
            break
        if not all(0 <= r < hi and 0 <= c < wi for r, c in nxt):
            break
        cur = nxt
        steps += 1

    if steps > 0:
        if obj_col != 0:
            mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
            ops.append(mv); sels.append(sel_of(sorted(obj)))   # grab the shape
            for _ in range(steps - 1):
                ops.append(mv); sels.append(sel_of([]))        # keep sliding it
            hole = sorted(set(obj) - cur)                      # vacated footprint only
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # colour-0 shape: Move would not carry its 0 cells, so lift and set down
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))
            ops.append(0); sels.append(sel_of(sorted(cur)))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])        # full grid: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
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
                                f"for task 05f2a901"
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
                    f"Failed to build a complete episode for task 05f2a901 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05f2a901-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
