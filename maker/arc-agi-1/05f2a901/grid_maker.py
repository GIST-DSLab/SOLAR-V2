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
from collections import Counter, deque

import numpy as np

from maker.sel_helpers import sel_of

# Task 05f2a901: a solid rectangle (anchor) and a ragged shape sit on a plain background.
# The ragged shape slides in a straight line toward the rectangle and stops one cell
# before it would run into it. Direction is a structural variant -> planned per instance.
DIRECTIONS = ["up", "down", "left", "right"]


def sample_colors(num_examples=None) -> dict:
    fgc = random.choice(list(range(1, 10)))          # mover must be non-zero (ARCLE Move ignores 0s)
    rest = [c for c in range(10) if c != fgc]
    bgc, destc = random.sample(rest, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(DIRECTIONS):
        examples = [{"direction": d} for d in DIRECTIONS]
        examples += [{"direction": random.choice(DIRECTIONS)} for _ in range(n_ex - len(DIRECTIONS))]
        random.shuffle(examples)
    else:
        examples = [{"direction": d} for d in random.sample(DIRECTIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "destc": destc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    lb, ub = bounds
    if ub < lb:
        lb, ub = ub, ub
    a = lb + int(round((ub - lb) * diff_lb))
    b = lb + int(round((ub - lb) * diff_ub))
    a = max(lb, min(ub, a))
    b = max(lb, min(ub, b))
    if a > b:
        a, b = b, a
    return random.randint(a, b)


def _neighbors4(rc):
    r, c = rc
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _connected4(cells):
    cells = set(cells)
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for nb in _neighbors4(cur):
            if nb in cells and nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return len(seen) == len(cells)


def _attempt(diff_lb, diff_ub, max_h, max_w, bgc, fgc, destc, direction):
    if direction in ("up", "down"):
        hlim, wlim = min(30, max_h), min(30, max_w)
    else:                                   # these orientations transpose the base grid
        hlim, wlim = min(30, max_w), min(30, max_h)
    if hlim < 8 or wlim < 8:
        return None
    h = _unifint(diff_lb, diff_ub, (8, hlim))
    w = _unifint(diff_lb, diff_ub, (8, wlim))

    objh = _unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = _unifint(diff_lb, diff_ub, (objh, w // 2))

    start = (random.randrange(objh), random.randrange(objw))
    cells = {start}
    ncells = _unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for _ in range(ncells - 1):
        cands = set()
        for rc in cells:
            for nb in _neighbors4(rc):
                if 0 <= nb[0] < objh and 0 <= nb[1] < objw and nb not in cells:
                    cands.add(nb)
        if not cands:
            break
        cells.add(random.choice(sorted(cands)))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(cells):
        cells.remove(random.choice(sorted(cells)))   # never a filled rectangle itself
    if len(cells) < 2 or not _connected4(cells):
        return None
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    cells = {(r - r0, c - c0) for r, c in cells}
    objh = max(r for r, _ in cells) + 1
    objw = max(c for _, c in cells) + 1
    if objh * objw == len(cells):
        return None
    if h - objh < 3:
        return None

    loci = _unifint(diff_lb, diff_ub, (3, h - objh))
    locj = _unifint(diff_lb, diff_ub, (0, w - objw))
    obj = {(r + loci, c + locj) for r, c in cells}

    sqd_cap = min(w, loci - 1, max(1, int((h * w / 5.0) ** 0.5)))   # keep background in the majority
    if sqd_cap < 1:
        return None
    sqd = _unifint(diff_lb, diff_ub, (1, sqd_cap))
    if loci - sqd - 1 < 0:
        return None
    locisq = random.randint(0, loci - sqd - 1)
    locjsq = random.randint(locj - sqd + 1, locj + objw - 1)
    sq = {(r, c) for r in range(locisq, locisq + sqd)
          for c in range(locjsq, locjsq + sqd) if 0 <= r < h and 0 <= c < w}
    if not sq:
        return None

    # slide the shape straight up, stopping one step before it would collide with the square
    cur = set(obj)
    k = 0
    while True:
        nxt = {(r - 1, c) for r, c in cur}
        if nxt & sq:
            break
        if min(r for r, _ in nxt) < 0:
            return None
        cur = nxt
        k += 1
        if k > 40:
            return None
    if k < 1:
        return None

    # keep "slide until blocked" and "slide until adjacent" in agreement
    probe = set(obj)
    kadj = None
    for step in range(0, k + 1):
        if any((r + dr, c + dc) in sq for r, c in probe
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            kadj = step
            break
        probe = {(r - 1, c) for r, c in probe}
    if kadj != k:
        return None

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for r, c in obj:
        gi[r, c] = fgc
    for r, c in sq:
        gi[r, c] = destc
        go[r, c] = destc
    for r, c in cur:
        go[r, c] = fgc

    if Counter(gi.flatten().tolist()).most_common(1)[0][0] != bgc:
        return None
    if Counter(go.flatten().tolist()).most_common(1)[0][0] != bgc:
        return None

    if direction == "down":
        gi, go = np.flipud(gi), np.flipud(go)
    elif direction == "left":
        gi, go = gi.T.copy(), go.T.copy()
    elif direction == "right":
        gi, go = np.fliplr(gi.T).copy(), np.fliplr(go.T).copy()
    if random.random() < 0.5:                       # extra mirror that preserves the direction
        if direction in ("up", "down"):
            gi, go = np.fliplr(gi), np.fliplr(go)
        else:
            gi, go = np.flipud(gi), np.flipud(go)

    if gi.shape[0] > max_h or gi.shape[1] > max_w:
        return None
    return {"input": gi.tolist(), "output": go.tolist()}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, destc, direction=None) -> dict:
    if direction is None:
        direction = random.choice(DIRECTIONS)
    for _ in range(500):
        res = _attempt(diff_lb, diff_ub, max_h, max_w, bgc, fgc, destc, direction)
        if res is not None:
            return res
    raise ValueError("generation failed")


def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            dq = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while dq:
                cr, cc = dq.popleft()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                            seen[nr, nc] = True
                            dq.append((nr, nc))
            comps.append((int(col), set(cells)))
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    ops, sels = [], []

    # everything below is measured from I alone
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, bgc)

    filled, ragged = [], []
    for col, cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
            filled.append((col, cells))
        else:
            ragged.append((col, cells))

    if not filled or not ragged:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    rect = max(filled, key=lambda t: len(t[1]))[1]     # solid rectangle = the anchor
    obj = max(ragged, key=lambda t: len(t[1]))[1]      # ragged shape = the traveller

    orr0, orr1, occ0, occ1 = _bbox(obj)
    rr0, rr1, rc0, rc1 = _bbox(rect)

    if orr0 > rr1:                                     # anchor is above -> slide up
        dr, dc = -1, 0
    elif orr1 < rr0:                                   # anchor is below -> slide down
        dr, dc = 1, 0
    elif occ0 > rc1:                                   # anchor is left  -> slide left
        dr, dc = 0, -1
    elif occ1 < rc0:                                   # anchor is right -> slide right
        dr, dc = 0, 1
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # advance until one step before the shape would run into the rectangle
    cur = set(obj)
    steps = 0
    while steps <= 60:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & rect:
            break
        if not all(0 <= r < hi and 0 <= c < wi for r, c in nxt):
            break
        cur = nxt
        steps += 1

    if steps > 0:
        mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(mv); sels.append(sel_of(sorted(obj)))     # first Move grabs the shape
        for _ in range(steps - 1):
            ops.append(mv); sels.append(sel_of([]))          # empty selection keeps it grabbed
        hole = sorted(set(obj) - cur)                        # only the vacated footprint
        if bgc != 0 and hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])       # full-grid rectangle: submit
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
