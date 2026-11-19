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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fgc, destc = random.sample(range(10), 3)
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    v = int(round(a + (b - a) * d + random.uniform(-0.5, 0.5)))
    return max(a, min(b, v))


def _dneigh(c):
    r, k = c
    return {(r - 1, k), (r + 1, k), (r, k - 1), (r, k + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, fgc=1, destc=2) -> dict:
    lim = max(8, min(max_h, max_w))          # transposing transforms swap h and w
    h = _unifint(diff_lb, diff_ub, (8, lim))
    w = _unifint(diff_lb, diff_ub, (8, lim))
    objh = _unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = _unifint(diff_lb, diff_ub, (objh, w // 2))
    bb = {(i, j) for i in range(objh) for j in range(objw)}
    sp = random.choice(sorted(bb))
    obj = {sp}
    ncells = _unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for _ in range(ncells - 1):
        cand = set()
        for c in obj:
            cand |= _dneigh(c)
        cand = sorted((cand & bb) - obj)
        if not cand:
            break
        obj.add(random.choice(cand))
    rs = [r for r, _ in obj]; cs = [c for _, c in obj]
    if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(obj):
        obj.remove(random.choice(sorted(obj)))
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
    sq = {(r, c) for r in range(locisq, locisq + sqd) for c in range(locjsq, locjsq + sqd)
          if 0 <= r < h and 0 <= c < w}
    for r, c in sq:
        gi[r, c] = destc
        go[r, c] = destc
    while not (obj & sq):
        obj = {(r - 1, c) for r, c in obj}
    obj = {(r + 1, c) for r, c in obj}
    for r, c in obj:
        if 0 <= r < h and 0 <= c < w:   # like fill(): off-grid cells are dropped
            go[r, c] = fgc
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def _roles(I):
    """(bgc, anchor cells, traveller colour, traveller cells), measured from I alone."""
    hi, wi = I.shape
    info = {}
    for col in sorted(set(I.flatten().tolist())):
        cells = list(zip(*np.nonzero(I == col)))
        rs = [int(r) for r, _ in cells]; cs = [int(c) for _, c in cells]
        bh = max(rs) - min(rs) + 1; bw = max(cs) - min(cs) + 1
        info[col] = dict(cells={(int(r), int(c)) for r, c in cells}, n=len(cells),
                         solid=(bh * bw == len(cells)),
                         small=(bh <= hi // 2 and bw <= wi // 2))
    cols = list(info)
    if len(cols) != 3:
        return None
    solids = [c for c in cols if info[c]["solid"]]
    if not solids:
        return None
    # anchor = the colour that exactly fills its bounding box (the square)
    anchor = min(solids, key=lambda c: info[c]["n"]) if len(solids) > 1 else solids[0]
    rest = [c for c in cols if c != anchor]
    # traveller fits within half the grid in both dims; the background spans the grid
    # (the square can be larger than the background, so majority count is not used)
    rest.sort(key=lambda c: (not info[c]["small"], info[c]["n"]))
    trav, bgc = rest[0], rest[1]
    return bgc, info[anchor]["cells"], trav, info[trav]["cells"]


def derive_operations(I, O=None, examples=None):
    # Reading: the ragged shape slides straight toward the solid square until it
    # touches it. The golf entry reads it as "the empty lines between the shape and the
    # anchor move to the far side". That is the same displacement with the same
    # quantity held fixed (the gap closes completely). Both readings agree, so the unit
    # stays the shape and the displacement is written as one grabbed Move chain.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    roles = _roles(I)
    if roles is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    bgc, rect, obj_col, obj = roles

    orr0 = min(r for r, _ in obj); orr1 = max(r for r, _ in obj)
    occ0 = min(c for _, c in obj); occ1 = max(c for _, c in obj)
    rr0 = min(r for r, _ in rect); rr1 = max(r for r, _ in rect)
    rc0 = min(c for _, c in rect); rc1 = max(c for _, c in rect)
    if orr0 > rr1:
        dr, dc = -1, 0        # square above -> slide up
    elif orr1 < rr0:
        dr, dc = 1, 0         # square below -> slide down
    elif occ0 > rc1:
        dr, dc = 0, -1        # square left  -> slide left
    elif occ1 < rc0:
        dr, dc = 0, 1         # square right -> slide right
    else:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    cur = set(obj)
    steps = 0
    while steps < 64:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & rect:
            break
        # part of the shape may slide past the grid edge (ARCLE clips it);
        # stop only if nothing would remain visible
        if not any(0 <= r < hi and 0 <= c < wi for r, c in nxt):
            break
        cur = nxt
        steps += 1

    if steps > 0:
        if obj_col != 0:
            mv = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
            ops.append(mv); sels.append(sel_of(sorted(obj)))      # grab the shape
            for _ in range(steps - 1):
                ops.append(mv); sels.append(sel_of([]))           # keep sliding it
            hole = sorted(set(obj) - cur)                         # vacated footprint only
            if bgc != 0 and hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
        else:
            # a colour-0 shape cannot be carried by Move: lift it, then set it down
            ops.append(int(bgc)); sels.append(sel_of(sorted(obj)))
            landed = sorted((r, c) for r, c in cur if 0 <= r < hi and 0 <= c < wi)
            ops.append(0); sels.append(sel_of(landed))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit
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
