"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numc)
    ccols = list(ccols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            go = fill(go, ringcols[0], plcdrings[0])
            idx = randint(1, 3)
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            go = fill(go, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
            remringcols = ringcols[1:idx] + ringcols[idx + 1:]
            numrs = unifint(diff_lb, diff_ub, (1, 2))
            locs = sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            tofillgi = merge(frozenset(
                recolor(col, frozenset(sample(totuple(remring), 4 - unifint(diff_lb, diff_ub, (0, 3)))))
                for remring, col in zip(remrings, remringcols)
            ))
            tofillgo = merge(frozenset(
                recolor(col, remring) for remring, col in zip(remrings, remringcols)
            ))
            if min(shape(tofillgi)) == 5:
                succ += 1
                gi = paint(gi, tofillgi)
                go = paint(go, tofillgo)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: the canvas colour the generator paints before placing objects;
    # objects are sparse lattice dots, so it dominates the grid.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # concentric rings of a 5x5 object, inner -> outer
    CENTER = [(2, 2)]
    RING3 = [(1, 1), (1, 3), (3, 1), (3, 3)]
    RING2 = [(0, 2), (2, 0), (2, 4), (4, 2)]
    RING1 = [(0, 0), (0, 4), (4, 0), (4, 4)]
    LATTICE = set(CENTER) | set(RING3) | set(RING2) | set(RING1)

    def window_ok(r0, c0):
        # centre dot must be present (generator always draws it)
        if I[r0 + 2, c0 + 2] == bgc:
            return None
        cells = []
        for dr in range(5):
            for dc in range(5):
                if I[r0 + dr, c0 + dc] != bgc:
                    if (dr, dc) not in LATTICE:
                        return None
                    cells.append((dr, dc))
        # object bbox must be exactly the 5x5 block
        rs = {d[0] for d in cells}
        cs = {d[1] for d in cells}
        if min(rs) != 0 or max(rs) != 4 or min(cs) != 0 or max(cs) != 4:
            return None
        # each ring must be single-coloured; at least one outer ring complete
        info = {}
        complete = False
        for name, ring in (("r3", RING3), ("r2", RING2), ("r1", RING1)):
            present = [(dr, dc) for (dr, dc) in ring if I[r0 + dr, c0 + dc] != bgc]
            if not present:
                info[name] = None
                continue
            colset = {int(I[r0 + dr, c0 + dc]) for (dr, dc) in present}
            if len(colset) != 1:
                return None
            if len(present) == 4:
                complete = True
            info[name] = (colset.pop(), present)
        if not complete:
            return None
        # objects are placed with a clear one-cell margin all around
        for r in range(r0 - 1, r0 + 6):
            for c in range(c0 - 1, c0 + 6):
                if r0 <= r <= r0 + 4 and c0 <= c <= c0 + 4:
                    continue
                if 0 <= r < hi and 0 <= c < wi and I[r, c] != bgc:
                    return None
        return info

    # locate every 5x5 ring object in I
    objects = []
    taken = np.zeros((hi, wi), dtype=bool)
    for r0 in range(hi - 4):
        for c0 in range(wi - 4):
            if taken[r0:r0 + 5, c0:c0 + 5].any():
                continue
            info = window_ok(r0, c0)
            if info is not None:
                objects.append((r0, c0, info))
                taken[r0:r0 + 5, c0:c0 + 5] = True

    ops, sels = [], []
    RINGS = (("r3", RING3), ("r2", RING2), ("r1", RING1))
    # complete each partially drawn ring, inner ring first, one object at a time
    for (r0, c0, info) in objects:
        for name, ring in RINGS:
            entry = info.get(name)
            if entry is None:
                continue
            col, present = entry
            missing = [(r0 + dr, c0 + dc) for (dr, dc) in ring
                       if I[r0 + dr, c0 + dc] == bgc]
            if not missing:
                continue
            ops.append(int(col))
            sels.append(sel_of(missing))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
