"""
ARC Task: e8593010 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # The generator draws bgc and fgc from colours that exclude 1, 2, 3
    # (those three are reserved for the size-classes in the output).
    cols = [0, 4, 5, 6, 7, 8, 9]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    a = (frozenset({ORIGIN}),)
    b = (frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN}))
    c = (
        frozenset({ORIGIN, DOWN, UNITY}),
        frozenset({ORIGIN, DOWN, RIGHT}),
        frozenset({UNITY, DOWN, RIGHT}),
        frozenset({UNITY, ORIGIN, RIGHT}),
        shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
        shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT),
    )
    prs = [(a, 3), (b, 2), (c, 1)]

    hub = max(5, min(30, int(max_h)))
    wub = max(5, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)

    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 2) // 2))
    nobjs = max(nobjs, 3)

    # every instance shows all three size-classes at least once, so the
    # size -> colour convention is always readable from the demonstrations
    plan = [prs[0], prs[1], prs[2]]
    random.shuffle(plan)
    plan += [random.choice(prs) for _ in range(nobjs - 3)]

    maxtr = 10
    for objs, col in plan:
        obj = random.choice(objs)
        ntr = 0
        while ntr < maxtr:
            if len(reminds) == 0:
                break
            loc = random.choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def _bgc_of(I):
    return int(Counter(np.asarray(I).flatten().tolist()).most_common(1)[0][0])


def _components(I, bgc):
    """4-connected same-colour blobs of non-background cells."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(cells))
    return comps


def _size_color_map(examples):
    """Read the 'blob of N cells becomes colour X' convention off the demos."""
    m = {}
    if not examples:
        return m
    for pair in examples:
        Ie = np.asarray(pair[0], dtype=int)
        Oe = np.asarray(pair[1], dtype=int)
        if Ie.shape != Oe.shape:
            continue
        bg = _bgc_of(Ie)
        for cells in _components(Ie, bg):
            r, c = cells[0]
            m.setdefault(len(cells), int(Oe[r, c]))
    return m


def derive_operations(I, O, examples=None):
    """
    Rule: every blob keeps its shape and place; its colour tells its size
    (1 cell -> 3, 2 cells -> 2, 3 cells -> 1).

    Account: the blob shapes REPEAT all over the grid.  For each distinct
    shape, recolour one instance by hand (that is the prototype), copy the
    finished prototype, and stamp it onto every other blob of that same
    shape.  Shape-classes are walked in size order, so the trajectory reads
    'singles first, then dominoes, then triominoes'.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = _bgc_of(I)
    cmap = {1: 3, 2: 2, 3: 1}          # task convention, confirmed by the demos
    cmap.update(_size_color_map(examples))

    ops, sels = [], []

    # group blobs by their exact normalised shape (so one stamp fits them all)
    groups = {}
    for cells in _components(I, bgc):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        bh = max(rs) - r0 + 1
        bw = max(cs) - c0 + 1
        key = (len(cells), bh, bw, tuple(sorted((r - r0, c - c0) for r, c in cells)))
        groups.setdefault(key, []).append((r0, c0, cells))

    for key in sorted(groups, key=lambda k: (k[0], -len(groups[k]), k[1], k[2], k[3])):
        size, bh, bw, _ = key
        col = int(cmap.get(size, 4 - size))
        occ = sorted(groups[key], key=lambda t: (t[0], t[1]))

        # --- prototype: paint this blob's own cells (not its bounding box) ---
        pr, pc, pcells = occ[0]
        ops.append(col)
        sels.append(sel_of(pcells))

        if len(occ) > 1:
            # --- copy the finished prototype: the FULL bbox rectangle is
            # intended here (blob + the bgc cells of its bounding box), that is
            # exactly what a stamp is ---
            rect = [(r, c) for r in range(pr, pr + bh) for c in range(pc, pc + bw)]
            ops.append(29)                      # CopyO: the freshly recoloured blob
            sels.append(sel_of(rect))
            # --- stamp it onto every further blob of the same shape ---
            for (r1, c1, _cells) in occ[1:]:
                ops.append(30)                  # Paste at that blob's top-left
                sels.append(sel_of([(r1, c1)]))

    # Submit: selection is the whole grid rectangle
    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task e8593010"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8593010"
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
                                f"for task e8593010"
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
                    f"Failed to build a complete episode for task e8593010 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8593010-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
