"""
ARC Task: 1f642eb9 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The block colour and the background matter to the rule. The marker colours do not:
    # each marker just carries its own colour, so they stay random in each instance.
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(cols, 2)
    H = max(6, min(30, max_h))
    W = max(6, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (6, H))
    w = unifint(diff_lb, diff_ub, (6, W))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    remcols = [c for c in cols if c not in (bgc, sqc)]
    numcells = unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs, ins = [], []
    c1, c2, c3, c4 = [random.choice((True, False)) for _ in range(4)]
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0)); ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1)); ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - c3):
        outs.append((0, b)); ins.append((loci, b))
    for b in range(locj + c2, locj + iw - c4):
        outs.append((h - 1, b)); ins.append((loci + ih - 1, b))
    n_slots = len(outs)
    numcells = max(1, min(numcells, n_slots))
    locs = set(random.sample(range(n_slots), numcells))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
    seq = [random.choice(ccols) for _ in range(len(outs))]
    for col, (r, c) in zip(seq, outs):
        gi[r][c] = col
    go = [row[:] for row in gi]
    for col, (r, c) in zip(seq, ins):
        go[r][c] = col
    return {"input": [tuple(r) for r in gi], "output": [tuple(r) for r in go]}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): look in from each of the four
    grid edges in turn. For every line whose edge cell is a marker, the marker's colour
    lands on the FIRST block cell along that line. The golf entry handles one whole
    edge per rotation, and the maker also groups by side, so the two readings agree.
    That gives one Color op per marker. The colour is read from the marker and the
    target is the block cell it faces. Both come from I only.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # The generator never puts a marker on a grid corner, so a corner is always background.
    bgc = int(I[0, 0])

    # The block is a solid rectangular component that does not touch the border
    # (the generator places it at least 2 cells in from every edge).
    seen = np.zeros((h, w), dtype=bool)
    best = None
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            rs = [y for y, _ in cells]; cs = [x for _, x in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                if best is None or len(cells) > best[0]:
                    best = (len(cells), r0, r1, c0, c1)

    ops, sels = [], []
    if best is not None:
        _, r0, r1, c0, c1 = best
        cur = I.copy()
        # Go round the four edges. Each edge cell that is a marker sends its colour
        # straight in until it meets the block.
        sides = [
            [((r, 0), (r, c0)) for r in range(r0, r1 + 1)],          # left edge
            [((0, c), (r0, c)) for c in range(c0, c1 + 1)],          # top edge
            [((r, w - 1), (r, c1)) for r in range(r0, r1 + 1)],      # right edge
            [((h - 1, c), (r1, c)) for c in range(c0, c1 + 1)],      # bottom edge
        ]
        for side in sides:
            for (sr, sc), (lr, lc) in side:
                col = int(I[sr, sc])
                if col == bgc:
                    continue
                if cur[lr, lc] == col:
                    continue
                ops.append(col)
                sels.append(sel_of([(lr, lc)]))
                cur[lr, lc] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
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
                                f"for task 1f642eb9"
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
                    f"Failed to build a complete episode for task 1f642eb9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f642eb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
