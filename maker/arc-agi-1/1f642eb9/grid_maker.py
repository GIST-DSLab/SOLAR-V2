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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, sqc: int) -> dict:
    cols = interval(0, 10, 1)
    max_h = max(6, min(30, max_h))
    max_w = max(6, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    remcols = difference(cols, (bgc, sqc))
    numcells = unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs = []
    ins = []
    c1 = choice((True, False))
    c2 = choice((True, False))
    c3 = choice((True, False))
    c4 = choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, 2 * ih + 2 * iw - 4, 1)
    locs = sample(inds, numcells)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    c = canvas(bgc, (h, w))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(c, sqc, bd)
    seq = [choice(ccols) for k in range(numcells)]
    for col, loc in zip(seq, outs):
        gi = fill(gi, col, {loc})
    go = tuple(e for e in gi)
    for col, loc in zip(seq, ins):
        go = fill(go, col, {loc})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured from I only):
      * one solid rectangle of a single colour sits in the interior  -> the "block"
      * every other non-background cell is an isolated marker sitting on a border
      * each marker travels along its row/column toward the block until it lands ON
        the block's perimeter cell facing it; the marker itself stays where it is
    So: one Color op per marker, painting the perimeter cell it projects onto with
    that marker's own colour (both read from I, never from O).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = colour the canvas was filled with (dominant colour of I)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- locate the block: largest connected same-colour component that is a full rectangle
    seen = np.zeros((h, w), dtype=bool)
    best = None
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [y for y, _ in cells]
            cs = [x for _, x in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                if best is None or len(cells) > best[0]:
                    best = (len(cells), r0, r1, c0, c1)

    ops, sels = [], []
    if best is None:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    _, r0, r1, c0, c1 = best

    # --- collect markers: non-background cells outside the block
    markers = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            markers.append((r, c, int(I[r, c])))

    left, right, top, bottom = [], [], [], []
    for (r, c, col) in markers:
        if r0 <= r <= r1:                       # travels horizontally toward the block
            if c < c0:
                left.append((r, c, col, r, c0))
            elif c > c1:
                right.append((r, c, col, r, c1))
        elif c0 <= c <= c1:                     # travels vertically toward the block
            if r < r0:
                top.append((r, c, col, r0, c))
            elif r > r1:
                bottom.append((r, c, col, r1, c))

    left.sort(key=lambda t: t[0])
    right.sort(key=lambda t: t[0])
    top.sort(key=lambda t: t[1])
    bottom.sort(key=lambda t: t[1])

    # one op per source marker: stamp its colour on the perimeter cell it reaches
    for group in (left, right, top, bottom):
        for (sr, sc, col, lr, lc) in group:
            ops.append(col)                     # Color<col>
            sels.append(sel_of([(lr, lc)]))     # exactly the landing cell

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
