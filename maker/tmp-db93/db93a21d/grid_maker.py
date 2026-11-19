"""
ARC Task: db93a21d (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


def sample_colors(num_examples=None) -> dict:
    # generator: cols = interval(0,10) minus {1,3}; bgc, fgc = sample(cols, 2)
    cols = [c for c in range(10) if c not in (1, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    def unif(bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * diff_lb)
        up = a + int((b - a) * diff_ub)
        lo = max(a, min(lo, b))
        up = max(lo, min(up, b))
        return random.randint(lo, up)

    # final grid is (h-1) x (w-2) after the crop at the end
    h_ub = min(31, max(12, max_h + 1))
    w_ub = min(32, max(12, max_w + 2))

    gi = go = None
    for _attempt in range(20):
        h = unif((12, h_ub))
        w = unif((12, w_ub))
        num = unif((1, max(1, (h * w) // 25)))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        avail = set((i, j) for i in range(h) for j in range(w))
        maxtrials = 4 * num
        tr = 0
        succ = 0
        while succ < num and tr <= maxtrials:
            if not avail:
                break
            oh = random.randint(1, max(1, h // 4))
            ow = oh
            fullh, fullw = 4 * oh, 4 * ow
            subs = [ij for ij in avail if ij[0] < h - fullh and ij[1] < w - fullw]
            if not subs:
                tr += 1
                continue
            loci, locj = random.choice(subs)
            big = set((i, j) for i in range(loci, loci + fullh)
                      for j in range(locj, locj + fullw))
            small = set((i, j) for i in range(loci + oh, loci + fullh - oh)
                        for j in range(locj + ow, locj + fullw - ow))
            threes = set((i, j) for i in range(h) for j in range(w) if go[i][j] == 3)
            if big <= (avail | threes):
                for (i, j) in small:
                    gi[i][j] = fgc
                for (i, j) in big:
                    go[i][j] = 3
                for (i, j) in small:
                    go[i][j] = fgc
                botrow = loci + fullh - 1 - oh          # bottom row of the small square
                for j in range(locj + ow, locj + fullw - ow):
                    for i in range(botrow, h):          # shoot downwards over bg only
                        if go[i][j] == bgc:
                            go[i][j] = 1
                succ += 1
                avail -= big
            tr += 1
        if succ >= 1:
            break

    gi = tuple(tuple(r[1:-1]) for r in gi[1:])
    go = tuple(tuple(r[1:-1]) for r in go[1:])
    return {"input": gi, "output": go}


def _squares_of(I, bgc):
    """Every foreground square in I, as (r0, r1, c0, c1) bounding boxes.

    The generator only ever draws solid squares of even side, always fully
    inside the cropped grid and never touching one another (a square keeps a
    clear margin of at least one cell), so 8-connected components == squares.
    """
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    out = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < hi and 0 <= nx < wi
                                and not seen[ny, nx] and I[ny, nx] != bgc):
                            seen[ny, nx] = True
                            q.append((ny, nx))
            rs = [y for y, _ in cells]
            cs = [x for _, x in cells]
            out.append((min(rs), max(rs), min(cs), max(cs)))
    return out


def derive_operations(I, O):
    """
    Rule, read off the inputs alone:
      * every square grows a frame of 3 around it, as thick as half its side
        (the frame only lands on cells that were background);
      * every square shoots a beam of 1 straight down its own columns, from
        just below the square to the bottom edge, over background only;
      * where a frame meets a beam the frame wins.

    The ROUTE through that rule (which square first, which way to sweep, frame
    before beam or beam then frame painted over it, frame in one block or ring
    by ring) is chosen from properties of the input grid itself, so a given
    puzzle always gets the same route and different puzzles get different ones.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    squares = _squares_of(I, bgc)
    if not squares:
        return [34], [[0, 0, hi - 1, wi - 1]]

    anchor = min(squares, key=lambda s: (s[0], s[2]))
    fgc = int(I[anchor[0], anchor[2]])
    n = len(squares)

    # ── route choices, all read from the input ───────────────────────────────
    order_mode = (n + fgc) % 4          # which square to deal with first
    phase_mode = (anchor[0] + anchor[2] + n) % 3   # frames/beams interleaving
    ring_by_ring = (bgc % 2 == 0)       # grow the frame outward, or fill it
    rings_inward = (hi % 2 == 1)        # ... and from which side
    split_beams = (wi % 2 == 0)         # one beam op, or column by column

    if order_mode == 0:                 # top row down
        order = sorted(squares, key=lambda s: (s[0], s[2]))
    elif order_mode == 1:               # bottom row up
        order = sorted(squares, key=lambda s: (-s[0], -s[2]))
    elif order_mode == 2:               # biggest square first
        order = sorted(squares, key=lambda s: (-(s[1] - s[0]), s[0], s[2]))
    else:                               # left column across
        order = sorted(squares, key=lambda s: (s[2], s[0]))

    cur = I.copy()
    ops, sels = [], []

    def frame_ops(sq):
        """Paint the 3-frame of one square (3 covers a beam already there)."""
        r0, r1, c0, c1 = sq
        pad = (c1 - c0 + 1) // 2
        rings = list(range(1, pad + 1))
        if not ring_by_ring:
            rings = [pad]               # whole frame in a single stroke
        elif rings_inward:
            rings = rings[::-1]
        done = set()
        for k in rings:
            cells = []
            for r in range(max(0, r0 - k), min(hi, r1 + k + 1)):
                for c in range(max(0, c0 - k), min(wi, c1 + k + 1)):
                    if (r, c) in done or I[r, c] != bgc or cur[r, c] == 3:
                        continue
                    cells.append((r, c))
            done.update((r, c) for r in range(r0 - k, r1 + k + 1)
                        for c in range(c0 - k, c1 + k + 1))
            if cells:
                ops.append(3)
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    cur[r, c] = 3

    def beam_ops(sq):
        """Shoot the 1-beam down from one square, over background only."""
        r0, r1, c0, c1 = sq
        cols = list(range(c0, c1 + 1))
        if (c0 + c1) >= (wi - 1):        # sweep away from the nearer side
            cols = cols[::-1]
        groups = [[c] for c in cols] if split_beams else [cols]
        for grp in groups:
            cells = []
            for c in grp:
                for r in range(r1 + 1, hi):
                    if cur[r, c] == bgc:
                        cells.append((r, c))
            if cells:
                ops.append(1)
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    cur[r, c] = 1

    if phase_mode == 0:                  # frames everywhere, then the beams
        for sq in order:
            frame_ops(sq)
        for sq in order:
            beam_ops(sq)
    elif phase_mode == 1:                # beams laid down, frames drawn over
        for sq in order:
            beam_ops(sq)
        for sq in order:
            frame_ops(sq)
    else:                                # one square finished at a time
        for sq in order:
            frame_ops(sq)
            beam_ops(sq)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid bbox: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task db93a21d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db93a21d"
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
                                f"for task db93a21d"
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
                    f"Failed to build a complete episode for task db93a21d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db93a21d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
