"""
ARC Task: 41e4d17e (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 6]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c != 6]
        bgc, fgc = random.sample(cols, 2)

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = set((r, c) for r in range(h) for c in range(w))
    bx = [(r, c) for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)]
    bd = [(r, c) for r in range(5) for c in range(5)]
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = random.choice(sorted(inds))
        bxs = [(r + loc[0], c + loc[1]) for r, c in bx]
        if all(p in inds for p in bxs):
            for r, c in bxs:
                gi[r][c] = fgc
                go[r][c] = fgc
            cr, cc = loc[0] + 2, loc[1] + 2
            for c in range(w):
                if go[cr][c] == bgc:
                    go[cr][c] = 6
            for r in range(h):
                if go[r][cc] == bgc:
                    go[r][cc] = 6
            for r, c in bd:
                inds.discard((r + loc[0], c + loc[1]))
            succ += 1
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Same rule as the existing maker, but I split it into different units.
    A grid line (a whole row or a whole column) is one unit. A line is a beam line
    if it runs through the centre of a hollow 3x3 box interior.
      Phase 1: paint each distinct beam ROW. The cells painted are the ones in that
               row that currently hold the background colour.
      Phase 2: paint each distinct beam COLUMN, again only its cells that currently
               hold the background colour. Cells the rows already turned into the
               beam colour are no longer background, so they are left alone.
    When two boxes share a row or column, that line is painted with one op.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # beam colour: the colour that every demo output has and no demo input has
    beam = None
    if examples:
        cand = None
        for pair in examples:
            try:
                ex_i, ex_o = pair[0], pair[1]
            except Exception:
                continue
            a = set(np.asarray(ex_i, dtype=int).flatten().tolist())
            b = set(np.asarray(ex_o, dtype=int).flatten().tolist())
            s = b - a
            cand = s if cand is None else (cand & s)
        if cand and len(cand) == 1:
            beam = int(next(iter(cand)))
    if beam is None:
        beam = 6

    # find the box interiors: 4-connected one-colour components of size 9 with a square bbox
    seen = np.zeros((h, w), dtype=bool)
    centers = []
    interior_cols = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            col = I[r0, c0]
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) != 9:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            hh = max(rs) - min(rs) + 1
            ww = max(cs) - min(cs) + 1
            if hh == ww:
                centers.append((min(rs) + 1, min(cs) + 1))
                interior_cols.append(int(col))

    # background = the colour of the box interiors (outlines are always bigger than
    # 9 cells); if there are no interiors, fall back to the most common colour
    if interior_cols:
        bgc = Counter(interior_cols).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    beam_rows = sorted(set(r for r, _ in centers))
    beam_cols = sorted(set(c for _, c in centers))

    cur = I.copy()
    ops, sels = [], []

    # Phase 1: horizontal beams, one whole row at a time
    for r in beam_rows:
        cells = [(r, c) for c in range(w) if cur[r, c] == bgc]
        if cells and bgc != beam:
            ops.append(int(beam))
            sels.append(sel_of(cells))
            for (y, x) in cells:
                cur[y, x] = beam

    # Phase 2: vertical beams, only the cells in the column that are still background
    for c in beam_cols:
        cells = [(r, c) for r in range(h) if cur[r, c] == bgc]
        if cells and bgc != beam:
            ops.append(int(beam))
            sels.append(sel_of(cells))
            for (y, x) in cells:
                cur[y, x] = beam

    # Submit: the selection is the whole grid, so a full-rectangle bbox is correct here
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
                        f"num_examples+1 ({num_examples + 1}) for task 41e4d17e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 41e4d17e"
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
                                f"for task 41e4d17e"
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
                    f"Failed to build a complete episode for task 41e4d17e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"41e4d17e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
