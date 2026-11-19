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
        return int(round(a + (b - a) * d))

    cols = [c for c in range(10) if c != 6]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(r, c) for r in range(h) for c in range(w)}
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
                gi[r, c] = fgc
                go[r, c] = fgc
            cr, cc = loc[0] + 2, loc[1] + 2
            for c in range(w):
                if go[cr, c] == bgc:
                    go[cr, c] = 6
            for r in range(h):
                if go[r, cc] == bgc:
                    go[r, cc] = 6
            inds -= {(r + loc[0], c + loc[1]) for r, c in bd}
            succ += 1
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule: each 4-connected single-colour component of size 9 with a square bbox is
    a 5x5 box's hollow interior. Every background cell in the row and in the column
    through that interior's centre turns into the beam colour.

    The maker draws one box at a time (row beam, then column beam). This version
    draws one direction at a time for the whole grid:
      op A: a single Color(beam) on every background cell of every centre row
            (all horizontal beams at once);
      op B: a single Color(beam) on every background cell of every centre column
            that isn't already beam-coloured (all vertical beams at once).
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # beam colour: the one colour that is in every demo output and in no demo input
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

    # find the box interiors (size-9 square components)
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
                centers.append((min(rs) + hh // 2, min(cs) + ww // 2))
                interior_cols.append(int(col))

    # background = colour of the hollow interiors (fallback: majority colour)
    if interior_cols:
        bgc = Counter(interior_cols).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    beam_rows = sorted({cr for cr, _ in centers})
    beam_cols = sorted({cc for _, cc in centers})

    cur = I.copy()
    ops, sels = [], []

    # all horizontal beams in one stroke
    row_cells = [(r, c) for r in beam_rows for c in range(w) if cur[r, c] == bgc]
    if row_cells:
        ops.append(int(beam))
        sels.append(sel_of(row_cells))
        for r, c in row_cells:
            cur[r, c] = beam

    # all vertical beams in one stroke
    col_cells = [(r, c) for c in beam_cols for r in range(h) if cur[r, c] == bgc]
    if col_cells:
        ops.append(int(beam))
        sels.append(sel_of(col_cells))
        for r, c in col_cells:
            cur[r, c] = beam

    # Submit: whole-grid rectangle
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
