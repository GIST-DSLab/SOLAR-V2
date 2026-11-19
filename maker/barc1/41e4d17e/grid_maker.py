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
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c != 6]
        bgc, fgc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
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
    The reference's reading: for each 5x5 box, draw a beam-coloured bar along the
    WHOLE row and the WHOLE column through its centre (over everything), then
    redraw the box objects on top so the bars look like they run underneath.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # beam colour: the colour every demo output has and no demo input has
    beam = None
    if examples:
        cand = None
        for pair in examples:
            try:
                a = set(np.asarray(pair[0], dtype=int).flatten().tolist())
                b = set(np.asarray(pair[1], dtype=int).flatten().tolist())
            except Exception:
                continue
            s = b - a
            cand = s if cand is None else (cand & s)
        if cand and len(cand) == 1:
            beam = int(next(iter(cand)))
    if beam is None:
        beam = 6

    def components(mask_fn):
        seen = np.zeros((h, w), dtype=bool)
        comps = []
        for r0 in range(h):
            for c0 in range(w):
                if seen[r0, c0] or not mask_fn(r0, c0):
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
                        if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                                and mask_fn(ny, nx) and I[ny, nx] == col):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append((int(col), cells))
        return comps

    # box interiors: single-colour 4-connected components of 9 cells in a 3x3 square
    centers = []
    bg_votes = Counter()
    for col, cells in components(lambda r, c: True):
        if len(cells) != 9:
            continue
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        if max(rs) - min(rs) == 2 and max(cs) - min(cs) == 2:
            centers.append((min(rs) + 1, min(cs) + 1))
            bg_votes[col] += 1
    if bg_votes:
        bgc = bg_votes.most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    centers.sort()

    cur = I.copy()
    ops, sels = [], []

    # 1. bars: whole row and whole column through each box centre
    done_rows, done_cols = set(), set()
    for (cr, cc) in centers:
        if cr not in done_rows:
            done_rows.add(cr)
            if np.any(cur[cr, :] != beam):
                ops.append(int(beam))
                sels.append([cr, 0, 0, w - 1])  # full row: exact rectangle
                cur[cr, :] = beam
        if cc not in done_cols:
            done_cols.add(cc)
            if np.any(cur[:, cc] != beam):
                ops.append(int(beam))
                sels.append([0, cc, h - 1, 0])  # full column: exact rectangle
                cur[:, cc] = beam

    # 2. redraw the box objects (non-background components of I) on top
    for col, cells in components(lambda r, c: I[r, c] != bgc):
        cells = sorted(cells)
        if any(cur[r, c] != col for r, c in cells):
            ops.append(int(col))
            sels.append(sel_of(cells))
            for r, c in cells:
                cur[r, c] = col

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole grid rectangle
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
