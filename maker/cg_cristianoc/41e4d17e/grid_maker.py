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
    # Beam colour 6 is hardcoded in the generator; bg and box colours are random.
    cols = [c for c in range(10) if c != 6]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or fgc is None:
        bgc, fgc = random.sample([c for c in range(10) if c != 6], 2)

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(r, c) for r in range(h) for c in range(w)}
    bx = [(r, c) for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)]
    bd = [(r, c) for r in range(5) for c in range(5)]

    maxtrials = 4 * num
    succ = tr = 0
    while succ < num and tr < maxtrials:
        lr, lc = random.choice(sorted(inds))
        bxs = [(r + lr, c + lc) for r, c in bx]
        if all(p in inds for p in bxs):
            for r, c in bxs:
                gi[r, c] = fgc
                go[r, c] = fgc
            cr, cc = lr + 2, lc + 2
            for c in range(w):
                if go[cr, c] == bgc:
                    go[cr, c] = 6
            for r in range(h):
                if go[r, cc] == bgc:
                    go[r, cc] = 6
            for r, c in bd:
                inds.discard((r + lr, c + lc))
            succ += 1
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (the golf reading): a box is a 5x5 window whose 16 border cells are
    the box colour and whose 3x3 interior is background.  Each box contributes
    its centre row to ONE set of beam rows and its centre column to ONE set of
    beam columns.  Then two strokes: every background cell in the beam rows
    becomes the beam colour, then every background cell in the beam columns does
    (outlines survive).  So all horizontal beams are one unit and all vertical
    beams are another, rather than painting box by box.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # beam colour: the convention from the demos (in every output, in no input)
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

    def find_boxes(fg, bg):
        found = []
        for r in range(h - 4):
            for c in range(w - 4):
                win = I[r:r + 5, c:c + 5]
                border = np.ones((5, 5), dtype=bool)
                border[1:4, 1:4] = False
                if np.all(win[border] == fg) and np.all(win[1:4, 1:4] == bg):
                    found.append((r + 2, c + 2))
        return found

    colours = [k for k, _ in Counter(I.flatten().tolist()).most_common()]
    bgc = colours[0]
    centers = []
    if len(colours) >= 2:
        # the box colour is the one that forms 5x5 frames around a background interior
        for fg, bg in ((colours[1], colours[0]), (colours[0], colours[1])):
            cs = find_boxes(fg, bg)
            if cs:
                centers, bgc = cs, bg
                break

    ops, sels = [], []
    beam_rows = sorted({r for r, _ in centers})
    beam_cols = sorted({c for _, c in centers})

    # stroke 1: all horizontal beams at once (background cells of the beam rows)
    row_cells = [(r, c) for r in beam_rows for c in range(w) if I[r, c] == bgc]
    if row_cells and bgc != beam:
        ops.append(int(beam)); sels.append(sel_of(row_cells))
    row_set = set(row_cells)

    # stroke 2: all vertical beams at once (background cells of the beam columns
    # that do not already hold the beam colour from stroke 1)
    col_cells = [(r, c) for c in beam_cols for r in range(h)
                 if I[r, c] == bgc and (r, c) not in row_set]
    if col_cells and bgc != beam:
        ops.append(int(beam)); sels.append(sel_of(col_cells))

    # Submit: whole-grid rectangle
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
