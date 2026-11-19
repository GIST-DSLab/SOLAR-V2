"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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

# The rule of this task: the input shows the top rows of a motif that repeats
# down the grid with a constant (dr, dc) step; the answer canvas is always 10
# rows tall and as wide as the input.
H_OUT = 10

VARIANTS = [{"mirror": False}, {"mirror": True}]


# --------------------------------------------------------------------------- #
# episode-level colors / structural plan
# --------------------------------------------------------------------------- #
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# --------------------------------------------------------------------------- #
# helpers shared by generate() (self-validation) and derive_operations()
# --------------------------------------------------------------------------- #
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    ilb = int(round(a + (b - a) * diff_lb))
    iub = int(round(a + (b - a) * diff_ub))
    if iub < ilb:
        ilb, iub = iub, ilb
    ilb = max(a, min(b, ilb))
    iub = max(a, min(b, iub))
    return random.randint(ilb, iub)


def _bg_color(I):
    return int(Counter(np.asarray(I, dtype=int).flatten().tolist()).most_common(1)[0][0])


def _fg_cells(I, bgc):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    return {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if int(I[r, c]) != bgc}


def _detect_step(cells):
    """Periodicity vector of the motif, measured on the INPUT only.

    Same criterion as the task rule: among all shifts (di, dj) whose overlap
    with the pattern is colour-consistent and non-empty, take the one with the
    biggest overlap (tie-break on di*dj, then di, then dj)."""
    best = None
    for di in range(1, 6):
        for dj in range(-10, 10):
            ov = 0
            ok = True
            for (r, c), col in cells.items():
                t = (r + di, c + dj)
                if t in cells:
                    ov += 1
                    if cells[t] != col:
                        ok = False
                        break
            if ok and ov > 0:
                key = (ov, di * dj, di, dj)
                if best is None or key > best:
                    best = key
    if best is None:
        return None
    return best[2], best[3]


def _plan(I):
    """(bgc, dr, dc, motif band, h, w) — everything measured from I."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = _bg_color(I)
    cells = _fg_cells(I, bgc)
    if not cells:
        return None
    step = _detect_step(cells)
    if step is None:
        return None
    dr, dc = step
    if dr >= h:
        return None
    band0 = {(r, c): col for (r, c), col in cells.items() if r < dr}
    if not band0:
        return None
    return bgc, dr, dc, band0, h, w


def _band_cells(band0, dr, dc, k, h, w):
    """The k-th repetition of the motif, restricted to the newly added rows."""
    out = {}
    for (r, c), col in band0.items():
        rr, cc = r + k * dr, c + k * dc
        if h <= rr < H_OUT and 0 <= cc < w:
            out[(rr, cc)] = col
    return out


def _render(I):
    p = _plan(I)
    if p is None:
        return None
    bgc, dr, dc, band0, h, w = p
    out = np.full((H_OUT, w), bgc, dtype=int)
    out[:h] = np.asarray(I, dtype=int)
    for k in range(1, H_OUT):
        if k * dr >= H_OUT:
            break
        for (rr, cc), col in _band_cells(band0, dr, dc, k, h, w).items():
            out[rr, cc] = col
    return out


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])
    remcols = [c for c in range(10) if c != bgc]
    w_ub = max(8, min(30, int(max_w)))
    h_ub = max(2, min(6, int(max_h)))

    while True:
        h = _unifint(diff_lb, diff_ub, (2, h_ub))
        w = _unifint(diff_lb, diff_ub, (8, w_ub))
        ncols = _unifint(diff_lb, diff_ub, (1, 9))
        ccols = random.sample(remcols, ncols)
        oh = _unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
        ow = _unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = _unifint(diff_lb, diff_ub, (1, oh * ow))
        picked = random.sample(bounds, ncells)
        obj = {ij: random.choice(ccols) for ij in picked}
        mi = min(i for i, j in obj)
        mj = min(j for i, j in obj)
        obj = {(i - mi, j - mj): c for (i, j), c in obj.items()}
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1

        locj = random.randint(0, w // 2)
        hoffs = random.randint(0, ow // 2 + 1)
        go = [[bgc] * w for _ in range(H_OUT)]
        for k in range(H_OUT // oh + 1):
            for (i, j), col in obj.items():
                r = i + k * oh
                c = j + locj + k * hoffs
                if 0 <= r < H_OUT and 0 <= c < w:
                    go[r][c] = col

        if len(set(v for row in go[h:] for v in row)) < 2:
            continue

        gi = [row[:] for row in go[:h]]
        if mirror:
            gi = [row[::-1] for row in gi]
            go = [row[::-1] for row in go]

        # keep only instances whose continuation is unambiguously readable
        # from the input alone
        pred = _render(gi)
        if pred is None or not np.array_equal(pred, np.asarray(go, dtype=int)):
            continue

        return {"input": gi, "output": go}


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    bgc, dr, dc, band0, _, _ = _plan(I)

    ops, sels = [], []

    # 1. grow the canvas to the rule's 10 rows (full rectangle -> bbox is exact)
    ops.append(33)
    sels.append([0, 0, H_OUT - 1, w - 1])

    # 2. lay the background over the newly added rows (full rectangle)
    if bgc != 0 and h < H_OUT:
        ops.append(int(bgc))
        sels.append([h, 0, H_OUT - 1 - h, w - 1])

    # 3. continue the repetition: one copy of the motif per step, drawn colour
    #    by colour, in the order the copies march down the grid
    for k in range(1, H_OUT):
        if k * dr >= H_OUT:
            break
        cells = _band_cells(band0, dr, dc, k, h, w)
        if not cells:
            continue
        bycol = {}
        for rc, col in cells.items():
            bycol.setdefault(col, []).append(rc)
        for col in bycol:
            bycol[col].sort()
        for col in sorted(bycol, key=lambda cc: bycol[cc][0]):
            ops.append(int(col))
            sels.append(sel_of(bycol[col]))

    ops.append(34)
    sels.append([0, 0, H_OUT - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
