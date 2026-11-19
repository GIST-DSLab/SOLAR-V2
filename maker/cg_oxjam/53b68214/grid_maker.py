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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

H_OUT = 10


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _period(I, bgc):
    """Repetition vector (dr, dc) of the pattern in I: the largest colour-consistent
    self-overlap among shifts dr in 1..5, dc in -9..9 (ties -> larger dr*dc)."""
    h, w = I.shape
    obj = {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if I[r, c] != bgc}
    best = None
    for dr in range(1, 6):
        for dc in range(-9, 10):
            same = 0
            ok = True
            for (r, c), v in obj.items():
                t = (r + dr, c + dc)
                if t in obj:
                    if obj[t] != v:
                        ok = False
                        break
                    same += 1
            if not ok or same == 0:
                continue
            key = (same, dr * dc)
            if best is None or key > best[0]:
                best = (key, dr, dc)
    if best is None:
        return None
    return best[1], best[2]


def _continue(I):
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    G = np.full((H_OUT, w), bgc, dtype=int)
    G[:h] = I
    per = _period(I, bgc)
    if per is None:
        return G
    dr, dc = per
    for r in range(h, H_OUT):
        for c in range(w):
            if G[r - dr, c] != bgc and 0 <= c + dc < w:
                G[r, c + dc] = G[r - dr, c]
    return G


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))
    if bgc is None:
        bgc = random.choice(range(10))
    max_w = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (8, max_w))
        remcols = [c for c in range(10) if c != bgc]
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = random.sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        cells = random.sample(bounds, ncells)
        mi = min(i for i, _ in cells); mj = min(j for _, j in cells)
        obj = [(random.choice(ccols), (i - mi, j - mj)) for i, j in cells]
        oh = max(i for _, (i, _) in obj) + 1
        ow = max(j for _, (_, j) in obj) + 1
        locj = random.randint(0, w // 2)
        go = [[bgc] * w for _ in range(10)]
        hoffs = random.randint(0, ow // 2 + 1)
        for k in range(10 // oh + 1):
            for c, (i, j) in obj:
                r, cc = i + k * oh, j + locj + k * hoffs
                if 0 <= r < 10 and 0 <= cc < w:
                    go[r][cc] = c
        if len(set(v for row in go[h:] for v in row)) <= 1:
            continue
        if random.choice((True, False)):
            go = [row[::-1] for row in go]
        gi = [row[:] for row in go[:h]]
        # keep only instances whose continuation is determined by the input
        if np.array_equal(_continue(np.array(gi, dtype=int)), np.array(go, dtype=int)):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []

    # 1. extend the canvas down to 10 rows (full rectangle)
    ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
    if h >= H_OUT:
        ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
        return ops, sels
    # 2. lay the background over the newly added rows (full rectangle);
    #    every new row keeps background cells, so this base is never fully covered
    if bgc != 0:
        ops.append(int(bgc)); sels.append([h, 0, H_OUT - 1 - h, w - 1])

    per = _period(I, bgc)
    G = np.full((H_OUT, w), bgc, dtype=int)
    G[:h] = I
    if per is not None:
        dr, dc = per
        # 3. row-by-row continuation (the unit is a ROW): each new row repeats
        #    the row one period above it as it now stands on the grid, shifted
        #    sideways by dc; drawn colour by colour, rows top to bottom
        for r in range(h, H_OUT):
            src = G[r - dr]
            bycol = {}
            for c in range(w):
                if src[c] != bgc and 0 <= c + dc < w:
                    bycol.setdefault(int(src[c]), []).append((r, c + dc))
            for col in sorted(bycol, key=lambda cc: bycol[cc][0][1]):
                ops.append(col); sels.append(sel_of(bycol[col]))
                for rc in bycol[col]:
                    G[rc] = col
    ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
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
