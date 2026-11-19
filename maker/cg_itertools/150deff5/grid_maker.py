"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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
from functools import lru_cache
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + d * (b - a)))))

    cols = [c for c in range(10) if c not in (2, 8)]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb, reforb = set(), set()
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))
    inds = {(i, j) for i in range(h) for j in range(w)}
    for _ in range(noccs):
        if random.random() < 0.5:
            obj, col = bo, 8
        else:
            obj, col = random.choice((ro1, ro2)), 2
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        forb = boforb if col == 8 else reforb
        cands = [(i, j) for (i, j) in inds
                 if i <= h - oh and j <= w - ow
                 and all((i + a, j + b) in inds for a, b in obj)
                 and (i, j) not in forb]
        if not cands:
            break
        i, j = random.choice(sorted(cands))
        if col == 8:
            for d in ((-2, 0), (2, 0), (0, 2), (0, -2)):
                boforb.add((i + d[0], j + d[1]))
        else:
            if obj == ro1:
                reforb.add((i, j + 3)); reforb.add((i, j - 3))
            else:
                reforb.add((i + 1, j)); reforb.add((i - 1, j))
        plcd = {(i + a, j + b) for a, b in obj}
        for a, b in plcd:
            gi[a][b] = fgc
            go[a][b] = col
        inds -= plcd
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading taken from the golf entry (it differs from the per-component maker):
    # the WHOLE grid is one search. Piece types are tried in a fixed priority order:
    # first the 2x2 block -> 8, then the vertical 3-run -> 2, then the horizontal 3-run -> 2.
    # At each step we take the first remaining fgc piece of the highest-priority type
    # (in raster order) that still leaves every other fgc cell coverable. That check
    # replaces the golf's backtracking. Each op paints exactly that piece's fgc cells.
    # Every piece is named from the current grid, so O is never consulted.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    rem = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc}

    def pieces_at(r, c):
        return (
            [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)],  # 2x2 block
            [(r, c), (r + 1, c), (r + 2, c)],                  # vertical line
            [(r, c), (r, c + 1), (r, c + 2)],                  # horizontal line
        )

    @lru_cache(maxsize=None)
    def tileable(cells):
        if not cells:
            return True
        s = set(cells)
        r, c = min(s)
        for p in pieces_at(r, c):
            if all(q in s for q in p) and tileable(frozenset(s - set(p))):
                return True
        return False

    def comps(cells):
        cells = set(cells)
        out = []
        while cells:
            st = [cells.pop()]
            comp = set(st)
            while st:
                r, c = st.pop()
                for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if n in cells:
                        cells.remove(n)
                        comp.add(n)
                        st.append(n)
            out.append(frozenset(comp))
        return out

    def feasible(left):
        return all(tileable(k) for k in comps(left))

    ops, sels = [], []
    while rem:
        placed = False
        for kind, col in ((0, 8), (1, 2), (2, 2)):
            for (r, c) in sorted(rem):
                p = pieces_at(r, c)[kind]
                if all(q in rem for q in p) and feasible(rem - set(p)):
                    ops.append(col)
                    sels.append(sel_of(p))
                    rem -= set(p)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            break

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
