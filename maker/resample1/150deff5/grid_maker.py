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
from maker.sel_helpers import sel_of


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = frozenset({(0, 0), (0, 1), (0, 2)})
    ro2 = frozenset({(0, 0), (1, 0), (2, 0)})
    boforb, reforb = set(), set()
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    noccs = _unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 10)))
    inds = {(i, j) for i in range(h) for j in range(w)}
    for _ in range(noccs):
        if random.random() < 0.5:
            obj, col = bo, 8
        else:
            obj, col = random.choice((ro1, ro2)), 2
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        forb = boforb if col == 8 else reforb
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow
                 and (i, j) not in forb
                 and all((i + a, j + b) in inds for a, b in obj)]
        if not cands:
            break
        li, lj = random.choice(sorted(cands))
        if col == 8:
            for d in ((-2, 0), (2, 0), (0, 2), (0, -2)):
                boforb.add((li + d[0], lj + d[1]))
        elif obj == ro1:
            reforb.add((li, lj + 3)); reforb.add((li, lj - 3))
        else:
            reforb.add((li + 1, lj)); reforb.add((li - 1, lj))
        for a, b in obj:
            gi[li + a][lj + b] = fgc
            go[li + a][lj + b] = col
            inds.discard((li + a, lj + b))
    return {"input": gi, "output": go}


_PIECES = [
    ("box", [(0, 0), (0, 1), (1, 0), (1, 1)], 8),
    ("h", [(0, 0), (0, 1), (0, 2)], 2),
    ("v", [(0, 0), (1, 0), (2, 0)], 2),
]


def _cover(cells, limit=64):
    """Exact cover of a component by 2x2 boxes and 1x3/3x1 bars,
    respecting the generator's placement constraints."""
    sols = []

    def ok(placed, kind, r, c):
        for k2, r2, c2 in placed:
            if kind == "box" and k2 == "box" and (abs(r - r2), abs(c - c2)) in ((2, 0), (0, 2)):
                return False
            if kind == "h" and k2 == "h" and r == r2 and abs(c - c2) == 3:
                return False
        return True

    def rec(rem, placed):
        if len(sols) >= limit:
            return
        if not rem:
            sols.append(list(placed))
            return
        r, c = min(rem)  # top-left-most uncovered cell must be a piece's anchor
        for kind, off, col in _PIECES:
            pc = {(r + a, c + b) for a, b in off}
            if pc <= rem and ok(placed, kind, r, c):
                placed.append((kind, r, c))
                rec(rem - pc, placed)
                placed.pop()

    rec(set(cells), [])
    return sols


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    # generator: canvas of bgc with a single foreground colour drawn on it
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = {(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc}

    # 4-connected foreground components
    seen = set()
    comps = []
    for s in sorted(fg):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) in fg and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(sorted(comp))

    ops, sels = [], []
    off = {k: (o, col) for k, o, col in _PIECES}
    for comp in comps:
        sols = _cover(comp)
        if not sols:
            continue
        chosen = sols[0]
        if len(sols) > 1:  # rare genuinely ambiguous tiling: take the consistent one
            for s in sols:
                if all(O[r + a, c + b] == off[k][1] for k, r, c in s for a, b in off[k][0]):
                    chosen = s
                    break
        # recolour each piece as a unit: boxes -> 8, bars -> 2
        for k, r, c in chosen:
            o, col = off[k]
            ops.append(col)
            sels.append(sel_of([(r + a, c + b) for a, b in o]))
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
