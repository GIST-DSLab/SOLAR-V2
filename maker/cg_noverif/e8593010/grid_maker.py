"""
ARC Task: e8593010 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of

_SHAPES_1 = [[(0, 0)]]
_SHAPES_2 = [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
_SHAPES_3 = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (0, 1)],
    [(1, 1), (1, 0), (0, 1)],
    [(1, 1), (0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2)],
]
_PRS = [(_SHAPES_1, 3), (_SHAPES_2, 2), (_SHAPES_3, 1)]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return max(a, min(b, random.randint(lo_i, hi_i)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    if bgc is None or fgc is None:
        cc = sample_colors()
        bgc = cc["bgc"] if bgc is None else bgc
        fgc = cc["fgc"] if fgc is None else fgc
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    reminds = {(r, c) for r in range(h) for c in range(w)}
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, ((h * w) // 2) // 2)))
    maxtr = 10
    for _ in range(nobjs):
        ntr = 0
        objs, col = random.choice(_PRS)
        obj = random.choice(objs)
        while ntr < maxtr:
            if not reminds:
                break
            loc = random.choice(sorted(reminds))
            olcd = {(loc[0] + dr, loc[1] + dc) for dr, dc in obj}
            if olcd.issubset(reminds):
                for r, c in olcd:
                    gi[r][c] = fgc
                    go[r][c] = col
                nb = set()
                for r, c in olcd:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nb.add((r + dr, c + dc))
                reminds = (reminds - olcd) - nb
                break
            ntr += 1
    return {"input": gi, "output": go}


def _components(G, fg_test):
    hi, wi = G.shape
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or not fg_test(G[r, c]):
                continue
            col = G[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and G[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Learn the conventions from the demonstrations: which colour marks objects
    # (the one that changes), and the size -> new colour map.
    fgc = None
    size_to_color = {}
    if examples:
        fg_counter = Counter()
        for ex_i, ex_o in examples:
            ex_i = np.asarray(ex_i, dtype=int)
            ex_o = np.asarray(ex_o, dtype=int)
            if ex_i.shape != ex_o.shape:
                continue
            diff = ex_i != ex_o
            fg_counter.update(ex_i[diff].tolist())
        if fg_counter:
            fgc = fg_counter.most_common(1)[0][0]
        if fgc is not None:
            for ex_i, ex_o in examples:
                ex_i = np.asarray(ex_i, dtype=int)
                ex_o = np.asarray(ex_o, dtype=int)
                if ex_i.shape != ex_o.shape:
                    continue
                for cells in _components(ex_i, lambda v: v == fgc):
                    r, c = cells[0]
                    size_to_color.setdefault(len(cells), int(ex_o[r, c]))
    if fgc is None or fgc not in set(I.flatten().tolist()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        fg_test = lambda v: v != bgc
    else:
        fg_test = lambda v: v == fgc
    for k, v in {1: 3, 2: 2, 3: 1}.items():
        size_to_color.setdefault(k, v)

    comps = _components(I, fg_test)

    # One whole object at a time, one size class at a time (smallest first):
    # each object is recoloured from a single seed by FloodFill.
    for size in sorted(size_to_color):
        tgt = size_to_color[size]
        for cells in comps:
            if len(cells) != size:
                continue
            sr, sc = cells[0]
            if I[sr, sc] == tgt:
                continue
            ops.append(10 + tgt)
            sels.append(sel_of([(sr, sc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task e8593010"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8593010"
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
                                f"for task e8593010"
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
                    f"Failed to build a complete episode for task e8593010 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8593010-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
