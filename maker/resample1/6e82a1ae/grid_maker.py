"""
ARC Task: 6e82a1ae (RE-ARC) — LLM-generated grid_maker
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
    cols = [0, 4, 5, 6, 7, 8, 9]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _rand_shape4():
    shp = {(0, 0)}
    for _ in range(3):
        cand = set()
        for (r, c) in shp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cand.add((r + dr, c + dc))
        cand -= shp
        shp.add(random.choice(sorted(cand)))
    mr = min(r for r, _ in shp)
    mc = min(c for _, c in shp)
    return frozenset((r - mr, c - mc) for r, c in shp)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=5, **kw) -> dict:
    b = [frozenset({(0, 0), (0, 1)}), frozenset({(0, 0), (1, 0)})]
    c = [
        frozenset({(0, 0), (1, 0), (1, 1)}),
        frozenset({(0, 0), (1, 0), (0, 1)}),
        frozenset({(1, 1), (1, 0), (0, 1)}),
        frozenset({(1, 1), (0, 0), (0, 1)}),
        frozenset({(0, 0), (1, 0), (2, 0)}),
        frozenset({(0, 0), (0, 1), (0, 2)}),
    ]
    d = list({_rand_shape4() for _ in range(100)})
    prs = [(b, 3), (c, 2), (d, 1)]
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    reminds = {(r, cc) for r in range(h) for cc in range(w)}
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, ((h * w) // 2) // 3)))
    for _ in range(nobjs):
        objs, col = random.choice(prs)
        obj = random.choice(objs)
        for _t in range(10):
            if not reminds:
                break
            loc = random.choice(sorted(reminds))
            olcd = {(r + loc[0], cc + loc[1]) for r, cc in obj}
            if olcd.issubset(reminds):
                for r, cc in olcd:
                    gi[r, cc] = fgc
                    go[r, cc] = col
                nb = set()
                for r, cc in olcd:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb.add((r + dr, cc + dc))
                reminds = (reminds - olcd) - nb
                break
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = sorted(set(I.flatten().tolist()))
    bgc = None
    if examples:
        # background = colour whose cells are always left unchanged in the demos
        cands = []
        for col in colors:
            ok = True
            seen = False
            for ei, eo in examples:
                ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
                m = ei == col
                if m.any():
                    seen = True
                    if not np.all(eo[m] == col):
                        ok = False
                        break
            if ok and seen:
                cands.append(col)
        if len(cands) == 1:
            bgc = cands[0]
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(sorted(cells))

    size_to_col = {2: 3, 3: 2, 4: 1}
    ops, sels = [], []
    for cells in comps:
        nc = size_to_col.get(len(cells))
        if nc is None:
            continue
        r0, c0 = cells[0]
        if I[r0, c0] == nc:
            continue
        ops.append(10 + nc)
        sels.append([r0, c0, 0, 0])  # single seed cell for FloodFill of this object
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
                        f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
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
                                f"for task 6e82a1ae"
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
                    f"Failed to build a complete episode for task 6e82a1ae "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e82a1ae-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
