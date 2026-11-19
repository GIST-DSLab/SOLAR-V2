"""
ARC Task: 2bee17df (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cola = random.choice(rem)
    rem = [c for c in rem if c != cola]
    colb = random.choice(rem)
    return {"bgc": bgc, "cola": cola, "colb": colb}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cola, colb) -> dict:
    h = _unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = _unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    g = np.full((h, w), bgc, dtype=int)
    indord = [(0, j) for j in range(w)]
    indord += [(i, w - 1) for i in range(1, h - 1)]
    indord += [(h - 1, j) for j in range(w - 1, 0, -1)]
    indord += [(i, 0) for i in range(h - 1, 0, -1)]
    k = len(indord)
    sp = random.randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = random.randint(k // 2 - 3, k // 2 + 1)
    for (r, c) in arr[:ep]:
        g[r, c] = cola
    for (r, c) in arr[ep:]:
        g[r, c] = colb
    nr = _unifint(diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
    for kk in range(nr):
        r0, c0, r1, c1 = 1 + kk, 1 + kk, h - 1 - kk, w - 1 - kk
        ring = set()
        for j in range(c0, c1 + 1):
            ring.add((r0, j)); ring.add((r1, j))
        for i in range(r0, r1 + 1):
            ring.add((i, c0)); ring.add((i, c1))
        for br in (cola, colb):
            nb = set()
            for r in range(h):
                for c in range(w):
                    if g[r, c] == br:
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nb.add((r + dr, c + dc))
            bc = sorted(p for p in ring & nb
                        if 0 <= p[0] < h and 0 <= p[1] < w and g[p] == bgc)
            jj = len(bc)
            lo = max(0, jj // 2 - 2)
            hi_ = min(jj, jj // 2 + 1)
            jj2 = random.randint(lo, hi_)
            for p in random.sample(bc, jj2):
                g[p] = br
    go = g.copy()
    inner = g[1:h - 1, 1:w - 1]
    for i in range(inner.shape[0]):
        if np.all(inner[i] == bgc):
            go[i + 1, 1:w - 1] = 3
    for j in range(inner.shape[1]):
        if np.all(inner[:, j] == bgc):
            go[1:h - 1, j + 1] = 3
    return {"input": g.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    inner = I[1:hi - 1, 1:wi - 1]
    bgc = Counter(inner.flatten().tolist()).most_common(1)[0][0]
    rows = [i + 1 for i in range(inner.shape[0]) if np.all(inner[i] == bgc)]
    cols = [j + 1 for j in range(inner.shape[1]) if np.all(inner[:, j] == bgc)]
    cur = I.copy()
    # each fully-background interior row -> paint the whole interior row 3
    for r in rows:
        cells = [(r, c) for c in range(1, wi - 1)]
        ops.append(3); sels.append(sel_of(cells))
        for p in cells:
            cur[p] = 3
    # each fully-background interior column -> paint its cells not already 3
    for c in cols:
        cells = [(r, c) for r in range(1, hi - 1) if cur[r, c] != 3]
        if not cells:
            continue
        ops.append(3); sels.append(sel_of(cells))
        for p in cells:
            cur[p] = 3
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bee17df"
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
                                f"for task 2bee17df"
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
                    f"Failed to build a complete episode for task 2bee17df "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bee17df-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
