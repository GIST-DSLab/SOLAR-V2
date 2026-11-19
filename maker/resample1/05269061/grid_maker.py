"""
ARC Task: 05269061 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"diag": True, "mirror": False},
    {"diag": True, "mirror": True},
    {"diag": False, "mirror": False},
]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}


def _unifint(lb, ub, bounds):
    a, b = bounds
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, round(random.uniform(lo, hi))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, diag=None, mirror=None, **kw) -> dict:
    if diag is None or mirror is None:
        v = random.choice(VARIANTS)
        diag, mirror = v["diag"], v["mirror"]
    colopts = list(range(1, 10))
    d = _unifint(diff_lb, diff_ub, (2, min(max_h, max_w, 30)))
    go = [[0] * d for _ in range(d)]
    gi = [[0] * d for _ in range(d)]
    if diag:
        num = _unifint(diff_lb, diff_ub, (2, min(2 * d - 2, 9)))
        nlines = 2 * d - 1
    else:
        num = _unifint(diff_lb, diff_ub, (2, min(d, 9)))
        nlines = d
    cols = [random.choice(colopts) for _ in range(num)]
    keeps = [random.choice(list(range(j, nlines, num))) for j in range(num)]
    for k in range(nlines):
        col = cols[k % num]
        if diag:
            cells = [(r, k - r) for r in range(d) if 0 <= k - r < d]
        else:
            cells = [(k, c) for c in range(d)]
        for r, c in cells:
            go[r][c] = col
            if keeps[k % num] == k:
                gi[r][c] = col
    if mirror:
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]
    return {"input": gi, "output": go}


def _line_cells(mode, k, d):
    if mode == "h":
        return [(k, c) for c in range(d)]
    if mode == "a":   # anti-diagonal r + c = k
        return [(r, k - r) for r in range(d) if 0 <= k - r < d]
    # main diagonal (mirrored): r + (d-1-c) = k
    return [(r, d - 1 - (k - r)) for r in range(d) if 0 <= k - r < d]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    d = I.shape[0]

    # orientation: the line family along which the input is constant
    def consistent(mode):
        n = d if mode == "h" else 2 * d - 1
        for k in range(n):
            vals = {int(I[r, c]) for r, c in _line_cells(mode, k, d)}
            if len(vals) > 1:
                return False
        return True

    mode = next((m for m in ("h", "a", "m") if consistent(m)), "h")
    n = d if mode == "h" else 2 * d - 1
    kept = {}
    for k in range(n):
        r, c = _line_cells(mode, k, d)[0]
        if I[r, c] != 0:
            kept[k] = int(I[r, c])
    # one kept line per residue class -> period = number of kept lines
    period = len(kept)
    ops, sels = [], []
    if period > 0:
        # extend each kept line periodically outward, nearest copies first
        for k0, col in sorted(kept.items()):
            targets = [k for k in range(n) if k % period == k0 % period and k != k0]
            targets.sort(key=lambda k: (abs(k - k0), k))
            for k in targets:
                cells = [(r, c) for r, c in _line_cells(mode, k, d) if I[r, c] != col]
                if cells:
                    ops.append(col); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, d - 1, d - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 05269061"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05269061"
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
                                f"for task 05269061"
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
                    f"Failed to build a complete episode for task 05269061 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05269061-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
