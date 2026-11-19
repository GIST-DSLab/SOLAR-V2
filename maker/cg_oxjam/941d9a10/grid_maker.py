"""
ARC Task: 941d9a10 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _line_mask(n, k, diff_lb, diff_ub):
    # n positions; nlines chosen as in generator; returns list of bool (True = line)
    opts = list(range(2, (n - 1) // 2 + 1, 2))
    idx = _unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nl = opts[idx]
    base = [(j % 2 == 1) for j in range(2 * nl + 1)]
    for _ in range(n - (2 * nl + 1)):
        loc = random.randint(0, len(base) - 1)
        base = base[:loc] + [False] + base[loc:]
    return base


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kw) -> dict:
    from collections import deque
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    rowline = _line_mask(h, None, diff_lb, diff_ub)
    colline = _line_mask(w, None, diff_lb, diff_ub)
    gi = [[fgc if (rowline[r] or colline[c]) else bgc for c in range(w)] for r in range(h)]
    # bgc components (8-connected as in generator; cells are rectangles anyway)
    seen = [[False] * w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if gi[r][c] == bgc and not seen[r][c]:
                q = deque([(r, c)]); seen[r][c] = True; cells = []
                while q:
                    a, b = q.popleft(); cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and not seen[na][nb] and gi[na][nb] == bgc:
                                seen[na][nb] = True; q.append((na, nb))
                objs.append(cells)
    left = lambda o: min(x[1] for x in o)
    up = lambda o: min(x[0] for x in o)
    blue = min(objs, key=lambda o: left(o) + up(o))
    green = max(objs, key=lambda o: left(o) + up(o))

    def ok(o):
        l, u = left(o), up(o)
        return (sum(left(x) < l for x in objs) == sum(left(x) > l for x in objs) and
                sum(up(x) < u for x in objs) == sum(up(x) > u for x in objs))
    red = next(o for o in objs if ok(o))
    go = [row[:] for row in gi]
    for col, o in ((1, blue), (3, green), (2, red)):
        for a, b in o:
            go[a][b] = col
    return {"input": gi, "output": go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = colour of the grid corners (always cell interiors)
    bgc = Counter([I[0, 0], I[0, -1], I[-1, 0], I[-1, -1]]).most_common(1)[0][0]

    # a separator row/column contains no background at all
    row_is_line = [not (I[r, :] == bgc).any() for r in range(hi)]
    col_is_line = [not (I[:, c] == bgc).any() for c in range(wi)]

    def bands(is_line):
        out, start = [], None
        for i, ln in enumerate(is_line + [True]):
            if not ln and start is None:
                start = i
            elif ln and start is not None:
                out.append((start, i - 1)); start = None
        return out

    rb = bands(row_is_line)
    cb = bands(col_is_line)

    def cell(ri, ci):
        (r0, r1), (c0, c1) = rb[ri], cb[ci]
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    ops, sels = [], []
    # walk the lattice diagonal: first band pair, middle band pair, last band pair
    targets = [
        (1, 0, 0),
        (2, len(rb) // 2, len(cb) // 2),
        (3, len(rb) - 1, len(cb) - 1),
    ]
    for color, ri, ci in targets:
        ops.append(color)
        sels.append(sel_of(cell(ri, ci)))   # whole band×band cell (all background)

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
                        f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
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
                                f"for task 941d9a10"
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
                    f"Failed to build a complete episode for task 941d9a10 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"941d9a10-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
