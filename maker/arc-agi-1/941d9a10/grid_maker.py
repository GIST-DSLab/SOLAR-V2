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
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    opts = interval(2, (h - 1) // 2 + 1, 2)
    nhidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nh = opts[nhidx]
    opts = interval(2, (w - 1) // 2 + 1, 2)
    nwidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nw = opts[nwidx]
    hgrid = canvas(bgc, (2 * nh + 1, w))
    for j in range(1, h, 2):
        hgrid = fill(hgrid, fgc, connect((j, 0), (j, w)))
    for k in range(h - (2 * nh + 1)):
        loc = randint(0, height(hgrid) - 1)
        hgrid = hgrid[:loc] + canvas(bgc, (1, w)) + hgrid[loc:]
    wgrid = canvas(bgc, (2 * nw + 1, h))
    for j in range(1, w, 2):
        wgrid = fill(wgrid, fgc, connect((j, 0), (j, h)))
    for k in range(w - (2 * nw + 1)):
        loc = randint(0, height(wgrid) - 1)
        wgrid = wgrid[:loc] + canvas(bgc, (1, h)) + wgrid[loc:]
    wgrid = dmirror(wgrid)
    gi = canvas(bgc, (h, w))
    fronts = ofcolor(hgrid, fgc) | ofcolor(wgrid, fgc)
    gi = fill(gi, fgc, fronts)
    objs = objects(gi, T, T, F)
    objs = colorfilter(objs, bgc)
    blue = argmin(objs, lambda o: leftmost(o) + uppermost(o))
    green = argmax(objs, lambda o: leftmost(o) + uppermost(o))
    f1 = lambda o: len(sfilter(objs, lambda o2: leftmost(o2) < leftmost(o))) == len(sfilter(objs, lambda o2: leftmost(o2) > leftmost(o)))
    f2 = lambda o: len(sfilter(objs, lambda o2: uppermost(o2) < uppermost(o))) == len(sfilter(objs, lambda o2: uppermost(o2) > uppermost(o)))
    red = extract(objs, lambda o: f1(o) and f2(o))
    go = fill(gi, 1, blue)
    go = fill(go, 3, green)
    go = fill(go, 2, red)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # bgc = color at grid corners (cell interiors, never grid lines)
    corner_cells = [I[0, 0], I[0, wi - 1], I[hi - 1, 0], I[hi - 1, wi - 1]]
    bgc = Counter(corner_cells).most_common(1)[0][0]

    # segment bgc-colored cells into 4-connected components (from I, not O)
    seen = np.zeros((hi, wi), dtype=bool)
    info = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and I[nr, nc] == bgc and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                rows = [p[0] for p in cells]
                colsl = [p[1] for p in cells]
                info.append({
                    'left': min(colsl), 'up': min(rows),
                    'r0': min(rows), 'c0': min(colsl),
                    'r1': max(rows), 'c1': max(colsl),
                })

    # blue = top-left (min leftmost+uppermost), green = bottom-right (max)
    blue = min(info, key=lambda o: o['left'] + o['up'])
    green = max(info, key=lambda o: o['left'] + o['up'])

    # red = median cell: equal #cells strictly-left/right and strictly-above/below
    def red_ok(o):
        lt = sum(1 for x in info if x['left'] < o['left'])
        gt = sum(1 for x in info if x['left'] > o['left'])
        ut = sum(1 for x in info if x['up'] < o['up'])
        dt = sum(1 for x in info if x['up'] > o['up'])
        return lt == gt and ut == dt
    red = next(o for o in info if red_ok(o))

    ops, sels = [], []
    for color, o in ((1, blue), (3, green), (2, red)):
        ops.append(color)
        sels.append([o['r0'], o['c0'], o['r1'] - o['r0'], o['c1'] - o['c0']])

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
