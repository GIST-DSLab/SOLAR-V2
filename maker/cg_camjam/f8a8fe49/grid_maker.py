"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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

# Reading taken from the code-golf entry: the work is done ONE LINE AT A TIME.
# Every line that crosses a box wall is reversed in a short window centred on
# the wall cell. That carries the pattern cells near the wall through the wall
# to the outside, and leaves the inside of the box empty.
# The maker treats each wall as a single band and flips the whole band at once.
# Here the unit is one interior line instead: a 1-wide strip centred on the
# wall, flipped about the wall cell. The window depth is the same on every line
# of one wall, as in the golf entry. The two readings agree on where the mirror
# is: the box line. They differ only in what one flip acts on.

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=5, boxc=2,
             transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(min(hi, max(lo, round(lo + d * (hi - lo)))))

    H, W = (max_w, max_h) if transpose else (max_h, max_w)
    H = min(H, 30); W = min(W, 30)
    h = unif(10, H)
    w = unif(4, W)
    fullh = unif(10, h)
    fullw = unif(3, w)
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    objh = (fullh // 2 - 3) // 2

    ci = [[bgc] * fullw for _ in range(fullh)]   # input: pattern inside the box
    co = [[bgc] * fullw for _ in range(fullh)]   # output: pattern reflected outside
    for _ in range(2):
        for g in (ci, co):
            for j in range(fullw):
                g[objh + 1][j] = boxc
            g[objh + 2][0] = boxc
            g[objh + 2][fullw - 1] = boxc
        cands = [(i, j) for i in range(objh) for j in range(1, fullw - 1)]
        n = len(cands)
        ncellsd = unif(0, n // 2)
        ncells = random.choice((ncellsd, n - ncellsd))
        ncells = min(max(1, ncells), n)
        cells = set(random.sample(cands, ncells))
        cells.add(random.choice([c for c in cands if c[0] == objh - 1]))
        for (i, j) in cells:
            co[i][j] = objc                       # outside
            ci[2 * objh + 2 - i][j] = objc        # exact mirror through box row objh+1
        ci = ci[::-1]
        co = co[::-1]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for i in range(fullh):
        for j in range(fullw):
            gi[loci + i][locj + j] = ci[i][j]
            go[loci + i][locj + j] = co[i][j]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # The pattern never reaches column 0 of the canvas and the box starts below
    # row 0, so the corner is always background.
    bgc = int(I[0, 0])

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    def area(col):
        b = bbox(col)
        return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)

    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})
    boxc = max(others, key=area)              # the box encloses the pattern
    objc = [c for c in others if c != boxc][0]

    r0, r1, c0, c1 = bbox(boxc)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))
    rs, cs = np.where(I == objc)

    ops, sels = [], []

    if horizontal:
        # Box walls are rows r0 and r1. Each interior column is one line that
        # crosses both walls.
        mid = (r0 + r1) / 2.0
        top = [int(r) for r in rs if r < mid]
        bot = [int(r) for r in rs if r > mid]
        sides = [(r0, max(top) - r0, 1), (r1, r1 - min(bot), -1)]
        for wall, m, inward in sides:
            for c in range(c0 + 1, c1):
                inner = [wall + inward * k for k in range(1, m + 1)]
                if not any(I[r, c] == objc for r in inner):
                    continue  # nothing on this line to reflect
                # Full 1-wide column window of 2m+1 cells centred on the wall
                # cell (background included). FlipV reflects it through the wall.
                ops.append(27); sels.append([wall - m, c, 2 * m, 0])
    else:
        # Box walls are columns c0 and c1. Each interior row is one line.
        mid = (c0 + c1) / 2.0
        left = [int(c) for c in cs if c < mid]
        right = [int(c) for c in cs if c > mid]
        sides = [(c0, max(left) - c0, 1), (c1, c1 - min(right), -1)]
        for wall, m, inward in sides:
            for r in range(r0 + 1, r1):
                inner = [wall + inward * k for k in range(1, m + 1)]
                if not any(I[r, c] == objc for c in inner):
                    continue
                # Full 1-tall row window of 2m+1 cells centred on the wall cell.
                # FlipH reflects it through the wall.
                ops.append(26); sels.append([r, wall - m, 0, 2 * m])

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
