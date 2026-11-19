"""
ARC Task: 928ad970 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, dotc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if bgc is None or linc is None or dotc is None:
        bgc, linc, dotc = random.sample(range(10), 3)
    H = max(10, max_h)
    W = max(10, max_w)
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    ih = unifint(diff_lb, diff_ub, (9, h))
    iw = unifint(diff_lb, diff_ub, (9, w))
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0 = loci, locj
    r1, c1 = loci + ih - 1, locj + iw - 1
    dot1 = (random.randint(r0 + 1, r1 - 1), c0)   # left side
    dot2 = (r0, random.randint(c0 + 1, c1 - 1))   # top side
    dot3 = (r1, random.randint(c0 + 1, c1 - 1))   # bottom side
    dot4 = (random.randint(r0 + 1, r1 - 1), c1)   # right side

    rows = list(range(loci + 2, loci + ih - 2))
    while True:
        a, b = sorted(random.sample(rows, 2))
        if a + 1 != b:
            break
    cols = list(range(locj + 2, locj + iw - 2))
    while True:
        c, d = sorted(random.sample(cols, 2))
        if c + 1 != d:
            break

    gi = [[bgc] * w for _ in range(h)]
    for (r, cc) in (dot1, dot2, dot3, dot4):
        gi[r][cc] = dotc
    for cc in range(c, d + 1):
        gi[a][cc] = linc
        gi[b][cc] = linc
    for r in range(a, b + 1):
        gi[r][c] = linc
        gi[r][d] = linc

    go = [row[:] for row in gi]
    dots = [dot1, dot2, dot3, dot4]
    ra = min(p[0] for p in dots) + 1
    rb = max(p[0] for p in dots) - 1
    ca = min(p[1] for p in dots) + 1
    cb = max(p[1] for p in dots) - 1
    for cc in range(ca, cb + 1):
        go[ra][cc] = linc
        go[rb][cc] = linc
    for r in range(ra, rb + 1):
        go[r][ca] = linc
        go[r][cb] = linc
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    dotc = min(cnt, key=cnt.get)          # four marker dots: the rarest colour
    linc = next(c for c in cnt if c != bgc and c != dotc)  # the inner box's colour

    dots = [tuple(map(int, p)) for p in np.argwhere(I == dotc)]
    top = min(p[0] for p in dots)
    bot = max(p[0] for p in dots)
    lft = min(p[1] for p in dots)
    rgt = max(p[1] for p in dots)

    # Each side of the new frame lies one cell inside its own dot and spans
    # between the two perpendicular dots. Walk around the frame:
    # top -> left -> bottom -> right.
    r0, r1 = top + 1, bot - 1
    c0, c1 = lft + 1, rgt - 1
    sides = [
        [(r0, c) for c in range(c0, c1 + 1)],   # top side (under the top dot)
        [(r, c0) for r in range(r0, r1 + 1)],   # left side (right of the left dot)
        [(r1, c) for c in range(c0, c1 + 1)],   # bottom side (above the bottom dot)
        [(r, c1) for r in range(r0, r1 + 1)],   # right side (left of the right dot)
    ]

    ops, sels = [], []
    grid = I.copy()
    for cells in sides:
        if any(grid[r, c] != linc for r, c in cells):
            ops.append(int(linc)); sels.append(sel_of(cells))
            for r, c in cells:
                grid[r, c] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task 928ad970"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 928ad970"
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
                                f"for task 928ad970"
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
                    f"Failed to build a complete episode for task 928ad970 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"928ad970-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
