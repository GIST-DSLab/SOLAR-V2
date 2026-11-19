"""
ARC Task: 4612dd53 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"keep_bar": True,  "horizontal": True},
    {"keep_bar": True,  "horizontal": False},
    {"keep_bar": False, "horizontal": True},
    {"keep_bar": False, "horizontal": False},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, col = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "col": col, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, col,
             keep_bar=None, horizontal=None, **kwargs) -> dict:
    def unifint(lo, hi):
        lo_v, hi_v = lo, hi
        if hi_v < lo_v:
            hi_v = lo_v
        a = int(round(lo_v + diff_lb * (hi_v - lo_v)))
        b = int(round(lo_v + diff_ub * (hi_v - lo_v)))
        return random.randint(min(a, b), max(a, b))

    if keep_bar is None:
        keep_bar = random.choice((True, False))
    if horizontal is None:
        horizontal = random.choice((True, False))

    h = unifint(8, max(8, max_h))
    w = unifint(8, max(8, max_w))
    ih = unifint(5, h - 1)
    iw = unifint(5, w - 1)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1
    bx = set()
    for c in range(c0, c1 + 1):
        bx.add((r0, c)); bx.add((r1, c))
    for r in range(r0, r1 + 1):
        bx.add((r, c0)); bx.add((r, c1))
    if horizontal:
        locc = random.randint(r0 + 2, r1 - 2)
        br = [(locc, c) for c in range(c0 + 1, c1)]
    else:
        locc = random.randint(c0 + 2, c1 - 2)
        br = [(r, locc) for r in range(r0 + 1, r1)]
    br_set = set(br)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    crns = random.sample(corners, 3)
    rembx = sorted(bx - set(crns))
    onbr = random.sample(br, 2)
    rembr = sorted(br_set - set(onbr))
    noccbx = unifint(0, len(rembx))
    noccbr = unifint(0, len(rembr))
    occbx = random.sample(rembx, noccbx)
    occbr = random.sample(rembr, noccbr)

    c = np.full((h, w), bgc, dtype=int)
    for (r, cc) in bx | br_set:
        c[r, cc] = col
    gi = c.copy()
    go = c.copy()
    for (r, cc) in occbx + occbr:
        gi[r, cc] = bgc
        go[r, cc] = 2
    if not keep_bar:
        for (r, cc) in br:
            gi[r, cc] = bgc
            go[r, cc] = bgc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    The rarest colour draws a broken rectangle (3 corners always present, so its
    bounding box is the rectangle) plus, optionally, a broken straight bar.
    Rule: the background gaps of the rectangle outline and of the bar's full
    line are filled with the constant colour 2.
    Two strokes: Color2 on the outline's background cells, Color2 on the
    bar line's background cells. Both selections are read from I alone.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    FILL = 2

    counts = Counter(I.flatten().tolist())
    col = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    pts = np.argwhere(I == col)
    ops, sels = [], []
    if len(pts) == 0:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    # whole rectangle outline, its gap (non-col) cells
    outline = set()
    for c in range(c0, c1 + 1):
        outline.add((r0, c)); outline.add((r1, c))
    for r in range(r0, r1 + 1):
        outline.add((r, c0)); outline.add((r, c1))
    outline_gaps = sorted(p for p in outline if I[p] != col)

    # the bar line, from the col cells strictly inside the rectangle
    inner = [(int(r), int(c)) for r, c in pts if r0 < r < r1 and c0 < c < c1]
    bar_gaps = []
    if inner:
        rows = {r for r, _ in inner}
        cols = {c for _, c in inner}
        if len(rows) == 1 and len(cols) == 1:
            horizontal = (r1 - r0 + 1) > (c1 - c0 + 1)
        else:
            horizontal = (len(rows) == 1)
        rr, cc = inner[0]
        if horizontal:
            line = [(rr, c) for c in range(c0 + 1, c1)]
        else:
            line = [(r, cc) for r in range(r0 + 1, r1)]
        bar_gaps = [p for p in line if I[p] != col]

    if outline_gaps:
        ops.append(FILL); sels.append(sel_of(outline_gaps))
    if bar_gaps:
        ops.append(FILL); sels.append(sel_of(bar_gaps))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
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
                                f"for task 4612dd53"
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
                    f"Failed to build a complete episode for task 4612dd53 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4612dd53-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
