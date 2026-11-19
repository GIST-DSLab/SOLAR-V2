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
    {"horizontal": True,  "erase_bar": False},
    {"horizontal": False, "erase_bar": False},
    {"horizontal": True,  "erase_bar": True},
    {"horizontal": False, "erase_bar": True},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, col = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "col": col, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, col,
             horizontal=None, erase_bar=None) -> dict:
    def unif(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))

    if horizontal is None:
        horizontal = random.choice((True, False))
    if erase_bar is None:
        erase_bar = random.choice((True, False))
    max_h = max(8, min(30, max_h))
    max_w = max(8, min(30, max_w))
    h = unif(8, max_h)
    w = unif(8, max_w)
    ih = unif(5, h - 1)
    iw = unif(5, w - 1)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r1, c1 = loci + ih - 1, locj + iw - 1
    bx = set()
    for c in range(locj, c1 + 1):
        bx.add((loci, c)); bx.add((r1, c))
    for r in range(loci, r1 + 1):
        bx.add((r, locj)); bx.add((r, c1))
    if horizontal:
        locc = random.randint(loci + 2, loci + ih - 3)
        br = [(locc, c) for c in range(locj + 1, c1)]
    else:
        locc = random.randint(locj + 2, locj + iw - 3)
        br = [(r, locc) for r in range(loci + 1, r1)]
    crns = random.sample([(loci, locj), (loci, c1), (r1, locj), (r1, c1)], 3)
    rembx = sorted(bx - set(crns))
    onbr = random.sample(br, 2)
    rembr = sorted(set(br) - set(onbr))
    occbx = random.sample(rembx, unif(0, len(rembx)))
    occbr = random.sample(rembr, unif(0, len(rembr)))
    c = np.full((h, w), bgc, dtype=int)
    for p in bx:
        c[p] = col
    for p in br:
        c[p] = col
    gi = c.copy()
    go = c.copy()
    for p in occbx + occbr:
        gi[p] = bgc
        go[p] = 2
    if erase_bar:
        for p in br:
            gi[p] = bgc
            go[p] = bgc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Same rule as the reference (complete the outline, and the inner bar if
    there is one, in colour 2 under the existing strokes), carved by LAYER
    instead of by side: lay colour 2 along the WHOLE frame in one stroke, then
    along the WHOLE bar line, then put the original strokes back on top by
    repainting every cell that holds the foreground colour in I.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    FILL = 2                                              # constant fixed by the rule
    counts = Counter(I.flatten().tolist())
    col = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]   # leastcolor
    pts = [(int(r), int(c)) for r, c in np.argwhere(I == col)]
    ops, sels = [], []
    if pts:
        r0 = min(r for r, _ in pts); r1 = max(r for r, _ in pts)
        c0 = min(c for _, c in pts); c1 = max(c for _, c in pts)
        # the rectangle's full outline (3 corners always present -> bbox is the box)
        frame = sorted({(r0, c) for c in range(c0, c1 + 1)}
                       | {(r1, c) for c in range(c0, c1 + 1)}
                       | {(r, c0) for r in range(r0, r1 + 1)}
                       | {(r, c1) for r in range(r0, r1 + 1)})
        # the bar's full line, reconstructed from the col cells strictly inside
        inner = [(r, c) for r, c in pts if r0 < r < r1 and c0 < c < c1]
        bar = []
        if inner:
            rows = {r for r, _ in inner}
            if len(inner) == 1:
                horizontal = (r1 - r0 + 1) > (c1 - c0 + 1)   # rule's tie-break
            else:
                horizontal = len(rows) == 1
            rr, cc = inner[0]
            bar = ([(rr, c) for c in range(c0 + 1, c1)] if horizontal
                   else [(r, cc) for r in range(r0 + 1, r1)])
        gaps = [p for p in frame + bar if I[p] != col]
        if gaps:                                          # otherwise nothing to complete
            ops.append(FILL); sels.append(sel_of(frame))  # whole frame -> 2
            if bar:
                ops.append(FILL); sels.append(sel_of(bar))  # whole bar line -> 2
            ops.append(col); sels.append(sel_of(pts))     # original strokes back on top
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle: submit
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
