"""
ARC Task: fcb5c309 (RE-ARC) — LLM-generated grid_maker
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

try:
    from dsl import *
    from utils import *
except Exception:
    pass

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, sqc = random.sample(cols, 3)
    return {"bgc": bgc, "dotc": dotc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, sqc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    go = None
    while tr < maxtr and succ < numsq:
        oh = randint(3, 7)
        ow = randint(3, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(sq)
        if bd.issubset(inds):
            gi = fill(gi, sqc, sq)
            ib = backdrop(inbox(sq))
            if numcells is None:
                numcells = unifint(diff_lb, diff_ub, (1, len(ib)))
                cells = sample(totuple(ib), numcells)
                take = True
            else:
                nc = unifint(diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = sample(totuple(ib), nc)
            gi = fill(gi, dotc, cells)
            if take:
                go = replace(subgrid(sq, gi), sqc, dotc)
                take = False
            inds = (inds - bd) - outbox(bd)
            succ += 1
        tr += 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = sample(totuple(inds), nnoise)
    gi = fill(gi, dotc, noise)
    return {"input": gi, "output": go}


def _components(I, color):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == color and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < hi and 0 <= nb < wi and I[na, nb] == color and not seen[na, nb]:
                            seen[na, nb] = True
                            stack.append((na, nb))
                comps.append(cells)
    return comps


def _is_box(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
        return False
    outline = set()
    for r in range(r0, r1 + 1):
        outline.add((r, c0))
        outline.add((r, c1))
    for c in range(c0, c1 + 1):
        outline.add((r0, c))
        outline.add((r1, c))
    return set(cells) == outline


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    colors = [c for c in cnt if c != bgc]

    # frame color: the non-bg color forming box outlines (largest count of boxes)
    best_col = colors[0]
    best_boxes = []
    best_n = -1
    for col in colors:
        comps = _components(I, col)
        boxes = [cc for cc in comps if _is_box(cc)]
        if len(boxes) > best_n:
            best_n = len(boxes)
            best_col = col
            best_boxes = boxes
    sqc = best_col

    remaining = [c for c in colors if c != sqc]
    dotc = min(remaining, key=lambda c: cnt[c]) if remaining else sqc

    # choose the box whose bounding rectangle contains the most dot-color cells
    def rect_dot(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        return int(np.sum(I[r0:r1 + 1, c0:c1 + 1] == dotc))

    chosen = max(best_boxes, key=rect_dot)
    rs = [r for r, c in chosen]
    cs = [c for r, c in chosen]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1

    ops = []
    sels = []

    # recolor this box's frame outline (sqc) -> dotc, on the true outline cells only
    ops.append(int(dotc))
    sels.append(sel_of(chosen))

    # crop to the box's full bounding rectangle (includes interior bg + dots)
    ops.append(33)
    sels.append([r0, c0, bh - 1, bw - 1])

    ops.append(34)
    sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
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
                                f"for task fcb5c309"
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
                    f"Failed to build a complete episode for task fcb5c309 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fcb5c309-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
