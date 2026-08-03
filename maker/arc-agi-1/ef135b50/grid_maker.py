"""
ARC Task: ef135b50 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import defaultdict


def sample_colors(num_examples=None) -> dict:
    # Generator samples: bgc, and ccols (pattern colors, drawn from cols - {bgc}).
    # The rule depends only on WHERE non-background cells are, never on their colors,
    # so only bgc must be pinned for the episode. 9 is reserved as the marker color.
    cols = [c for c in range(10) if c != 9]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = remove(9, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, max_h))
        w = unifint(diff_lb, diff_ub, (8, max_w))
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        nsq = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 30)))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        inds = asindices(gi)
        pats = set()
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = randint(1, max(1, h // 3 * 2))
            ow = randint(1, max(1, w // 3 * 2))
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            loci, locj = loc
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            if bd.issubset(inds):
                succ += 1
                inds = (inds - bd) - mapply(neighbors, bd)
                gi = fill(gi, choice(ccols), bd)
                pats.add(bd)
        res = set()
        ofc = ofcolor(gi, bgc)
        for pat1 in pats:
            for pat2 in remove(pat1, pats):
                if hmatching(pat1, pat2):
                    um = max(uppermost(pat1), uppermost(pat2))
                    bm = min(lowermost(pat1), lowermost(pat2))
                    lm = min(rightmost(pat1), rightmost(pat2)) + 1
                    rm = max(leftmost(pat1), leftmost(pat2)) - 1
                    res = res | backdrop(frozenset({(um, lm), (bm, rm)}))
        res = (res & ofc) - box(asindices(gi))
        go = fill(gi, 9, res)
        if go != gi:
            break
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I, O only used for its shape at Submit):
      - Background = the connected same-colour component that is NOT a solid rectangle
        and has the largest bounding-box area.  (Every placed pattern is a solid
        rectangle by construction, so this always selects the backdrop colour.)
      - In each interior row, the non-background blocks "see" each other: every
        background cell lying between the leftmost and rightmost non-background cell
        of that row becomes 9.  Grid-border rows are exempt.
      - Vertically stacked identical gaps form one rectangular corridor -> one Color9.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- 1. background colour: biggest non-rectangular component ---------------
    seen = np.zeros((h, w), dtype=bool)
    bgc = int(I[0, 0])
    best_area = -1
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bh = max(rs) - min(rs) + 1
            bw = max(cs) - min(cs) + 1
            if len(cells) == bh * bw:
                continue                      # solid rectangle -> a placed pattern
            if bh * bw > best_area:
                best_area = bh * bw
                bgc = col

    # --- 2. per interior row, background runs strictly inside the non-bg span ---
    runs = []
    for r in range(1, h - 1):
        occ = [c for c in range(w) if I[r, c] != bgc]
        if len(occ) < 2:
            continue
        lo, hi = occ[0], occ[-1]
        c = lo
        while c <= hi:
            if I[r, c] == bgc:
                s = c
                while c <= hi and I[r, c] == bgc:
                    c += 1
                runs.append((r, s, c - 1))
            else:
                c += 1

    # --- 3. merge vertically stacked identical runs into corridor rectangles ----
    by_span = defaultdict(list)
    for r, c0, c1 in runs:
        by_span[(c0, c1)].append(r)
    rects = []
    for (c0, c1), rows in by_span.items():
        rows.sort()
        start = prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                rects.append((start, c0, prev, c1))
                start = prev = r
        rects.append((start, c0, prev, c1))
    rects.sort(key=lambda t: (t[0], t[1]))

    # --- 4. one Color9 per corridor -------------------------------------------
    ops, sels = [], []
    for (r0, c0, r1, c1) in rects:
        ops.append(9)
        sels.append([r0, c0, r1 - r0, c1 - c0])

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ef135b50"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ef135b50"
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
                                f"for task ef135b50"
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
                    f"Failed to build a complete episode for task ef135b50 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ef135b50-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
