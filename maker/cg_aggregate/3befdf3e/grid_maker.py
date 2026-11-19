"""
ARC Task: 3befdf3e (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # The rule (swap frame/core colours, grow arms) never depends on which
    # foreground colours are used, so only the background is fixed per episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 40)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        if not inds:
            break
        rh = random.choice((1, 2))
        rw = random.choice((1, 2))
        fullh = 2 + 3 * rh
        fullw = 2 + 3 * rw
        cands = [ij for ij in inds if ij[0] <= h - fullh and ij[1] <= w - fullw]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        fullobj = {(a, b) for a in range(loci, loci + fullh) for b in range(locj, locj + fullw)}
        if not fullobj.issubset(inds):
            continue
        succ += 1
        inds -= fullobj
        incol, outcol = random.sample(ccols, 2)
        i0, j0 = loci + rh + 1, locj + rw + 1
        i1, j1 = loci + 2 * rh, locj + 2 * rw
        ofincol = {(a, b) for a in range(i0, i1 + 1) for b in range(j0, j1 + 1)}
        ofoutcol = {(a, b) for a in range(i0 - 1, i1 + 2) for b in range(j0 - 1, j1 + 2)} - ofincol
        for a, b in ofincol:
            gi[a][b] = incol
            go[a][b] = outcol
        for a, b in ofoutcol:
            gi[a][b] = outcol
            go[a][b] = incol
        ilocs = {p[0] for p in ofoutcol}
        jlocs = {p[1] for p in ofoutcol}
        for a, b in fullobj - (ofincol | ofoutcol):
            if a in ilocs or b in jlocs:
                go[a][b] = outcol
    if succ == 0:
        raise ValueError("no object placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: framed boxes cover a small fraction of the canvas
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    seen = np.zeros((hi, wi), dtype=bool)
    boxes = []
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            boxes.append((min(rs), max(rs), min(cs), max(cs)))

    for r0, r1, c0, c1 in boxes:
        outcol = int(I[r0, c0])
        incol = int(I[r0 + 1, c0 + 1])
        rh = r1 - r0 - 1
        rw = c1 - c0 - 1

        ring = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)
                if a in (r0, r1) or b in (c0, c1)]
        inner = [(a, b) for a in range(r0 + 1, r1) for b in range(c0 + 1, c1)]

        # 1) swap: frame takes the core colour, core takes the frame colour
        ops.append(incol); sels.append(sel_of(ring))
        ops.append(outcol); sels.append(sel_of(inner))

        # 2) grow one arm per side, going round: up, right, down, left.
        #    Each arm is a full rectangle: the box's span on that side,
        #    reaching outward by the core's size in that direction.
        up = [(a, b) for a in range(max(0, r0 - rh), r0) for b in range(c0, c1 + 1)]
        right = [(a, b) for a in range(r0, r1 + 1) for b in range(c1 + 1, min(wi, c1 + 1 + rw))]
        down = [(a, b) for a in range(r1 + 1, min(hi, r1 + 1 + rh)) for b in range(c0, c1 + 1)]
        left = [(a, b) for a in range(r0, r1 + 1) for b in range(max(0, c0 - rw), c0)]
        for arm in (up, right, down, left):
            if arm and any(I[a, b] != outcol for a, b in arm):
                ops.append(outcol); sels.append(sel_of(arm))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
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
                                f"for task 3befdf3e"
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
                    f"Failed to build a complete episode for task 3befdf3e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3befdf3e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
