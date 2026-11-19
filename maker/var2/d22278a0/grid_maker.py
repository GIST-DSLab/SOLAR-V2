"""
ARC Task: d22278a0 (RE-ARC) — LLM-generated grid_maker
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


# how many corners carry a colour — the one discrete structural variant of this task
VARIANTS = [{"ncorns": 1}, {"ncorns": 2}, {"ncorns": 3}, {"ncorns": 4}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, 4)          # palette for the (up to four) marked corners

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
        if all(e["ncorns"] == 1 for e in examples):
            # the nearest-corner tie-break is invisible with a single corner: show a multi case
            examples[0] = {"ncorns": random.choice([2, 3, 4])}

    plan = examples + [dict(random.choice(examples))]   # test case is one that was shown
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None, ncorns=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    if bgc is None:
        bgc = random.choice(list(cols))
    remcols = [c for c in cols if c != bgc]
    if ccols is None:
        ccols = random.sample(remcols, 4)
    ccols = [c for c in ccols if c != bgc]
    while len(ccols) < 4:
        extra = random.choice([c for c in remcols if c not in ccols])
        ccols.append(extra)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    crns = corners(inds)
    if ncorns is None:
        ncorns = unifint(diff_lb, diff_ub, (1, 4))
    ncorns = max(1, min(4, int(ncorns)))
    crns = random.sample(totuple(crns), ncorns)
    ccs = ccols[:ncorns]

    for col, crn in zip(ccs, crns):
        gi = fill(gi, col, {crn})
        go = fill(go, col, {crn})
        rings = {crn}
        for k in range(1, max(h, w) // 2 + 2, 1):
            rings = rings | outbox(outbox(rings))
        if len(crns) > 1:
            ff = lambda ij, crn=crn: manhattan({ij}, {crn}) < min(
                apply(rbind(manhattan, {ij}), apply(initset, remove(crn, crns)))
            )
        else:
            ff = lambda ij: True
        locs = sfilter(inds, ff) & rings
        go = fill(go, col, locs)

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I alone):
      every corner cell whose colour differs from the background carries a colour.  That
      colour is stamped on the concentric square rings of even Chebyshev distance around
      that corner, kept only where the cell is strictly Manhattan-nearer to that corner
      than to any other marked corner.

    The rule is fixed; the ROUTE through it is not, exactly as two people would differ.
    Four choices, each read off the input grid (never off O), so one instance always
    yields one route and the variety comes from the instances:
      * which corner to start with  (reading order / by colour / reverse reading order)
      * sweep outward from each corner or inward from the outermost ring
      * paint a whole ring in one stroke, or its two arms (the run across, the run down)
      * finish one corner before the next, or grow every corner together ring by ring
    Colours are laid with Color ops on the exact cells, so a corner coloured 0 is put
    down by an explicit Color0 — nothing here is carried by Paste/Move/Copy.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # only corners are ever marked (grid is at least 4x4), so any inner cell is background
    bgc = int(I[1, 1])

    corner_pos = [(0, 0), (0, wi - 1), (hi - 1, 0), (hi - 1, wi - 1)]
    marked = [p for p in corner_pos if int(I[p[0], p[1]]) != bgc]
    if not marked:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    cols = {p: int(I[p[0], p[1]]) for p in marked}

    maxd = max(hi, wi)

    def territory(p):
        """ring index d -> the cells of that ring this corner owns (d=0 is already in I)"""
        cr, cc = p
        others = [q for q in marked if q != p]
        rings = {}
        for d in range(2, maxd + 1, 2):
            cells = []
            for r in range(max(0, cr - d), min(hi, cr + d + 1)):
                for c in range(max(0, cc - d), min(wi, cc + d + 1)):
                    if max(abs(r - cr), abs(c - cc)) != d:
                        continue
                    m = abs(r - cr) + abs(c - cc)
                    if all(m < abs(r - orr) + abs(c - occ) for (orr, occ) in others):
                        cells.append((r, c))
            if cells:
                rings[d] = cells
        return rings

    # --- route choices, all derived from the input picture -------------------
    csum = sum(cols.values())
    cmax = max(cols.values())
    n = len(marked)
    order_mode    = (csum + n) % 3
    outward       = ((hi + wi) % 2 == 0)
    split_arms    = ((cmax + wi) % 2 == 1)
    ring_major    = (n > 1 and (csum + hi) % 2 == 1)
    row_arm_first = ((csum + hi + wi) % 2 == 0)

    if order_mode == 0:
        order = sorted(marked)                                    # reading order
    elif order_mode == 1:
        order = sorted(marked, key=lambda q: (cols[q], q))        # palest colour first
    else:
        order = sorted(marked, reverse=True)                      # reverse reading order

    terr = {p: territory(p) for p in order}

    def emit(p, d, cells):
        cr, cc = p
        col = cols[p]
        if not split_arms:
            ops.append(col)                       # Color<col> over this ring segment
            sels.append(sel_of(cells))
            return
        row_arm = [x for x in cells if abs(x[0] - cr) == d]   # the run across (holds the elbow)
        col_arm = [x for x in cells if abs(x[0] - cr) != d]   # the run down
        for arm in ([row_arm, col_arm] if row_arm_first else [col_arm, row_arm]):
            if arm:
                ops.append(col)
                sels.append(sel_of(arm))

    if ring_major:
        all_d = sorted({d for p in order for d in terr[p]}, reverse=not outward)
        for d in all_d:                                   # every corner grows together
            for p in order:
                if d in terr[p]:
                    emit(p, d, terr[p][d])
    else:
        for p in order:                                   # one corner finished at a time
            for d in sorted(terr[p], reverse=not outward):
                emit(p, d, terr[p][d])

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
                        f"num_examples+1 ({num_examples + 1}) for task d22278a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d22278a0"
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
                                f"for task d22278a0"
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
                    f"Failed to build a complete episode for task d22278a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d22278a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
