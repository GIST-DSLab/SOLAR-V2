"""
ARC Task: 2c608aff (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of
import numpy as np


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    ccol = random.choice([c for c in cols if c != bgc])
    dcol = random.choice([c for c in cols if c not in (bgc, ccol)])
    return {"bgc": bgc, "ccol": ccol, "dcol": dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol, dcol) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    boxh = unifint(diff_lb, diff_ub, (2, max(2, h // 2)))
    boxw = unifint(diff_lb, diff_ub, (2, max(2, w // 2)))
    boxh = min(boxh, h)
    boxw = min(boxw, w)
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - backdrop(outbox(bd)))
    noiseb = max(1, len(reminds) // 4)
    # at least one noise cell, so the rule is always exercised (an empty
    # instance would be an identity mapping and unlearnable/degenerate)
    nnoise = unifint(diff_lb, diff_ub, (1, noiseb))
    nnoise = max(1, min(nnoise, len(reminds)))
    noise = sample(reminds, nnoise)
    gi = fill(gi, dcol, noise)
    go = tuple(e for e in gi)
    hs = interval(loci, loci + boxh, 1)
    ws = interval(locj, locj + boxw, 1)
    for ij in noise:
        a, b = ij
        if a in hs:
            go = fill(go, dcol, connect(ij, (a, locj)))
        elif b in ws:
            go = fill(go, dcol, connect(ij, (loci, b)))
    go = fill(go, ccol, bd)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: a solid rectangle (ccol) sits in a field of scattered noise pixels (dcol).
    Every noise pixel that lies in the rectangle's row band or column band is
    connected to the rectangle by a straight ray of its own colour.

    Note on ops: nothing translates here.  The verifier's `gravitate` only supplies
    the DIRECTION/LENGTH of the ray -- `connect(identity, identity + gravitate)`
    keeps the source pixel exactly where it is (it is still dcol in O) and paints
    the whole segment between it and the rectangle.  An ARCLE Move would vacate the
    source cell and restore the path to background, i.e. produce the wrong grid, so
    the honest route is one Color(dcol) per ray, selecting that ray's own cells.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- the rectangle: the colour whose cells exactly fill their own bounding box
    box = None
    for c in np.unique(I):
        cells = np.argwhere(I == c)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        area = int(r1 - r0 + 1) * int(c1 - c0 + 1)
        if len(cells) == area and area >= 4:
            if box is None or len(cells) > box[0]:
                box = (len(cells), int(r0), int(c0), int(r1), int(c1), int(c))
    if box is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    _, r0, c0, r1, c1, ccol = box

    # --- the noise colour: rarest colour other than the rectangle's
    counts = {int(c): int((I == c).sum()) for c in np.unique(I) if int(c) != ccol}
    if not counts:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    dcol = min(sorted(counts), key=lambda k: counts[k])

    # --- collect one ray per aligned noise pixel, grouped by side
    left, right, top, bottom = [], [], [], []
    for r, c in map(tuple, np.argwhere(I == dcol)):
        if r0 <= r <= r1:                       # row-aligned -> horizontal ray
            if c < c0:
                left.append((c0 - c, [(r, cc) for cc in range(c, c0)]))
            elif c > c1:
                right.append((c - c1, [(r, cc) for cc in range(c1 + 1, c + 1)]))
        elif c0 <= c <= c1:                     # column-aligned -> vertical ray
            if r < r0:
                top.append((r0 - r, [(rr, c) for rr in range(r, r0)]))
            elif r > r1:
                bottom.append((r - r1, [(rr, c) for rr in range(r1 + 1, r + 1)]))

    cur = I.copy()
    # side by side; within a side the longest ray first, so each op draws a whole
    # connecting line and shorter rays already covered by it need no op at all
    for group in (left, right, top, bottom):
        for _, ray in sorted(group, key=lambda t: -t[0]):
            todo = [(r, c) for (r, c) in ray if cur[r, c] != dcol]
            if not todo:
                continue
            ops.append(int(dcol))
            sels.append(sel_of(todo))
            for r, c in todo:
                cur[r, c] = dcol

    # whole grid is exactly this rectangle -> bbox form is the full region
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2c608aff"
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
                                f"for task 2c608aff"
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
                    f"Failed to build a complete episode for task 2c608aff "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2c608aff-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
