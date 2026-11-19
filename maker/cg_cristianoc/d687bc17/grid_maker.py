"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c1, connect((0, 0), (0, w - 1)))
    gi = fill(gi, c2, connect((0, 0), (h - 1, 0)))
    gi = fill(gi, c3, connect((h - 1, w - 1), (0, w - 1)))
    gi = fill(gi, c4, connect((h - 1, w - 1), (h - 1, 0)))
    inds = asindices(gi)
    gi = fill(gi, bgc, corners(inds))
    go = tuple(e for e in gi)
    cands = backdrop(inbox(inbox(inds)))
    ndots = unifint(diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    dots = sample(totuple(cands), ndots)
    dots = {(choice((c1, c2, c3, c4)), ij) for ij in dots}
    n1 = toindices(sfilter(dots, lambda cij: cij[0] == c1))
    n1coverage = apply(last, n1)
    if len(n1coverage) == w - 4 and w > 5:
        n1coverage = remove(choice(totuple(n1coverage)), n1coverage)
    for jj in n1coverage:
        loci = choice([ij[0] for ij in sfilter(n1, lambda ij: ij[1] == jj)])
        gi = fill(gi, c1, {(loci, jj)})
        go = fill(go, c1, {(1, jj)})
    n2 = toindices(sfilter(dots, lambda cij: cij[0] == c2))
    n2coverage = apply(first, n2)
    if len(n2coverage) == h - 4 and h > 5:
        n2coverage = remove(choice(totuple(n2coverage)), n2coverage)
    for ii in n2coverage:
        locj = choice([ij[1] for ij in sfilter(n2, lambda ij: ij[0] == ii)])
        gi = fill(gi, c2, {(ii, locj)})
        go = fill(go, c2, {(ii, 1)})
    n3 = toindices(sfilter(dots, lambda cij: cij[0] == c4))
    n3coverage = apply(last, n3)
    if len(n3coverage) == w - 4 and w > 5:
        n3coverage = remove(choice(totuple(n3coverage)), n3coverage)
    for jj in n3coverage:
        loci = choice([ij[0] for ij in sfilter(n3, lambda ij: ij[1] == jj)])
        gi = fill(gi, c4, {(loci, jj)})
        go = fill(go, c4, {(h - 2, jj)})
    n4 = toindices(sfilter(dots, lambda cij: cij[0] == c3))
    n4coverage = apply(first, n4)
    if len(n4coverage) == h - 4 and h > 5:
        n4coverage = remove(choice(totuple(n4coverage)), n4coverage)
    for ii in n4coverage:
        locj = choice([ij[1] for ij in sfilter(n4, lambda ij: ij[0] == ii)])
        gi = fill(gi, c3, {(ii, locj)})
        go = fill(go, c3, {(ii, w - 2)})
    noisecands = ofcolor(gi, bgc)
    noisecols = difference(cols, (bgc, c1, c2, c3, c4))
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
    noise = sample(totuple(noisecands), nnoise)
    noiseobj = {(choice(noisecols), ij) for ij in noise}
    gi = paint(gi, noiseobj)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # border side colours (border cells other than corners never carry noise)
    c1 = int(I[0, 1])        # top row
    c2 = int(I[1, 0])        # left column
    c3 = int(I[1, w - 1])    # right column
    c4 = int(I[h - 1, 1])    # bottom row
    interior_vals = [int(I[r, c]) for r in range(1, h - 1) for c in range(1, w - 1)]
    bgc = Counter(interior_vals).most_common(1)[0][0]
    side_cols = {c1, c2, c3, c4}

    ops, sels = [], []

    # ---- 1. clear everything that belongs to no border: interior noise + stray corners ----
    junk = [(r, c) for (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
            if int(I[r, c]) != bgc]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v not in side_cols:
                junk.append((r, c))
    if junk:
        ops.append(bgc)
        sels.append(sel_of(junk))

    # ---- 2. interior objects = 4-connected same-colour components of a border colour ----
    seen = set()
    comps = {c1: [], c2: [], c3: [], c4: []}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v not in side_cols or (r, c) in seen:
                continue
            stack, cells = [(r, c)], []
            seen.add((r, c))
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 1 <= na < h - 1 and 1 <= nb < w - 1 and (na, nb) not in seen \
                            and int(I[na, nb]) == v:
                        seen.add((na, nb))
                        stack.append((na, nb))
            comps[v].append(sorted(cells))

    def slide(cells, dr, dc, steps, color):
        if steps <= 0:
            return
        op = 20 if dr < 0 else 21 if dr > 0 else 22 if dc > 0 else 23
        dest = [(r + dr * steps, c + dc * steps) for r, c in cells]
        if color == 0:
            # zero cells cannot be grabbed by ARCLE's object buffer: paint instead
            ops.append(0); sels.append(sel_of(dest))
            hole = sorted(set(cells) - set(dest))
            if hole:
                ops.append(bgc); sels.append(sel_of(hole))
            return
        ops.append(op); sels.append(sel_of(cells))        # grab the whole object
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))       # keep sliding same object
        hole = sorted(set(cells) - set(dest))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    # top-colour objects (horizontal runs) rise to row 1
    for cells in sorted(comps[c1], key=lambda cs: (cs[0][1], cs[0][0])):
        r = cells[0][0]
        slide(cells, -1, 0, r - 1, c1)
    # left-colour objects (vertical runs) slide to column 1
    for cells in sorted(comps[c2], key=lambda cs: (cs[0][0], cs[0][1])):
        c = cells[0][1]
        slide(cells, 0, -1, c - 1, c2)
    # bottom-colour objects sink to row h-2
    for cells in sorted(comps[c4], key=lambda cs: (cs[0][1], cs[0][0])):
        r = max(p[0] for p in cells)
        slide(cells, 1, 0, (h - 2) - r, c4)
    # right-colour objects slide to column w-2
    for cells in sorted(comps[c3], key=lambda cs: (cs[0][0], cs[0][1])):
        c = max(p[1] for p in cells)
        slide(cells, 0, 1, (w - 2) - c, c3)

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox = exactly the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
