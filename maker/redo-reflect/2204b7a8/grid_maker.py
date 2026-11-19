"""
ARC Task: 2204b7a8 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    ist = int(a + (b - a) * diff_lb)
    ien = int(a + (b - a) * diff_ub)
    if ist > ien:
        ist, ien = ien, ist
    return random.randint(ist, ien)


# The one discrete structural variant of this task: the generator ends with a
# coin flip that dmirrors (transposes) the pair, so the two border LINES are
# either the first/last COLUMN or the first/last ROW.  Both must be shown.
VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)                                   # canvas
    rem = [c for c in cols if c != bgc]
    ccol = random.choice(rem)                                   # the marks
    rem2 = [c for c in rem if c != ccol]
    c1 = random.choice(rem2)                                    # near line
    c2 = random.choice([c for c in rem2 if c != c1])            # far line

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]                       # test seen before
    return {"bgc": bgc, "ccol": ccol, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol, c1, c2, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])
    # build in the "vertical lines" frame (h x w); transpose at the end if asked
    hb = max(4, min(30, max_w if transposed else max_h))
    wb = max(4, min(30, max_h if transposed else max_w))
    while True:
        h = _unifint(diff_lb, diff_ub, (4, hb))
        w = _unifint(diff_lb, diff_ub, (4, wb))
        nc_ub = (h * (w - 2)) // 2 - 1
        if nc_ub < 1:
            continue
        nc = _unifint(diff_lb, diff_ub, (1, nc_ub))
        inds = [(i, j) for i in range(h) for j in range(1, w - 1)]
        locs = random.sample(inds, min(nc, len(inds)))
        if w % 2 == 1:                       # odd width -> middle column stays clean
            locs = [ij for ij in locs if ij[1] != w // 2]
        if not locs:
            continue
        gi = [[bgc] * w for _ in range(h)]
        for i in range(h):
            gi[i][0] = c1
            gi[i][w - 1] = c2
        for (i, j) in locs:
            gi[i][j] = ccol
        go = [row[:] for row in gi]
        for (i, j) in locs:
            go[i][j] = c1 if j < w // 2 else c2
        break
    if transposed:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule: every mark takes the colour of the border line on ITS OWN side.
    The two sides are mirror images of one another in role, so the trajectory
    performs that reflection: paint the half touching the c1 line, MIRROR the
    grid so the c2 line becomes the near line (its half swings into the near
    position), then paint the marks that are now near, and mirror back.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # orientation: a uniform first row means the lines are rows, not columns
    row_lines = len(set(I[0].tolist())) == 1

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    c1 = int(I[0, 0])           # near line colour
    c2 = int(I[h - 1, w - 1])   # far line colour
    ccol = [c for c in sorted(set(I.flatten().tolist()))
            if c not in (bgc, c1, c2)][0]

    marks = [(r, c) for r in range(h) for c in range(w) if I[r, c] == ccol]

    if row_lines:
        near = [(r, c) for (r, c) in marks if r < h // 2]
        far = [(r, c) for (r, c) in marks if r >= h - h // 2]
        flip_op = 27                                   # FlipV: up <-> down
        mirrored_far = [(h - 1 - r, c) for (r, c) in far]
    else:
        near = [(r, c) for (r, c) in marks if c < w // 2]
        far = [(r, c) for (r, c) in marks if c >= w - w // 2]
        flip_op = 26                                   # FlipH: left <-> right
        mirrored_far = [(r, w - 1 - c) for (r, c) in far]

    # whole-grid rectangle: the flip really does act on every cell, background
    # and both border lines included, so the bbox IS the intended cell set.
    full = [0, 0, h - 1, w - 1]

    ops, sels = [], []

    # 1. the marks lying against the c1 line take that line's colour
    if near:
        ops.append(c1)
        sels.append(sel_of(near))

    if far:
        # 2. reflect the grid: the c2 line and its marks swing into the near side
        ops.append(flip_op)
        sels.append(full)
        # 3. the marks now lying against the leading line (every ccol cell that
        #    is still left on the grid) take that line's colour, c2
        ops.append(c2)
        sels.append(sel_of(mirrored_far))
        # 4. reflect back to the original reading frame
        ops.append(flip_op)
        sels.append(full)

    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
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
                                f"for task 2204b7a8"
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
                    f"Failed to build a complete episode for task 2204b7a8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2204b7a8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
