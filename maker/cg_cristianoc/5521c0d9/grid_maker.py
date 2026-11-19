"""
ARC Task: 5521c0d9 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (4, max(4, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_w))))
    if h != w and (h > max_w or w > max_h):
        # rotations swap h and w; keep both orientations within bounds
        h = min(h, max_w)
        w = min(w, max_h)
        h = max(h, 4)
        w = max(w, 6)
    inds = interval(0, w, 1)
    nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
    speps = sample(inds, nobjs * 2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        speps = sample(inds, nobjs * 2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    forb = -1
    for sp, ep in zip(starts, ends):
        col = choice(remove(forb, ccols))
        forb = col
        hdev = unifint(diff_lb, diff_ub, (0, h // 2))
        hei = choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        ulc = (h - hei, sp)
        lrc = (h - 1, ep)
        obj = backdrop(frozenset({ulc, lrc}))
        gi = fill(gi, col, obj)
        go = fill(go, col, shift(obj, (-hei, 0)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading: every same-coloured connected component is a bar standing on
    one grid edge; each bar slides away from that edge by its own thickness
    (its extent perpendicular to the edge), clipping at the far wall."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: three of the four border lines are pure background
    border = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
              + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    # unit = 4-connected single-colour component of non-background cells
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen \
                            and I[na, nb] == col:
                        seen.add((na, nb))
                        stack.append((na, nb))
            comps.append((col, cells))

    for col, cells in comps:
        rows = [a for a, _ in cells]
        cs = [b for _, b in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cs), max(cs)
        # the edge this bar stands on decides the travel direction
        if r1 == h - 1 and r0 > 0:
            dr, dc, steps, mop = -1, 0, r1 - r0 + 1, 20      # MoveU
        elif r0 == 0 and r1 < h - 1:
            dr, dc, steps, mop = 1, 0, r1 - r0 + 1, 21       # MoveD
        elif c0 == 0 and c1 < w - 1:
            dr, dc, steps, mop = 0, 1, c1 - c0 + 1, 22       # MoveR
        elif c1 == w - 1 and c0 > 0:
            dr, dc, steps, mop = 0, -1, c1 - c0 + 1, 23      # MoveL
        else:
            continue

        def on(p):
            return 0 <= p[0] < h and 0 <= p[1] < w

        dst = [(a + dr * steps, b + dc * steps) for a, b in cells]
        dst = [p for p in dst if on(p)]

        if col == 0:
            # colour 0 is transparent to ARCLE object ops: cannot be carried
            ops.append(bgc); sels.append(sel_of(cells))
            if dst:
                ops.append(0); sels.append(sel_of(dst))
            continue

        # how far the bar can travel before its leading face meets the far wall
        if dr == -1:
            gap = r0
        elif dr == 1:
            gap = h - 1 - r1
        elif dc == 1:
            gap = w - 1 - c1
        else:
            gap = c0
        # bgc == 0: keep pushing past the wall, ARCLE clips what leaves the grid
        #           (each step visibly shortens the bar, the hole is already bgc).
        # bgc != 0: slide until it meets the wall; the footprint repaint below then
        #           removes whatever the full shift would have pushed off-grid.
        n = steps if bgc == 0 else min(steps, gap)

        cur = list(cells)
        for k in range(n):
            ops.append(mop)
            sels.append(sel_of(cur) if k == 0 else sel_of([]))   # grab once
            cur = [(a + dr, b + dc) for a, b in cur]
            cur = [p for p in cur if on(p)]

        # the bar's original footprint is background afterwards
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
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
                                f"for task 5521c0d9"
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
                    f"Failed to build a complete episode for task 5521c0d9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5521c0d9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
