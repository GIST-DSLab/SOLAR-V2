"""
ARC Task: 11852cab (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    if ccols is None:
        numc = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, numc)
    ccols = [c for c in ccols if c != bgc] or [c for c in remcols][:1]
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            go = fill(go, ringcols[0], plcdrings[0])
            idx = randint(1, 3)
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            go = fill(go, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx + 1:]
            remringcols = ringcols[1:idx] + ringcols[idx + 1:]
            numrs = unifint(diff_lb, diff_ub, (1, 2))
            locs = sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            tofillgi = merge(frozenset(
                recolor(col, frozenset(sample(totuple(remring), 4 - unifint(diff_lb, diff_ub, (0, 3)))))
                for remring, col in zip(remrings, remringcols)
            ))
            tofillgo = merge(frozenset(
                recolor(col, remring) for remring, col in zip(remrings, remringcols)
            ))
            if min(shape(tofillgi)) == 5:
                succ += 1
                gi = paint(gi, tofillgi)
                go = paint(go, tofillgo)
    return {"input": gi, "output": go}


def _find_blocks(grid, bgc):
    h, w = len(grid), len(grid[0])

    def val(r, c):
        if 0 <= r < h and 0 <= c < w:
            return grid[r][c]
        return bgc

    blocks = []
    for r0 in range(0, h - 4):
        for c0 in range(0, w - 4):
            if grid[r0 + 2][c0 + 2] == bgc:
                continue
            ok = True
            # surrounding 7x7 frame must be background
            for i in range(-1, 6):
                for j in (-1, 5):
                    if val(r0 + i, c0 + j) != bgc or val(r0 + j, c0 + i) != bgc:
                        ok = False
            if not ok:
                continue
            fg = []
            for i in range(5):
                for j in range(5):
                    v = grid[r0 + i][c0 + j]
                    if v != bgc:
                        if (i + j) % 2 == 1:
                            ok = False
                        fg.append((i, j))
            if not ok or not fg:
                continue
            ri = [p[0] for p in fg]
            cj = [p[1] for p in fg]
            if min(ri) != 0 or max(ri) != 4 or min(cj) != 0 or max(cj) != 4:
                continue
            blocks.append((r0, c0))
    return blocks


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    if examples:
        cnt = {}
        for ex in examples:
            for v in np.asarray(ex[0], dtype=int).flatten().tolist():
                cnt[v] = cnt.get(v, 0) + 1
        for v in I.flatten().tolist():
            cnt[v] = cnt.get(v, 0) + 1
    else:
        cnt = {}
        for v in I.flatten().tolist():
            cnt[v] = cnt.get(v, 0) + 1
    bgc = max(sorted(cnt), key=lambda c: cnt[c])
    # bgc must be the most common colour of THIS input as well
    ci = {}
    for v in I.flatten().tolist():
        ci[v] = ci.get(v, 0) + 1
    if bgc not in ci or ci[bgc] != max(ci.values()):
        bgc = max(sorted(ci), key=lambda c: ci[c])

    grid = [list(map(int, row)) for row in I.tolist()]
    ops, sels = [], []

    # Reading: each 5x5 block is a pattern with 4-fold ROTATIONAL symmetry.
    # Rotate the block a quarter turn (CCW) and draw the marks it carried
    # just before the turn back on top; three such turns overlay all four
    # rotations, closing every partial ring.  Blocks are located from I only.
    for (r0, c0) in _find_blocks(grid, bgc):
        rect = [r0, c0, 4, 4]  # exactly the whole square 5x5 block, bg included
        for _ in range(3):
            sub = [row[c0:c0 + 5] for row in grid[r0:r0 + 5]]
            new = [[sub[j][4 - i] for j in range(5)] for i in range(5)]  # CCW
            if new == sub:
                break  # already rotation-symmetric: nothing left to complete
            ops.append(24)
            sels.append(rect)
            for i in range(5):
                for j in range(5):
                    grid[r0 + i][c0 + j] = new[i][j]
            bycol = {}
            for i in range(5):
                for j in range(5):
                    v = sub[i][j]
                    if v != bgc and grid[r0 + i][c0 + j] != v:
                        bycol.setdefault(v, []).append((r0 + i, c0 + j))
            for v in sorted(bycol):
                ops.append(int(v))
                sels.append(sel_of(bycol[v]))
                for (r, c) in bycol[v]:
                    grid[r][c] = v

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 11852cab"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 11852cab"
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
                                f"for task 11852cab"
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
                    f"Failed to build a complete episode for task 11852cab "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"11852cab-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
