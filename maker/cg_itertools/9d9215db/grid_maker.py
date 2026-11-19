"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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

ROTS = ["identity", "rot90", "rot180", "rot270"]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _rot90(g):      # clockwise
    return [list(x) for x in zip(*g[::-1])]


def _rot180(g):
    return [row[::-1] for row in g[::-1]]


def _rot270(g):     # counter-clockwise
    return [list(x) for x in zip(*g)][::-1]


def _apply_rot(g, rot):
    if rot == "rot90":
        return _rot90(g)
    if rot == "rot180":
        return _rot180(g)
    if rot == "rot270":
        return _rot270(g)
    return [row[:] for row in g]


def sample_colors(num_examples=None) -> dict:
    # Only the background is fixed per episode; mark colours are free because
    # the rule (mirror marks, join mirrored pairs) never depends on which colour.
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"rot": r} for r in ROTS]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    if rot in ("rot90", "rot270"):
        hlim, wlim = max_w, max_h
    else:
        hlim, wlim = max_h, max_w

    hhi = max(5, min(14, (hlim - 1) // 2))
    whi = max(5, min(14, (wlim - 1) // 2))
    hh = _unifint(diff_lb, diff_ub, (5, hhi))
    ww = _unifint(diff_lb, diff_ub, (5, whi))
    h = hh * 2 + 1
    w = ww * 2 + 1

    cols = [c for c in range(10) if c != bgc]
    ub = min(h, w) // 4

    nrings = _unifint(diff_lb, diff_ub, (1, ub))
    ring_ks = random.sample(range(ub), nrings)
    ring_col = {k: random.choice(cols) for k in ring_ks}

    nlines = _unifint(diff_lb, diff_ub, (1, ub - 1))
    line_ks = random.sample(range(ub - 1), nlines)
    line_col = {k: random.choice(cols) for k in line_ks}

    gi = [[bgc] * w for _ in range(h)]
    for k, col in ring_col.items():
        i = 2 * k + 1
        gi[i][i] = col
    for k, col in line_col.items():
        i = 2 * k + 1
        gi[i][i + 2] = col
        gi[i + 2][i] = col

    go = [row[:] for row in gi]
    for k, col in line_col.items():
        i = 2 * k + 1
        for r in (i, h - 1 - i):
            for c in range(i + 2, w - 2 - i, 2):
                go[r][c] = col
        for c in (i, w - 1 - i):
            for r in range(i + 2, h - 2 - i, 2):
                go[r][c] = col
    for k, col in ring_col.items():
        i = 2 * k + 1
        for r in (i, h - 1 - i):
            for c in (i, w - 1 - i):
                go[r][c] = col

    gi = _apply_rot(gi, rot)
    go = _apply_rot(go, rot)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): the picture is 4-fold mirror
    symmetric, and every ray is a FULL line that runs from a mark to that
    mark's own mirror image.  So: first replicate the seed corner's marks into
    the other three corners (mirror about the centre column, then the centre
    row); then, with both endpoints of every line on the board, join each
    mirrored pair of marks with one odd-spaced dotted line -- top edge, bottom
    edge, left edge, right edge of that ring.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ch, cw = (H - 1) // 2, (W - 1) // 2
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    g = I.copy()

    fg = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]
    top = (max(r for r, _ in fg) < ch) if fg else True
    left = (max(c for _, c in fg) < cw) if fg else True

    def mirror_block(r0, c0, h, w, r1, c1, flip_op):
        # copy the h x w block at (r0,c0) (whole rectangle, background included)
        src = g[r0:r0 + h, c0:c0 + w].copy()
        dst = g[r1:r1 + h, c1:c1 + w]
        if np.any((src != 0) & (src != dst)):       # the Paste visibly lands
            ops.append(29); sels.append([r0, c0, h - 1, w - 1])      # CopyO
            ops.append(30); sels.append([r1, c1, 0, 0])              # Paste
        repair = []
        for i in range(h):
            for j in range(w):
                if src[i, j] != 0:
                    dst[i, j] = src[i, j]
                elif dst[i, j] != 0:     # 0-coloured cells never travel in a Paste
                    repair.append((r1 + i, c1 + j))
        if repair:
            ops.append(0); sels.append(sel_of(repair))               # Color0
            for r, c in repair:
                g[r, c] = 0
        # whole pasted rectangle mirrored in place (skipped if already symmetric)
        blk = g[r1:r1 + h, c1:c1 + w]
        fl = np.fliplr(blk) if flip_op == 26 else np.flipud(blk)
        if not np.array_equal(fl, blk):
            ops.append(flip_op); sels.append([r1, c1, h - 1, w - 1])
            g[r1:r1 + h, c1:c1 + w] = fl

    # ---- 1. replicate the seed corner's marks into all four corners ----------
    rs = 0 if top else ch + 1
    cs = 0 if left else cw + 1
    cd = cw + 1 if left else 0
    if fg:
        mirror_block(rs, cs, ch, cw, rs, cd, 26)        # across the centre column
        rd = ch + 1 if top else 0
        mirror_block(rs, 0, ch, W, rd, 0, 27)           # across the centre row

    # ---- 2. join every mirrored mark pair with its full dotted line ----------
    ub = min(H, W) // 4
    for k in range(max(0, ub - 1)):
        a, b = 2 * k + 1, 2 * k + 3
        col = int(g[a, b])                     # (a, b) and (b, a) carry the pair
        if col == bgc:
            continue
        lines = [
            [(a, c) for c in range(b + 2, W - 2 - b, 2)],          # top edge
            [(H - 1 - a, c) for c in range(b + 2, W - 2 - b, 2)],  # bottom edge
            [(r, a) for r in range(b + 2, H - 2 - b, 2)],          # left edge
            [(r, W - 1 - a) for r in range(b + 2, H - 2 - b, 2)],  # right edge
        ]
        for cells in lines:
            if cells:
                ops.append(col); sels.append(sel_of(cells))
                for r, c in cells:
                    g[r, c] = col

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
