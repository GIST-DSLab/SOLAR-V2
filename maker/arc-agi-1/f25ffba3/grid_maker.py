"""
ARC Task: f25ffba3 (RE-ARC) — LLM-generated grid_maker
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

# Discrete structural variants of this task:
#   transpose=False -> grid is split LEFT/RIGHT  (one vertical half is blank) -> mirror is left<->right
#   transpose=True  -> grid is split TOP/BOTTOM  (one horizontal half is blank) -> mirror is up<->down
#   side            -> which half holds the pattern
VARIANTS = [
    {"transpose": False, "side": 0},
    {"transpose": False, "side": 1},
    {"transpose": True,  "side": 0},
    {"transpose": True,  "side": 1},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc]
    random.shuffle(palette)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test variant was shown in examples
    return {"bgc": bgc, "palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, palette=None,
             transpose=None, side=None) -> dict:
    def _unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            hi = lo
        if lo < a:
            lo = a
        if hi > b:
            hi = b
        return random.randint(lo, hi)

    if bgc is None:
        bgc = random.choice(range(1, 10))
    if palette is None:
        palette = [c for c in range(1, 10) if c != bgc]
    if transpose is None:
        transpose = random.choice([True, False])
    if side is None:
        side = random.choice([0, 1])

    # pattern half is (hh, ww); full grid is (hh, 2*ww), transposed to (2*ww, hh)
    if transpose:
        lim_rows, lim_cols = max_w, max_h
    else:
        lim_rows, lim_cols = max_h, max_w

    kmax = max(1, min(14, (lim_rows - 1) // 2))
    hh = 2 * _unifint(diff_lb, diff_ub, (1, kmax)) + 1
    wmax = max(3, min(15, lim_cols // 2))
    ww = _unifint(diff_lb, diff_ub, (3, wmax))

    numcols = _unifint(diff_lb, diff_ub, (1, min(8, len(palette))))
    fgcols = random.sample(palette, numcols)

    # ---- grow a connected (8-neighbour) blob of coloured cells ----
    all_c = [(r, c) for r in range(hh) for c in range(ww)]

    def nbrs(p):
        r, c = p
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hh and 0 <= cc < ww:
                        out.append((rr, cc))
        return out

    start = random.choice(all_c)
    avail = set(all_c)
    avail.discard(start)
    cells = {start: random.choice(fgcols)}
    frontier = set(q for q in nbrs(start) if q in avail)

    def grow():
        if not frontier:
            return False
        p = random.choice(sorted(frontier))
        frontier.discard(p)
        avail.discard(p)
        cells[p] = random.choice(fgcols)
        for q in nbrs(p):
            if q in avail:
                frontier.add(q)
        return True

    nc = _unifint(diff_lb, diff_ub, (2, max(2, hh * ww - 2)))
    for _ in range(nc - 1):
        if not grow():
            break
    # object must straddle the middle rows (so neither half of the untransposed
    # grid is monochrome -> the left/right mirror branch is the one that applies)
    guard = 0
    while True:
        rs = [r for (r, _) in cells]
        if min(rs) <= hh // 2 - 1 and max(rs) >= hh // 2 + 1:
            break
        if not grow() or guard > hh * ww:
            break
        guard += 1

    # ---- left-pack every row (stable order: non-background cells first) ----
    gix = np.full((hh, ww), bgc, dtype=int)
    for r in range(hh):
        vals = [cells[(r, c)] for c in range(ww) if (r, c) in cells]
        for i, v in enumerate(vals):
            gix[r, i] = v

    canv = np.full((hh, ww), bgc, dtype=int)
    gi = np.concatenate([gix, canv], axis=1)
    go = np.concatenate([gix, np.fliplr(gix)], axis=1)

    if side == 1:
        gi, go = np.fliplr(gi), np.fliplr(go)
    if random.random() < 0.5:
        gi, go = np.flipud(gi), np.flipud(go)
    if transpose:
        gi, go = gi.T.copy(), go.T.copy()

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bg = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # Which mirror axis?  If one horizontal half of the grid is a single colour the
    # grid is split top/bottom -> mirror up-down; otherwise it is split left/right.
    top = I[:h // 2]
    bot = I[h // 2 + h % 2:]
    row_split = (top.size > 0 and len(np.unique(top)) == 1) or \
                (bot.size > 0 and len(np.unique(bot)) == 1)

    if row_split:
        hh = h // 2                       # each half is hh rows tall
        top_nb = int(np.count_nonzero(I[:hh] != bg))
        bot_nb = int(np.count_nonzero(I[h - hh:] != bg))
        if top_nb <= bot_nb:              # the emptier half is the destination
            src_r, dst_r = h - hh, 0
        else:
            src_r, dst_r = 0, h - hh
        src = I[src_r:src_r + hh, :]
        if not np.all(src == bg):
            # full rectangle: the whole patterned half, background included
            ops.append(28); sels.append([src_r, 0, hh - 1, w - 1])   # CopyI source half
            ops.append(30); sels.append([dst_r, 0, 0, 0])            # Paste at blank half's top-left
            if not np.array_equal(src, np.flipud(src)):
                # full rectangle: mirror the whole pasted half up<->down in place
                ops.append(27); sels.append([dst_r, 0, hh - 1, w - 1])
    else:
        ww = w // 2                       # each half is ww columns wide
        left_nb = int(np.count_nonzero(I[:, :ww] != bg))
        right_nb = int(np.count_nonzero(I[:, w - ww:] != bg))
        if left_nb <= right_nb:
            src_c, dst_c = w - ww, 0
        else:
            src_c, dst_c = 0, w - ww
        src = I[:, src_c:src_c + ww]
        if not np.all(src == bg):
            # full rectangle: the whole patterned half, background included
            ops.append(28); sels.append([0, src_c, h - 1, ww - 1])   # CopyI source half
            ops.append(30); sels.append([0, dst_c, 0, 0])            # Paste at blank half's top-left
            if not np.array_equal(src, np.fliplr(src)):
                # full rectangle: mirror the whole pasted half left<->right in place
                ops.append(26); sels.append([0, dst_c, h - 1, ww - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task f25ffba3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25ffba3"
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
                                f"for task f25ffba3"
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
                    f"Failed to build a complete episode for task f25ffba3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25ffba3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
