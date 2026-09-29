"""
ARC Task: d43fd935 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccol = random.choice(remcols)
    remcols = [c for c in remcols if c != ccol]
    ndcols = random.randint(1, 8)
    dcols = random.sample(remcols, ndcols)
    return {"bgc": bgc, "ccol": ccol, "dcols": dcols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccol=None, dcols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccol is None:
        ccol = random.choice([c for c in cols if c != bgc])
    if dcols is None:
        rem = [c for c in cols if c not in (bgc, ccol)]
        dcols = random.sample(rem, random.randint(1, 8))
    dcols = list(dcols)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)

    gi = [[bgc] * w for _ in range(h)]
    bd = set()
    for r in range(loci, loci + boxh):
        for c in range(locj, locj + boxw):
            gi[r][c] = ccol
            bd.add((r, c))
    reminds = [(r, c) for r in range(h) for c in range(w) if (r, c) not in bd]
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = random.sample(reminds, nnoise)

    def in_rows(r):
        return loci <= r <= loci + boxh - 1

    def in_cols(c):
        return locj <= c <= locj + boxw - 1

    truenoise = [ij for ij in noise if not in_rows(ij[0]) and not in_cols(ij[1])]
    rem = [ij for ij in noise if ij not in set(truenoise)]
    top = [ij for ij in rem if ij[0] < loci]
    bottom = [ij for ij in rem if ij[0] > loci + boxh - 1]
    left = [ij for ij in rem if ij[1] < locj]
    right = [ij for ij in rem if ij[1] > locj + boxw - 1]

    for (r, c) in truenoise:
        gi[r][c] = random.choice(dcols)
    go = [row[:] for row in gi]

    for jj in sorted(set(c for r, c in top)):
        col = random.choice(dcols)
        subs = [(r, c) for r, c in top if c == jj]
        for r, c in subs:
            gi[r][c] = col
        for r in range(min(r for r, _ in subs), loci):
            go[r][jj] = col
    for jj in sorted(set(c for r, c in bottom)):
        col = random.choice(dcols)
        subs = [(r, c) for r, c in bottom if c == jj]
        for r, c in subs:
            gi[r][c] = col
        for r in range(loci + boxh, max(r for r, _ in subs) + 1):
            go[r][jj] = col
    for ii in sorted(set(r for r, c in left)):
        col = random.choice(dcols)
        subs = [(r, c) for r, c in left if r == ii]
        for r, c in subs:
            gi[r][c] = col
        for c in range(min(c for _, c in subs), locj):
            go[ii][c] = col
    for ii in sorted(set(r for r, c in right)):
        col = random.choice(dcols)
        subs = [(r, c) for r, c in right if r == ii]
        for r, c in subs:
            gi[r][c] = col
        for c in range(locj + boxw, max(c for _, c in subs) + 1):
            go[ii][c] = col

    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops = []
    sels = []

    bgc = int(np.bincount(I.flatten()).argmax())

    visited = np.zeros_like(I, dtype=bool)
    box_r, box_c, box_h, box_w = -1, -1, -1, -1
    max_size = 0
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or int(I[r, c]) == bgc:
                continue
            color = int(I[r, c])
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= hi or x < 0 or x >= wi:
                    continue
                if visited[y, x] or int(I[y, x]) != color:
                    continue
                visited[y, x] = True
                cells.append((y, x))
                stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
            if cells:
                rs = [y for y, x in cells]
                cs = [x for y, x in cells]
                min_r, max_r_ = min(rs), max(rs)
                min_c, max_c_ = min(cs), max(cs)
                bh = max_r_ - min_r + 1
                bw = max_c_ - min_c + 1
                if bh * bw == len(cells) and bh >= 2 and bw >= 2 and len(cells) > max_size:
                    max_size = len(cells)
                    box_r, box_c, box_h, box_w = min_r, min_c, bh, bw

    def emit_line(color, cells):
        # A line whose cells all already hold the colour would do nothing — skip it.
        if all(int(I[r, c]) == color for r, c in cells):
            return
        ops.append(color)
        sels.append(sel_of(cells))

    if box_r >= 0:
        for j in range(box_c, box_c + box_w):
            top_rows = [r for r in range(box_r) if int(I[r, j]) != bgc]
            if top_rows:
                topmost = min(top_rows)
                color = int(I[topmost, j])
                emit_line(color, [(r, j) for r in range(topmost, box_r)])

        for j in range(box_c, box_c + box_w):
            bot_rows = [r for r in range(box_r + box_h, hi) if int(I[r, j]) != bgc]
            if bot_rows:
                botmost = max(bot_rows)
                color = int(I[bot_rows[0], j])
                emit_line(color, [(r, j) for r in range(box_r + box_h, botmost + 1)])

        for i in range(box_r, box_r + box_h):
            left_cols = [c for c in range(box_c) if int(I[i, c]) != bgc]
            if left_cols:
                leftmost = min(left_cols)
                color = int(I[i, leftmost])
                emit_line(color, [(i, c) for c in range(leftmost, box_c)])

        for i in range(box_r, box_r + box_h):
            right_cols = [c for c in range(box_c + box_w, wi) if int(I[i, c]) != bgc]
            if right_cols:
                rightmost = max(right_cols)
                color = int(I[i, right_cols[0]])
                emit_line(color, [(i, c) for c in range(box_c + box_w, rightmost + 1)])

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
                        f"num_examples+1 ({num_examples + 1}) for task d43fd935"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d43fd935"
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
                                f"for task d43fd935"
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
                    f"Failed to build a complete episode for task d43fd935 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d43fd935-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
