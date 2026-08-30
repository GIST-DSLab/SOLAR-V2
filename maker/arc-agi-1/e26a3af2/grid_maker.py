"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _majority(vals):
    return Counter(vals).most_common(1)[0][0]


def _stripes(grid):
    """Stripe axis + band list, read from the grid itself (never from O).

    The task rule: the picture is built of solid stripes that got sprinkled with
    noise.  A stripe line (row for horizontal stripes, column for vertical ones)
    is dominated by its stripe colour, so the majority colour per line recovers
    the stripe pattern; the axis with MORE distinct line-majorities is the axis
    the stripes stack along -- exactly the criterion the task's rule uses.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    rmaj = [_majority(g[r].tolist()) for r in range(H)]
    cmaj = [_majority(g[:, c].tolist()) for c in range(W)]
    vertical = len(set(cmaj)) > len(set(rmaj))
    maj = cmaj if vertical else rmaj
    bands, s = [], 0
    for i in range(1, len(maj) + 1):
        if i == len(maj) or maj[i] != maj[s]:
            bands.append((maj[s], s, i - 1))
            s = i
    return vertical, bands


def _clean(grid):
    g = np.asarray(grid, dtype=int).copy()
    vertical, bands = _stripes(g)
    for col, a, b in bands:
        if vertical:
            g[:, a:b + 1] = col
        else:
            g[a:b + 1, :] = col
    return g


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
VARIANTS = [{"vertical": False}, {"vertical": True}]


def sample_colors(num_examples=None) -> dict:
    # stripe colours: a fixed episode-wide palette (colour roles stay stable).
    # 0 is excluded so the whole grid is non-zero -- ARCLE's geometric ops and
    # canvas resizes are transparent to 0, which would destroy 0-coloured stripes.
    palette = [c for c in range(1, 10)]
    random.shuffle(palette)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, palette=None, vertical=None) -> dict:
    if palette is None:
        palette = [c for c in range(1, 10)]
        random.shuffle(palette)
    if vertical is None:
        vertical = random.choice([True, False])

    max_h = max(4, min(int(max_h), 30))
    max_w = max(4, min(int(max_w), 30))

    # L = axis the stripes stack along, A = length along a stripe
    if vertical:
        L_max, A_max = max_w, max_h
    else:
        L_max, A_max = max_h, max_w

    for _attempt in range(60):
        w = _unifint(diff_lb, diff_ub, (4, max(4, A_max)))
        nr_max = max(1, min(9, L_max // 2))
        nr = _unifint(diff_lb, diff_ub, (1, nr_max))
        scols = palette[:nr]

        heights = [2] * nr
        extra_max = max(0, min(30 - nr, L_max - 2 * nr))
        numexp = _unifint(diff_lb, diff_ub, (0, extra_max))
        for _ in range(numexp):
            heights[random.randrange(nr)] += 1

        cap = (w - 1) // 2          # noise per line stays a strict minority
        rows_in, rows_out = [], []
        total_noise = 0
        for idx, col in enumerate(scols):
            a = heights[idx]
            sg = [[col] * w for _ in range(a)]
            ub = max(0, (a * w) // 2 - 1)
            nnoise = _unifint(diff_lb, diff_ub, (0, ub))
            cells = [(i, j) for i in range(a) for j in range(w)]
            random.shuffle(cells)
            per_row = [0] * a
            placed = 0
            for (i, j) in cells:
                if placed >= nnoise:
                    break
                if per_row[i] >= cap:
                    continue
                sg[i][j] = random.choice([c for c in range(1, 10) if c != col])
                per_row[i] += 1
                placed += 1
            total_noise += placed
            rows_in.extend(sg)
            rows_out.extend([[col] * w for _ in range(a)])

        if total_noise == 0:                       # guarantee something to clean
            col = scols[0]
            rows_in[0][0] = random.choice([c for c in range(1, 10) if c != col])

        gi = np.array(rows_in, dtype=int)
        go = np.array(rows_out, dtype=int)
        if vertical:
            gi = gi.T.copy()
            go = go.T.copy()

        if gi.shape[0] > max_h or gi.shape[1] > max_w:
            continue
        # the stripe pattern must be unambiguously readable from the input alone
        if not np.array_equal(_clean(gi), go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}

    # very defensive fallback: simple two-stripe instance
    w = min(max(4, max_w), 30) if not vertical else min(max(4, max_h), 30)
    sg = [[palette[0]] * w, [palette[0]] * w, [palette[1]] * w, [palette[1]] * w]
    gi = np.array(sg, dtype=int)
    go = gi.copy()
    gi[0][0] = palette[2]
    if vertical:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    vertical, bands = _stripes(I)

    if vertical:
        dirty = [b for b in bands
                 if not np.all(I[:, b[1]:b[2] + 1] == b[0])]
    else:
        dirty = [b for b in bands
                 if not np.all(I[b[1]:b[2] + 1, :] == b[0])]

    has_zero = 0 in set(I.flatten().tolist())

    if vertical and dirty and not has_zero:
        # The stripes run vertically here.  Give the picture a quarter turn so
        # the stripes lie across the grid, wipe each stripe clean, then turn it
        # back.  Selections below are whole rectangles (padded square canvas,
        # full-width stripe bands), so bbox form is exactly the intended cells.
        sq = max(H, W)
        square = (H == W)

        if not square:
            ops.append(33); sels.append([0, 0, sq - 1, sq - 1])        # pad to square
        ops.append(25); sels.append([0, 0, sq - 1, sq - 1])            # quarter turn CW
        if not square:
            ops.append(33); sels.append([0, sq - H, W - 1, H - 1])     # crop the turned picture

        # working grid is now W x H, stripe k occupies rows c0..c1
        G = np.rot90(I, k=3).copy()
        for col, c0, c1 in bands:
            if np.all(G[c0:c1 + 1, :] == col):
                continue                                               # stripe already clean
            ops.append(int(col)); sels.append([c0, 0, c1 - c0, H - 1])
            G[c0:c1 + 1, :] = col

        if not square:
            ops.append(33); sels.append([0, 0, sq - 1, sq - 1])        # pad to square again
        ops.append(24); sels.append([0, 0, sq - 1, sq - 1])            # quarter turn back (CCW)
        if not square:
            ops.append(33); sels.append([sq - H, 0, H - 1, W - 1])     # crop back to H x W

    elif vertical:
        for col, c0, c1 in dirty:                                      # full-column stripe rects
            ops.append(int(col)); sels.append([0, c0, H - 1, c1 - c0])

    else:
        for col, r0, r1 in dirty:                                      # full-row stripe rects
            ops.append(int(col)); sels.append([r0, 0, r1 - r0, W - 1])

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
