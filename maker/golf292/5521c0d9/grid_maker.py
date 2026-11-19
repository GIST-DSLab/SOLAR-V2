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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (4, max(4, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_w))))
    inds = list(range(w))
    nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
    speps = random.sample(inds, nobjs * 2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
        speps = random.sample(inds, nobjs * 2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    remcols = [c for c in cols if c != bgc]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    forb = -1
    for sp, ep in zip(starts, ends):
        col = random.choice([c for c in ccols if c != forb])
        forb = col
        hdev = unifint(diff_lb, diff_ub, (0, h // 2))
        hei = random.choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        gi[h - hei:h, sp:ep + 1] = col
        lo = max(0, h - 2 * hei)
        go[lo:h - hei, sp:ep + 1] = col
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Each bar stands on one grid edge and jumps away from it by its own
    thickness. All bars are first cleared at once (one Color(bgc) on every
    foreground cell), then each bar is re-drawn at its destination with one
    CopyI of the bar (the part that still lands on the grid) + one Paste."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ho, wo = np.asarray(O).shape
    ops, sels = [], []

    # background: 3 of the 4 border lines are always fully background
    border = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
              + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    lines = [
        ("bottom", [(h - 1, c) for c in range(w)]),
        ("top", [(0, c) for c in range(w)]),
        ("left", [(r, 0) for r in range(h)]),
        ("right", [(r, w - 1) for r in range(h)]),
    ]
    side = None
    for name, cells in lines:
        if any(I[r, c] != bgc for r, c in cells):
            side = name
            break
    if side is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # canonical view V: bars stand on V's last row and jump upward
    if side == "bottom":
        V = I; back = lambda i, j: (i, j)
    elif side == "top":
        V = I[::-1, :]; back = lambda i, j: (h - 1 - i, j)
    elif side == "left":
        V = np.rot90(I, 1); back = lambda i, j: (j, w - 1 - i)
    else:
        V = np.rot90(I, 3); back = lambda i, j: (h - 1 - j, i)
    H, W = V.shape

    # bars = maximal runs of columns sharing (colour, top row)
    objs = []
    for c in range(W):
        if V[H - 1, c] == bgc:
            continue
        col = int(V[H - 1, c]); r = H - 1
        while r - 1 >= 0 and V[r - 1, c] == col:
            r -= 1
        if objs and objs[-1][0] == col and objs[-1][1] == r and objs[-1][3] == c - 1:
            objs[-1][3] = c
        else:
            objs.append([col, r, c, c])

    def rect(i0, i1, j0, j1):
        pts = [back(i0, j0), back(i1, j1)]
        r0 = min(p[0] for p in pts); r1 = max(p[0] for p in pts)
        c0 = min(p[1] for p in pts); c1 = max(p[1] for p in pts)
        return [r0, c0, r1 - r0, c1 - c0]

    # 1. clear every bar from where it stands (all foreground cells of I)
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ops.append(bgc); sels.append(sel_of(fg))

    # 2. redraw each bar one thickness further from its edge
    for col, r, c0, c1 in objs:
        ht = H - r
        s0 = max(r, ht)                      # bar rows whose image stays on the grid
        dst = rect(s0 - ht, H - 1 - ht, c0, c1)
        if col == 0:
            # 0 is transparent to Copy/Paste: draw the bar's full rectangle directly
            ops.append(0); sels.append(dst)
            continue
        ops.append(28); sels.append(rect(s0, H - 1, c0, c1))   # CopyI: full bar rectangle
        ops.append(30); sels.append([dst[0], dst[1], 0, 0])    # Paste at destination

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
