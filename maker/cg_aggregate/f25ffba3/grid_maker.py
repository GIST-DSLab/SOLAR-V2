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
from maker.sel_helpers import sel_of

# The code-golf entry reads the task the same way the maker does: the empty half
# becomes a mirror image of the patterned half. So the unit is one whole half,
# and the steps are copy, place, then mirror. Nothing in the route needs to change.
# This version keeps the maker's copy -> paste -> in-place flip trajectory.

SIDES = [{"flip_lr": False}, {"flip_lr": True}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(1, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SIDES):
        examples = [dict(v) for v in SIDES]
        examples += [dict(random.choice(SIDES)) for _ in range(n_ex - len(SIDES))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(SIDES, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * max(0.0, min(1.0, diff_lb))))
    hi = a + int(round((b - a) * max(0.0, min(1.0, diff_ub))))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, flip_lr=None,
             flip_ud=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(1, 10))
    if flip_lr is None:
        flip_lr = random.choice((True, False))
    if flip_ud is None:
        flip_ud = random.choice((True, False))
    kmax = max(1, min(14, (max_h - 1) // 2))
    h = _unifint(diff_lb, diff_ub, (1, kmax)) * 2 + 1
    wmax = max(3, min(15, max_w // 2))
    w = _unifint(diff_lb, diff_ub, (3, wmax))
    remcols = [c for c in range(1, 10) if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    nc = _unifint(diff_lb, diff_ub, (2, h * w - 2))
    bx = {(r, c) for r in range(h) for c in range(w)}

    def nbrs(cells):
        out = set()
        for (r, c) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        out.add((r + dr, c + dc))
        return out

    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        cand = sorted(bx & nbrs(obj.keys()))
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx -= {ch}
    while min(r for r, _ in obj) > h // 2 - 1 or max(r for r, _ in obj) < h // 2 + 1:
        cand = sorted(bx & nbrs(obj.keys()))
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx -= {ch}
    gix = [[bgc] * w for _ in range(h)]
    for (r, c), v in obj.items():
        gix[r][c] = v
    # order each row: non-bg cells first (stable), as in the generator
    gix = [sorted(row, key=lambda v: v == bgc) for row in gix]
    gi = [row + [bgc] * w for row in gix]
    go = [row + row[::-1] for row in gix]
    if flip_lr:
        gi = [row[::-1] for row in gi]
        go = [row[::-1] for row in go]
    if flip_ud:
        gi = gi[::-1]
        go = go[::-1]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # bg fills the whole empty half and part of the patterned half, so it is the majority colour
    bg = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []

    top = I[:h // 2]
    bot = I[h // 2 + h % 2:]
    row_split = (top.size > 0 and len(np.unique(top)) == 1) or \
                (bot.size > 0 and len(np.unique(bot)) == 1)

    if row_split:
        hh = h // 2
        top_nb = int(np.count_nonzero(I[:hh] != bg))
        bot_nb = int(np.count_nonzero(I[h - hh:] != bg))
        if top_nb <= bot_nb:
            src_r, dst_r = h - hh, 0
        else:
            src_r, dst_r = 0, h - hh
        src = I[src_r:src_r + hh, :]
        if not np.all(src == bg):
            # full rectangle: the whole patterned half, background included
            ops.append(28); sels.append([src_r, 0, hh - 1, w - 1])
            ops.append(30); sels.append([dst_r, 0, 0, 0])
            if not np.array_equal(src, np.flipud(src)):
                # full rectangle: mirror the whole pasted half up<->down in place
                ops.append(27); sels.append([dst_r, 0, hh - 1, w - 1])
    else:
        ww = w // 2
        left_nb = int(np.count_nonzero(I[:, :ww] != bg))
        right_nb = int(np.count_nonzero(I[:, w - ww:] != bg))
        if left_nb <= right_nb:
            src_c, dst_c = w - ww, 0
        else:
            src_c, dst_c = 0, w - ww
        src = I[:, src_c:src_c + ww]
        if not np.all(src == bg):
            # full rectangle: the whole patterned half, background included (colours are 1-9, never 0)
            ops.append(28); sels.append([0, src_c, h - 1, ww - 1])
            ops.append(30); sels.append([0, dst_c, 0, 0])
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
