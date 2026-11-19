"""
ARC Task: 496994bd (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [
    {"content_left": True,  "separator": False},
    {"content_left": False, "separator": False},
    {"content_left": True,  "separator": True},
    {"content_left": False, "separator": True},
]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(1, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        # make sure both content sides are shown
        base = [dict(VARIANTS[0]), dict(VARIANTS[1])] if n_ex >= 2 else []
        rest = [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(base))]
        examples = base + rest
        random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None,
             content_left=None, separator=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    if bgc is None:
        bgc = random.choice(range(1, 10))
    if content_left is None:
        content_left = random.choice([True, False])
    if separator is None:
        separator = random.choice([True, False])

    cols = [c for c in range(1, 10) if c != bgc]
    h = unifint(diff_lb, diff_ub, (3, min(30, max_h)))
    wcap = (max_w - (1 if separator else 0)) // 2
    w = unifint(diff_lb, diff_ub, (3, max(3, min(14, wcap))))
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(cols, numcols)

    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = {(i, j) for i in range(h) for j in range(w)}
    top = [ij for ij in bx if ij[0] < h // 2]
    bot = [ij for ij in bx if ij[0] > h // 2]
    obj = {}
    a = random.choice(top); obj[a] = random.choice(remcols)
    b = random.choice(bot); obj[b] = random.choice(remcols)
    bx -= {a, b}
    for _ in range(nc - 2):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = list(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx -= {ch}

    gix = np.full((h, w), bgc, dtype=int)
    for (i, j), c in obj.items():
        gix[i, j] = c
    # each row: non-bg cells first (stable), bg at the end
    for i in range(h):
        row = list(gix[i])
        nonbg = [v for v in row if v != bgc]
        gix[i] = nonbg + [bgc] * (w - len(nonbg))

    canv = np.full((h, w), bgc, dtype=int)
    sep = np.full((h, 1), bgc, dtype=int)
    mir = gix[:, ::-1]
    if separator:
        gi = np.hstack([gix, sep, canv])
        go = np.hstack([gix, sep, mir])
    else:
        gi = np.hstack([gix, canv])
        go = np.hstack([gix, mir])
    if not content_left:
        gi = gi[:, ::-1]
        go = go[:, ::-1]
    if random.choice([True, False]):
        gi = gi[::-1, :]
        go = go[::-1, :]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Rule: the grid is two equal column half-blocks (width w = wi // 2, with
    a single background separator column when wi is odd). One half holds the
    content, the other is blank. The content half is duplicated into the blank
    half and mirrored left<->right there.

    The code-golf reading (first half, then the same lines in reverse order)
    carves the task the same way: the half-block is the unit and reversing its
    line order is exactly a mirror of that block. So the operations agree:
    copy the content half from the input, paste it into the blank half, and
    flip it in place.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    w = wi // 2
    left_c, right_c = 0, wi - w
    left_blk = I[:, left_c:left_c + w]
    right_blk = I[:, right_c:right_c + w]

    # The blank half-block is a single uniform colour; that colour is bgc.
    left_uniform = len(np.unique(left_blk)) == 1
    right_uniform = len(np.unique(right_blk)) == 1
    if right_uniform and not left_uniform:
        src_c, dst_c = left_c, right_c
    elif left_uniform and not right_uniform:
        src_c, dst_c = right_c, left_c
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
        if np.any(left_blk != bgc):
            src_c, dst_c = left_c, right_c
        else:
            src_c, dst_c = right_c, left_c

    src_block = I[:, src_c:src_c + w]

    # Full half-block rectangles: these ops act on the whole block
    # (background included), so the rectangle is exactly the intended cells.
    src_rect = [(r, c) for r in range(hi) for c in range(src_c, src_c + w)]
    dst_rect = [(r, c) for r in range(hi) for c in range(dst_c, dst_c + w)]

    # 1. copy the content half-block from the input
    ops.append(28); sels.append(sel_of(src_rect))
    # 2. lay it over the blank half-block (paste origin = its top-left cell)
    ops.append(30); sels.append(sel_of([(0, dst_c)]))
    # 3. mirror the pasted block left<->right in place (skip if already symmetric)
    if not np.array_equal(src_block, src_block[:, ::-1]):
        ops.append(26); sels.append(sel_of(dst_rect))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 496994bd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 496994bd"
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
                                f"for task 496994bd"
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
                    f"Failed to build a complete episode for task 496994bd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"496994bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
