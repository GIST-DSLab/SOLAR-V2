"""
ARC Task: 90c28cc7 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Picture colours are drawn per instance from 1..9 and the rule ignores colour;
    # the frame/background is the hardcoded 0. Nothing to fix per episode.
    return {}


def _dedupe(rows):
    out = []
    for r in rows:
        if r not in out:
            out.append(r)
    return out


def _dmirror(g):
    return [list(col) for col in zip(*g)]


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kwargs) -> dict:
    def unifint(lb, ub):
        lo = lb + round((ub - lb) * diff_lb)
        hi = lb + round((ub - lb) * diff_ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    cols = list(range(1, 10))
    h = unifint(2, min(10, max_h))
    w = unifint(2, min(10, max_w))
    nc = unifint(2, 9)
    colss = random.sample(cols, nc)
    gi = [[random.choice(colss) for _ in range(w)] for _ in range(h)]
    gi = _dmirror(_dedupe(_dmirror(_dedupe(gi))))
    go = [list(r) for r in gi]
    h, w = len(gi), len(gi[0])
    fullh = unifint(h, max(h, max_h))
    fullw = unifint(w, max(w, max_w))
    inh = unifint(h, fullh)
    inw = unifint(w, fullw)
    while h < inh or w < inw:
        if h < inh:
            idx = random.randint(0, h - 1)
            gi = gi[:idx + 1] + gi[idx:]
        else:
            idx = random.randint(0, w - 1)
            gi = [r[:idx + 1] + r[idx:] for r in gi]
        h, w = len(gi), len(gi[0])
    while h < fullh or w < fullw:
        if h < fullh:
            z = [[0] * w]
            gi = z + gi if random.random() < 0.5 else gi + z
        else:
            gi = [[0] + r for r in gi] if random.random() < 0.5 else [r + [0] for r in gi]
        h, w = len(gi), len(gi[0])
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O):
    """
    Rule (read off I alone): a picture with no 0 cells sits in a 0 frame; every distinct
    row of the picture was stretched into a run of identical rows, likewise columns.

    Unit of work: ONE RUN OF REPEATED LINES. Each run of repeated rows is removed by
    grabbing the whole picture block below it and sliding it up over the repeats
    (one grab, then unit MoveU steps with an empty selection). Then each run of repeated
    columns is removed by sliding the block to its right left. The vacated tail becomes
    frame (0); a final crop keeps the compacted picture. O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    nz = np.argwhere(I != 0)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    G = I.copy()  # python-side tracking of the working grid as ops apply

    def rect(ra, rb, ca, cb):
        return [(r, c) for r in range(ra, rb + 1) for c in range(ca, cb + 1)]

    # ---- rows: picture occupies rows r0..bot, cols c0..c1 ----
    bot = r1
    g = r0 + 1
    while g <= bot:
        if not np.array_equal(G[g, c0:c1 + 1], G[g - 1, c0:c1 + 1]):
            g += 1
            continue
        e = g
        while e + 1 <= bot and np.array_equal(G[e + 1, c0:c1 + 1], G[g - 1, c0:c1 + 1]):
            e += 1
        L = e - g + 1                       # number of repeated rows to delete
        if e < bot:
            # slide the picture block below the run (rows e+1..bot) up by L
            ops.append(20); sels.append(sel_of(rect(e + 1, bot, c0, c1)))   # grab
            for _ in range(L - 1):
                ops.append(20); sels.append(sel_of([]))                     # continue
            blk = G[e + 1:bot + 1, c0:c1 + 1].copy()
            G[g:bot + 1, c0:c1 + 1] = 0
            G[g:g + blk.shape[0], c0:c1 + 1] = blk
        # (a run at the very bottom has nothing below it; the final crop drops it)
        bot -= L
        g += 1

    # ---- columns: picture occupies rows r0..bot, cols c0..right ----
    right = c1
    g = c0 + 1
    while g <= right:
        if not np.array_equal(G[r0:bot + 1, g], G[r0:bot + 1, g - 1]):
            g += 1
            continue
        e = g
        while e + 1 <= right and np.array_equal(G[r0:bot + 1, e + 1], G[r0:bot + 1, g - 1]):
            e += 1
        L = e - g + 1
        if e < right:
            # slide the picture block right of the run (cols e+1..right) left by L
            ops.append(23); sels.append(sel_of(rect(r0, bot, e + 1, right)))  # grab
            for _ in range(L - 1):
                ops.append(23); sels.append(sel_of([]))                       # continue
            blk = G[r0:bot + 1, e + 1:right + 1].copy()
            G[r0:bot + 1, g:right + 1] = 0
            G[r0:bot + 1, g:g + blk.shape[1]] = blk
        right -= L
        g += 1

    ho, wo = bot - r0 + 1, right - c0 + 1
    # crop to the compacted picture: a full rectangle whose cells are all nonzero
    if (r0, c0, ho, wo) != (0, 0, hi, wi):
        ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 90c28cc7"
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
                                f"for task 90c28cc7"
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
                    f"Failed to build a complete episode for task 90c28cc7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"90c28cc7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
