"""
ARC Task: 29623171 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, fgc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, fgc=None) -> dict:
    from random import choice, sample, randint

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        if hi < lo:
            hi = lo
        return randint(lo, hi)

    if bgc is None or linc is None or fgc is None:
        bgc, linc, fgc = random.sample(range(10), 3)
    for _ in range(200):
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (2, 6))
        nh = unifint(diff_lb, diff_ub, (2, 4))
        nw = unifint(diff_lb, diff_ub, (2, 4))
        if h * w < 4:
            continue
        fullh = h * nh + nh - 1
        fullw = w * nw + nw - 1
        if fullh <= max_h and fullw <= max_w:
            break
    else:
        h, w, nh, nw = 2, 2, 2, 2
        fullh, fullw = 5, 5
    gi = np.full((fullh, fullw), linc, dtype=int)
    go = gi.copy()
    inds = [(i, j) for i in range(h) for j in range(w)]
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))
    mostc = sample(inds, nmostc)

    def block(loc, marks, win):
        a, b = loc
        gi[a:a + h, b:b + w] = bgc
        for (i, j) in marks:
            gi[a + i, b + j] = fgc
        go[a:a + h, b:b + w] = fgc if win else bgc

    block(srcloc, mostc, True)
    for rl in llocs:
        if rl == srcloc:
            continue
        noth = unifint(diff_lb, diff_ub, (0, nmostc))
        block(rl, sample(inds, noth), noth == nmostc)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Find the grid of blocks from I alone. Try each block size b where
    # (n+1) is divisible by (b+1). The separator rows and columns must all be
    # one colour L, and L must not appear inside any block. This holds even
    # when a whole row inside a block row happens to be uniform background.
    def splits(n):
        return [b for b in range(1, n + 1)
                if (n + 1) % (b + 1) == 0 and (n + 1) // (b + 1) >= 2]

    cfg = None
    for bh in splits(hi):
        for bw in splits(wi):
            lr = list(range(bh, hi, bh + 1))
            lc = list(range(bw, wi, bw + 1))
            vals = set(I[lr, :].ravel().tolist()) | set(I[:, lc].ravel().tolist())
            if len(vals) != 1:
                continue
            L = next(iter(vals))
            inner = np.ones_like(I, bool)
            inner[lr, :] = False
            inner[:, lc] = False
            if L in set(I[inner].tolist()):
                continue
            if cfg is None or bh * bw < cfg[0] * cfg[1]:
                cfg = (bh, bw, lr, lc)
    if cfg is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    bh, bw, lr, lc = cfg
    inner = np.ones_like(I, bool)
    inner[lr, :] = False
    inner[:, lc] = False
    cnt = Counter(I[inner].tolist())
    if len(cnt) < 2:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    bgc = cnt.most_common()[0][0]    # marks are always fewer than half of a block
    fgc = cnt.most_common()[-1][0]   # the scattered mark colour

    blocks = []
    for r0 in range(0, hi, bh + 1):
        for c0 in range(0, wi, bw + 1):
            n = int((I[r0:r0 + bh, c0:c0 + bw] == fgc).sum())
            blocks.append((r0, c0, n))
    best = max(n for _, _, n in blocks)

    # Reading shared by the verifier and the golf entry (it differs from the
    # maker's version, which erased only the losing blocks one at a time):
    # 1) erase every mark in the grid at once: all cells of the mark colour
    #    become background,
    # 2) then fill each block that had the most marks with the mark colour.
    marks = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]
    ops.append(int(bgc)); sels.append(sel_of(marks))
    for r0, c0, n in blocks:
        if n == best:
            # the intended region is exactly this full block rectangle
            ops.append(int(fgc)); sels.append([r0, c0, bh - 1, bw - 1])

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29623171"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29623171"
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
                                f"for task 29623171"
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
                    f"Failed to build a complete episode for task 29623171 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29623171-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
