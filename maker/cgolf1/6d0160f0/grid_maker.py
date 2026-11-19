"""
ARC Task: 6d0160f0 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    cols = [c for c in range(10) if c != 4]

    def unif(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))

    def fits(n, m):
        return n * n + n - 1 <= m

    hmax = max([n for n in range(2, 6) if fits(n, max_h)] or [2])
    wmax = max([n for n in range(2, 6) if fits(n, max_w)] or [2])
    h = unif(2, hmax)
    w = unif(2, wmax)
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = np.full((fullh, fullw), bgc, dtype=int)
    for i in range(h, fullh, h + 1):
        gi[i, :] = linc
    for j in range(w, fullw, w + 1):
        gi[:, j] = linc
    go = gi.copy()
    noccs = unif(1, h * w)
    dense = [(a, b) for a in range(h) for b in range(w)]
    sparse = [(a * (h + 1), b * (w + 1)) for a, b in dense]
    locs = random.sample(sparse, noccs)
    trgtl = random.choice(locs)
    remlocs = [l for l in locs if l != trgtl]
    ntrgt = unif(1, h * w - 1)
    place = random.choice(dense)
    ncols = unif(1, 9)
    ccols = random.sample(cols, ncols)
    cands = [p for p in dense if p != place]
    trg = random.sample(cands, ntrgt)
    trgtobj = [(4, place)] + [(random.choice(ccols), ij) for ij in trg]
    for v, (a, b) in trgtobj:
        if v != linc:
            go[a + place[0] * (h + 1), b + place[1] * (w + 1)] = v
    for v, (a, b) in trgtobj:
        gi[a + trgtl[0], b + trgtl[1]] = v
    for rl in remlocs:
        tlo = random.choice(ccols)
        ncells = unif(1, h * w - 1)
        inds = random.sample(dense, ncells)
        pool = [c for c in ccols if c != tlo] if len(ccols) > 1 else ccols
        for (a, b) in inds:
            gi[a + rl[0], b + rl[1]] = random.choice(pool)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the output is the line lattice with every
    # block blank, plus the 4-marked block copied into the block whose index equals
    # the 4's offset inside its own block.  So: clear all blocks, copy the marked
    # block from the input, paste it into its target block, then touch up the cells
    # a transparent paste cannot handle.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # n blocks of n cells per side: n*n + n - 1 == side
    bh = next(n for n in range(1, 31) if n * n + n - 1 == hi)
    bw = next(n for n in range(1, 31) if n * n + n - 1 == wi)
    linc = int(I[bh, 0])

    def is_block(r, c):
        return r % (bh + 1) != bh and c % (bw + 1) != bw

    # background = the colour the demonstration outputs' blocks are cleared to
    cnt = Counter()
    if examples:
        for _, eo in examples:
            eo = np.asarray(eo, dtype=int)
            if eo.ndim != 2:
                continue
            eh = next((n for n in range(1, 31) if n * n + n - 1 == eo.shape[0]), None)
            ew = next((n for n in range(1, 31) if n * n + n - 1 == eo.shape[1]), None)
            if eh is None or ew is None:
                continue
            for r in range(eo.shape[0]):
                for c in range(eo.shape[1]):
                    if r % (eh + 1) != eh and c % (ew + 1) != ew:
                        cnt[int(eo[r, c])] += 1
    if not cnt:
        cnt = Counter(int(I[r, c]) for r in range(hi) for c in range(wi) if is_block(r, c))
    bgc = cnt.most_common(1)[0][0]

    ops, sels = [], []
    # the unique 4 marks the source block; its offset inside that block names the target
    p4 = next((r, c) for r in range(hi) for c in range(wi) if I[r, c] == 4)
    sr, sc = (p4[0] // (bh + 1)) * (bh + 1), (p4[1] // (bw + 1)) * (bw + 1)
    tr, tc = (p4[0] % (bh + 1)) * (bh + 1), (p4[1] % (bw + 1)) * (bw + 1)
    same = (sr, sc) == (tr, tc)

    def in_src(r, c):
        return sr <= r < sr + bh and sc <= c < sc + bw

    # 1. clear every block to background (all non-background cells off the lines);
    #    if the 4 points at its own block, that block is kept in place
    wipe = [(r, c) for r in range(hi) for c in range(wi)
            if is_block(r, c) and I[r, c] != bgc and not (same and in_src(r, c))]
    if wipe:
        ops.append(bgc); sels.append(sel_of(wipe))

    if not same:
        # 2. copy the marked block from the input (full block rectangle)
        ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
        # 3. paste it into the block the 4 points at
        ops.append(30); sels.append([tr, tc, 0, 0])

    # 4. touch-ups on the landed block:
    #    line-coloured cells of the marked block become background
    blk = [(a, b) for a in range(bh) for b in range(bw)]
    if linc != 0 or same:
        lc = [(tr + a, tc + b) for a, b in blk if I[sr + a, sc + b] == linc]
        if lc:
            ops.append(bgc); sels.append(sel_of(lc))
    #    0-valued content cells (a paste cannot carry 0) are painted explicitly
    if linc != 0 and bgc != 0 and not same:
        zc = [(tr + a, tc + b) for a, b in blk if I[sr + a, sc + b] == 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
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
                                f"for task 6d0160f0"
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
                    f"Failed to build a complete episode for task 6d0160f0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d0160f0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
