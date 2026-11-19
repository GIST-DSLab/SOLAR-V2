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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of

_COLS = [c for c in range(10) if c != 4]


def sample_colors(num_examples=None) -> dict:
    bgc, linc = random.sample(_COLS, 2)
    return {"bgc": bgc, "linc": linc}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    a = lo + int(round(d * (hi - lo) * 0.999))
    b = max(a, lo + int(round(diff_ub * (hi - lo))))
    return random.randint(min(a, hi), min(b, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(_COLS, 2)
    # grid side = k*k + k - 1 must fit
    hmax = max([k for k in range(2, 6) if k * k + k - 1 <= max_h] or [2])
    wmax = max([k for k in range(2, 6) if k * k + k - 1 <= max_w] or [2])
    h = _unif(diff_lb, diff_ub, 2, hmax)
    w = _unif(diff_lb, diff_ub, 2, wmax)
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = [[bgc] * fullw for _ in range(fullh)]
    for i in range(h, fullh, h + 1):
        for j in range(fullw):
            gi[i][j] = linc
    for j in range(w, fullw, w + 1):
        for i in range(fullh):
            gi[i][j] = linc
    go = [row[:] for row in gi]
    noccs = _unif(diff_lb, diff_ub, 1, h * w)
    denseinds = [(a, b) for a in range(h) for b in range(w)]
    sparseinds = [(a * (h + 1), b * (w + 1)) for a, b in denseinds]
    locs = random.sample(sparseinds, noccs)
    trgtl = random.choice(locs)
    remlocs = [l for l in locs if l != trgtl]
    ntrgt = _unif(diff_lb, diff_ub, 1, h * w - 1)
    place = random.choice(denseinds)
    ncols = _unif(diff_lb, diff_ub, 1, 9)
    ccols = random.sample(_COLS, ncols)
    candss = [ij for ij in denseinds if ij != place]
    trgrem = random.sample(candss, ntrgt)
    trgtobj = [(4, place)] + [(random.choice(ccols), ij) for ij in trgrem]
    oi, oj = place[0] * (h + 1), place[1] * (w + 1)
    for v, (a, b) in trgtobj:
        if v != linc:
            go[a + oi][b + oj] = v
        gi[a + trgtl[0]][b + trgtl[1]] = v
    for rl in remlocs:
        tlo = random.choice(ccols)
        ncells = _unif(diff_lb, diff_ub, 1, h * w - 1)
        inds = random.sample(denseinds, ncells)
        pool = [c for c in ccols if c != tlo] if len(ccols) > 1 else ccols
        for (a, b) in inds:
            gi[a + rl[0]][b + rl[1]] = random.choice(pool)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # frontier lines split the grid into equal blocks
    line_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bh, bw = line_rows[0], line_cols[0]
    linc = int(I[line_rows[0], 0])
    lr, lc = set(line_rows), set(line_cols)
    block_cells = [(r, c) for r in range(hi) for c in range(wi)
                   if r not in lr and c not in lc]

    # background: block colour of the demonstrated outputs (fallback: input)
    cnt = Counter()
    if examples:
        for _, eo in examples:
            eo = np.asarray(eo, dtype=int)
            for r in range(eo.shape[0]):
                for c in range(eo.shape[1]):
                    if r % (bh + 1) != bh and c % (bw + 1) != bw:
                        cnt[int(eo[r, c])] += 1
    if not cnt:
        cnt = Counter(int(I[r, c]) for r, c in block_cells)
    bgc = cnt.most_common(1)[0][0]

    # the 4 marks the source block; its in-block position names the destination
    p4 = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == 4][0]
    sbr, sbc = p4[0] // (bh + 1), p4[1] // (bw + 1)
    dbr, dbc = p4[0] % (bh + 1), p4[1] % (bw + 1)
    sr, sc = sbr * (bh + 1), sbc * (bw + 1)
    tr, tc = dbr * (bh + 1), dbc * (bw + 1)
    src_cells = [(r, c) for r in range(sr, sr + bh) for c in range(sc, sc + bw)]
    src_set = set(src_cells)
    linc_cells = [(r, c) for r, c in src_cells if int(I[r, c]) == linc]
    zero_cells = [(r, c) for r, c in src_cells if int(I[r, c]) == 0]

    if (sbr, sbc) == (dbr, dbc):
        # block stays: line-coloured cells inside it become background
        if linc_cells:
            ops.append(bgc); sels.append(sel_of(linc_cells))
        others = [p for p in block_cells if p not in src_set]
        if any(int(I[r, c]) != bgc for r, c in others):
            ops.append(bgc); sels.append(sel_of(others))   # wipe all other blocks
    else:
        dr, dc = tr - sr, tc - sc
        # copy the whole marked block (full rectangle)
        ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
        # wipe every block region to background in one stroke
        ops.append(bgc); sels.append(sel_of(block_cells))
        # paste the block into its destination
        ops.append(30); sels.append([tr, tc, 0, 0])
        # transparent paste could not carry 0-coloured content
        if bgc != 0 and linc != 0 and zero_cells:
            ops.append(0); sels.append(sel_of([(r + dr, c + dc) for r, c in zero_cells]))
        # line-coloured content becomes background
        if linc != 0 and linc_cells:
            ops.append(bgc); sels.append(sel_of([(r + dr, c + dc) for r, c in linc_cells]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole-grid rectangle for Submit
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
