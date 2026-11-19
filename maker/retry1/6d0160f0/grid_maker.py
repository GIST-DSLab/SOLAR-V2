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

COLS = [0, 1, 2, 3, 5, 6, 7, 8, 9]


def sample_colors(num_examples=None) -> dict:
    # The background and separator colours stay the same for the whole episode.
    # The solver reads the background from the demonstrations, because the input
    # alone cannot always settle it.
    bgc, linc = random.sample(COLS, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    cols = list(COLS)
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)
    hmax = max(k for k in range(2, 6) if k * k + k - 1 <= max_h) if max_h >= 5 else 2
    wmax = max(k for k in range(2, 6) if k * k + k - 1 <= max_w) if max_w >= 5 else 2
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = np.full((fullh, fullw), bgc, dtype=int)
    for iloc in range(h, fullh, h + 1):
        gi[iloc, :] = linc
    for jloc in range(w, fullw, w + 1):
        gi[:, jloc] = linc
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = [(a, b) for a in range(h) for b in range(w)]
    sparseinds = [(a * (h + 1), b * (w + 1)) for a, b in denseinds]
    locs = random.sample(sparseinds, noccs)
    trgtl = random.choice(locs)
    remlocs = [l for l in locs if l != trgtl]
    ntrgt = unifint(diff_lb, diff_ub, (1, h * w - 1))
    place = random.choice(denseinds)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(cols, ncols)
    candss = [ij for ij in denseinds if ij != place]
    trgrem = random.sample(candss, ntrgt)
    trgtobj = [(4, place)] + [(random.choice(ccols), ij) for ij in trgrem]
    go = gi.copy()
    offr, offc = place[0] * (h + 1), place[1] * (w + 1)
    for v, (i, j) in trgtobj:
        if v != linc:
            go[i + offr, j + offc] = v
    for v, (i, j) in trgtobj:
        gi[i + trgtl[0], j + trgtl[1]] = v
    for rl in remlocs:
        tlo = random.choice(ccols)
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = random.sample(denseinds, ncells)
        pool = [c for c in ccols if c != tlo] if len(ccols) > 1 else ccols
        for (i, j) in inds:
            gi[i + rl[0], j + rl[1]] = random.choice(pool)
    return {"input": gi.tolist(), "output": go.tolist()}


def _block_size(n):
    return next(k for k in range(1, 31) if k * k + k - 1 == n)


def _block_cells(shape):
    hh, ww = shape
    h, w = _block_size(hh), _block_size(ww)
    return [(r, c) for r in range(hh) for c in range(ww)
            if r % (h + 1) != h and c % (w + 1) != w]


def derive_operations(I, O, examples=None):
    # Work on the whole grid: first clear every block cell to background in one
    # stroke, then COPY the marked tile from the unchanged input, paste it into
    # its destination block, and fix what the transparent paste cannot express.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h, w = _block_size(hi), _block_size(wi)
    linc = int(I[h, 0])
    blk = _block_cells(I.shape)

    # Background colour is fixed for the episode. Read it from the block cells
    # of the demonstration outputs (each one uses its own grid shape). Those
    # blocks are almost all background.
    bgc = None
    if examples:
        cnt = Counter()
        for _, eo in examples:
            eo = np.asarray(eo, dtype=int)
            try:
                cells = _block_cells(eo.shape)
            except StopIteration:
                continue
            cnt.update(int(eo[r, c]) for r, c in cells)
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None or bgc == linc or bgc == 4:
        bgc = Counter(int(I[r, c]) for r, c in blk
                      if int(I[r, c]) != 4).most_common(1)[0][0]

    # The single 4 marks the source tile. Its position inside that tile names
    # the destination block.
    p = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 4][0]
    sr, sc = (p[0] // (h + 1)) * (h + 1), (p[1] // (w + 1)) * (w + 1)
    tr, tc = (p[0] % (h + 1)) * (h + 1), (p[1] % (w + 1)) * (w + 1)
    src = [(r, c) for r in range(sr, sr + h) for c in range(sc, sc + w)]

    ops, sels = [], []
    moved = (sr, sc) != (tr, tc)
    if moved:
        # 1. clear every block cell (all non-separator cells) to background
        ops.append(bgc); sels.append(sel_of(blk))
        # 2. copy the marked tile from the input (the full tile rectangle)
        ops.append(28); sels.append([sr, sc, h - 1, w - 1])
        # 3. paste it into the destination block
        ops.append(30); sels.append([tr, tc, 0, 0])
        # 4. the paste skips tile cells that hold 0 (only real content when linc != 0)
        if linc != 0 and bgc != 0:
            zeros = [(r - sr + tr, c - sc + tc) for r, c in src if I[r, c] == 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))
    else:
        # the tile already sits in its destination: clear every other block
        srcset = set(src)
        others = [(r, c) for r, c in blk if (r, c) not in srcset]
        if any(I[r, c] != bgc for r, c in others):
            ops.append(bgc); sels.append(sel_of(others))
    # 5. tile cells in the separator colour become background
    #    (if linc == 0 the paste never wrote them, so they are already background)
    lcells = [(r - sr + tr, c - sc + tc) for r, c in src if I[r, c] == linc]
    if lcells and linc != bgc and (linc != 0 or not moved):
        ops.append(bgc); sels.append(sel_of(lcells))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])  # whole-grid rectangle for Submit
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
