"""
ARC Task: 4522001f (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS] + [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, sqc=1, dotc=2, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    cap = max(3, min(10, max_h // 3, max_w // 3))
    h = unifint(diff_lb, diff_ub, (3, cap))
    w = unifint(diff_lb, diff_ub, (3, cap))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * (3 * w) for _ in range(3 * h)]
    sqi = [(dotc, (1, 1)), (sqc, (0, 0)), (sqc, (0, 1)), (sqc, (1, 0))]
    sqo = [(a, b) for a in range(4) for b in range(4)]
    sqo += [(a + 4, b + 4) for a, b in sqo]

    def nb(cells):
        out = set()
        for r, c in cells:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    def place(i, j):
        cells = set()
        for col, (a, b) in sqi:
            gi[i + a][j + b] = col
            cells.add((i + a, j + b))
        for a, b in sqo:
            go[i + a][j + b] = sqc
        return cells

    loci = random.randint(0, min(h - 2, 3 * h - 8))
    locj = random.randint(0, min(w - 2, 3 * w - 8))
    pc = place(loci, locj)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc} - nb(pc)
    succ = tr = 0
    while tr < 10 * noccs and succ < noccs:
        tr += 1
        cands = [p for p in iinds if p[0] <= h - 2 and p[1] <= w - 2]
        if not cands:
            break
        i, j = random.choice(sorted(cands))
        cells = {(i + a, j + b) for _, (a, b) in sqi}
        if cells <= iinds:
            succ += 1
            place(i, j)
            iinds = (iinds - cells) - nb(cells)
    gi = np.rot90(np.array(gi), -rot)
    go = np.rot90(np.array(go), -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the code-golf entry): there is ONE canonical picture -- dot in
    the bottom-right of each 2x2 seed, squares growing down-right from the seed's
    top-left, canvas anchored top-left. Every other orientation is that same picture
    mirrored. (The seed and the two-square pattern are both symmetric under transpose,
    so any rotation reduces to flips.)
    The maker treats each orientation separately: it shifts the picture to the opposite
    corner, then erases the seed cells left behind. This reading treats the orientation
    as a mirror around one fixed picture instead:
      1. mirror the input into the canonical orientation,
      2. grow the canvas to 3h x 3w (fill the new area with background),
      3. draw each seed's two 4x4 squares down-right (the first square covers the seed,
         so nothing needs erasing),
      4. mirror the whole canvas back.
    Nothing reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                   # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None      # 1 cell per seed

    # orientation of the seeds: which corner of its 2x2 block holds the dot
    votes = []
    for rd in range(hi):
        for cd in range(wi):
            if I[rd, cd] != dotc:
                continue
            dr = 1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0
            dc = 1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0
            votes.append((dr, dc))
    DR = Counter(v[0] for v in votes).most_common(1)[0][0]
    DC = Counter(v[1] for v in votes).most_common(1)[0][0]

    G = I.copy()
    ops, sels = [], []

    def emit(op, sel, newG):
        # tracks the grid as it stands now; an op with no visible effect is not emitted
        nonlocal G
        if np.array_equal(newG, G):
            return
        G = newG
        ops.append(op)
        sels.append(sel)

    def flips():
        # whole-canvas mirror: the full rectangle is the intended selection
        h, w = G.shape
        if DC == 0:
            emit(26, [0, 0, h - 1, w - 1], G[:, ::-1].copy())
        if DR == 0:
            emit(27, [0, 0, h - 1, w - 1], G[::-1, :].copy())

    # 1. mirror the input into canonical orientation (dot bottom-right)
    flips()

    # 2. grow canvas to 3h x 3w; the rows/cols the resize added become background
    big = np.zeros((ho, wo), dtype=int)
    big[:hi, :wi] = G
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    G = big
    if bgc != 0:
        g = G.copy(); g[hi:, :] = bgc
        emit(bgc, [hi, 0, ho - 1 - hi, wo - 1], g)
        g = G.copy(); g[:hi, wi:] = bgc
        emit(bgc, [0, wi, hi - 1, wo - 1 - wi], g)

    # 3. each canonical seed (2x2 block whose bottom-right cell is the dot):
    #    paint a 4x4 square at its top-left, and another 4 cells further down-right
    seeds = sorted((r - 1, c - 1) for r in range(1, hi) for c in range(1, wi)
                   if G[r, c] == dotc and G[r - 1, c - 1] == sqc
                   and G[r - 1, c] == sqc and G[r, c - 1] == sqc)
    for (i, j) in seeds:
        for k in (0, 1):
            r0, c0 = i + 4 * k, j + 4 * k
            r1, c1 = min(r0 + 3, ho - 1), min(c0 + 3, wo - 1)
            if r0 > r1 or c0 > c1:
                continue
            g = G.copy(); g[r0:r1 + 1, c0:c1 + 1] = sqc
            emit(sqc, [r0, c0, r1 - r0, c1 - c0], g)   # the full 4x4 square is the target

    # 4. mirror the whole canvas back to the input's orientation
    flips()

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
                        f"num_examples+1 ({num_examples + 1}) for task 4522001f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4522001f"
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
                                f"for task 4522001f"
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
                    f"Failed to build a complete episode for task 4522001f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4522001f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
