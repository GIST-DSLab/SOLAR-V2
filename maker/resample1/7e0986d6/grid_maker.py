"""
ARC Task: 7e0986d6 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    nsq = random.randint(1, 5)
    sqcols = random.sample(rem, nsq)
    rem = [c for c in rem if c not in sqcols]
    nn = random.randint(1, len(rem))
    noisecols = random.sample(rem, nn)
    return {"bgc": bgc, "sqcols": sqcols, "noisecols": noisecols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqcols, noisecols) -> dict:
    h = random.randint(10, max(10, max_h))
    w = random.randint(10, max(10, max_w))
    numsq = random.randint(1, max(1, (h * w) // 25))
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    maxtr = 5 * numsq
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        sq = {(i, j) for i in range(li, li + oh) for j in range(lj, lj + ow)}
        if sq <= inds:
            succ += 1
            ob = {(i, j) for i in range(li - 1, li + oh + 1) for j in range(lj - 1, lj + ow + 1)}
            inds = inds - ob
            col = random.choice(sqcols)
            for i, j in sq:
                go[i][j] = col
    gi = [row[:] for row in go]
    namt = random.randint(1, max(1, (h * w) // 9))
    cands = {(i, j) for i in range(h) for j in range(w)}
    for _ in range(namt):
        if not cands:
            break
        loc = random.choice(sorted(cands))
        col = gi[loc[0]][loc[1]]
        torem = {(loc[0] + a, loc[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}
        torem = {(i, j) for i, j in torem if 0 <= i < h and 0 <= j < w and gi[i][j] == col}
        cands = cands - torem
        gi[loc[0]][loc[1]] = random.choice(noisecols)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    sq_known, noise_known = set(), set()
    bgc = None
    if examples:
        outc, inc = Counter(), set()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            outc.update(eo.flatten().tolist())
            inc |= set(ei.flatten().tolist())
        bgc = outc.most_common(1)[0][0]
        sq_known = set(outc) - {bgc}          # colours that survive into outputs = square colours
        noise_known = inc - set(outc)         # colours only in inputs = noise colours
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected same-colour components (noise can never split a square 8-wise)
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True
                            stack.append((x, y))
            comps.append((int(col), cells))

    def is_square(col, cells):
        if col in sq_known:
            return True
        if col in noise_known:
            return False
        return (len(cells) >= 3 and len({r for r, _ in cells}) >= 2
                and len({c for _, c in cells}) >= 2)

    ops, sels = [], []
    G = I.copy()
    sqcols = {col for col, cells in comps if is_square(col, cells)}
    # 1. each square: repaint the noise cells inside its rectangle with the square colour
    for col, cells in comps:
        if col not in sqcols:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        holes = [(r, c) for r in range(min(rs), max(rs) + 1)
                 for c in range(min(cs), max(cs) + 1) if G[r, c] != col]
        if holes:
            ops.append(col)
            sels.append(sel_of(holes))
            for r, c in holes:
                G[r, c] = col
    # 2. remaining noise pixels on the background -> background colour
    rest = [(r, c) for r in range(h) for c in range(w)
            if G[r, c] != bgc and G[r, c] not in sqcols]
    if rest:
        ops.append(bgc)
        sels.append(sel_of(rest))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole grid for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
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
                                f"for task 7e0986d6"
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
                    f"Failed to build a complete episode for task 7e0986d6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7e0986d6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
