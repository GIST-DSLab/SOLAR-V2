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


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    V = [0, 1, 2, 3]                      # rotf: identity / rot90 / rot180 / rot270
    if n_ex >= len(V):
        ex = V[:] + [random.choice(V) for _ in range(n_ex - len(V))]
        random.shuffle(ex)
    else:
        ex = random.sample(V, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, dotc=2, rot=None, **kw) -> dict:
    def unifint(lb, ub):
        if ub <= lb:
            return lb
        return lb + int(round((ub - lb) * random.uniform(diff_lb, diff_ub)))
    if rot is None:
        rot = random.randint(0, 3)
    cap = min(10, max_h // 3, max_w // 3)     # output is 3h x 3w (and may be rotated)
    if cap < 3:
        raise ValueError("grid too small")
    h = unifint(3, cap)
    w = unifint(3, cap)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((3 * h, 3 * w), bgc, dtype=int)
    sqi = [(sqc, (0, 0)), (sqc, (0, 1)), (sqc, (1, 0)), (dotc, (1, 1))]
    sqo = [(a, b) for a in range(4) for b in range(4)] + \
          [(a + 4, b + 4) for a in range(4) for b in range(4)]

    def place(i, j):
        for col, (a, b) in sqi:
            gi[i + a, j + b] = col
        for a, b in sqo:
            go[i + a, j + b] = sqc

    def dn(r, c):
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    loci = random.randint(0, min(h - 2, 3 * h - 8))
    locj = random.randint(0, min(w - 2, 3 * w - 8))
    place(loci, locj)
    first = {(loci + a, locj + b) for _, (a, b) in sqi}
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc}
    for p in first:
        iinds -= dn(*p)
    noccs = unifint(0, (h * w) // 9)
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
            iinds -= cells
            for p in cells:
                iinds -= dn(*p)
            place(i, j)
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the golf entry): there is ONE canonical picture -- every seed's
    dot sits at the bottom-right of its 2x2 block, the canvas is 3h x 3w with the input in
    its top-left corner, and each seed becomes a 4x4 square from its top-left corner plus a
    second 4x4 square 4 cells further down-right. Every other orientation is that picture
    mirrored. So: mirror the input into the canonical orientation, grow the canvas, draw
    the canonical squares, then mirror the whole canvas back. Mirroring the whole canvas
    is what carries the picture into the opposite corner (the maker's explicit offset).
    Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                    # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None       # 1 cell per seed
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    DR = Counter(1 if (r - 1 >= 0 and I[r - 1, c] == sqc) else 0
                 for r, c in dots).most_common(1)[0][0]
    DC = Counter(1 if (c - 1 >= 0 and I[r, c - 1] == sqc) else 0
                 for r, c in dots).most_common(1)[0][0]

    ops, sels = [], []
    G = I.copy()
    # 1. mirror the whole input rectangle into the canonical orientation (dot bottom-right)
    if DR == 0:
        ops.append(27); sels.append([0, 0, hi - 1, wi - 1]); G = G[::-1, :]
    if DC == 0:
        ops.append(26); sels.append([0, 0, hi - 1, wi - 1]); G = G[:, ::-1]
    # 2. grow canvas to 3h x 3w
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    C = np.zeros((ho, wo), dtype=int)
    C[:hi, :wi] = G
    # 3. the rows / columns the resize just added become background (full rectangles)
    if bgc != 0:
        ops.append(bgc); sels.append([hi, 0, ho - 1 - hi, wo - 1]); C[hi:, :] = bgc
        ops.append(bgc); sels.append([0, wi, hi - 1, wo - 1 - wi]); C[:hi, wi:] = bgc
    # 4. canonical rule: each seed -> two 4x4 squares growing down-right from its corner
    cdots = [(r, c) for r in range(hi) for c in range(wi) if G[r, c] == dotc]
    for (rd, cd) in cdots:
        i, j = rd - 1, cd - 1
        cells = sorted({(i + a + 4 * k, j + b + 4 * k)
                        for k in (0, 1) for a in range(4) for b in range(4)
                        if 0 <= i + a + 4 * k < ho and 0 <= j + b + 4 * k < wo})
        if any(C[r, c] != sqc for r, c in cells):
            ops.append(sqc); sels.append(sel_of(cells))
            for r, c in cells:
                C[r, c] = sqc
    # 5. mirror the whole canvas back to the seeds' real orientation
    if DR == 0:
        ops.append(27); sels.append([0, 0, ho - 1, wo - 1])
    if DC == 0:
        ops.append(26); sels.append([0, 0, ho - 1, wo - 1])
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
