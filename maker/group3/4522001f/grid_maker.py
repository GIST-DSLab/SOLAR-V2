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


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    rots = [0, 1, 2, 3]
    if n_ex >= 4:
        ex = rots[:] + [random.choice(rots) for _ in range(n_ex - 4)]
        random.shuffle(ex)
    else:
        ex = random.sample(rots, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, rot=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a)); hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))
    if rot is None:
        rot = random.randint(0, 3)
    h = unifint(diff_lb, diff_ub, (3, min(10, max_h // 3)))
    w = unifint(diff_lb, diff_ub, (3, min(10, max_w // 3)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * (3 * w) for _ in range(3 * h)]
    sqi = [(dotc, (1, 1)), (sqc, (0, 0)), (sqc, (0, 1)), (sqc, (1, 0))]
    sqo = [(a, b) for a in range(4) for b in range(4)] + \
          [(a + 4, b + 4) for a in range(4) for b in range(4)]

    def place(loc):
        for col, (a, b) in sqi:
            gi[a + loc[0]][b + loc[1]] = col
        for a, b in sqo:
            r, c = a + loc[0], b + loc[1]
            if 0 <= r < 3 * h and 0 <= c < 3 * w:
                go[r][c] = sqc

    def dn(cells):
        s = set()
        for r, c in cells:
            s |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return s

    loc = (random.randint(0, min(h - 2, 3 * h - 8)), random.randint(0, min(w - 2, 3 * w - 8)))
    place(loc)
    first = {(a + loc[0], b + loc[1]) for _, (a, b) in sqi}
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r][c] == bgc} - dn(first)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    succ = tr = 0
    while tr < 10 * noccs and succ < noccs:
        tr += 1
        cands = [p for p in iinds if p[0] <= h - 2 and p[1] <= w - 2]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        pc = {(a + loc[0], b + loc[1]) for _, (a, b) in sqi}
        if pc <= iinds:
            succ += 1
            iinds = (iinds - pc) - dn(pc)
            place(loc)
    gi = np.rot90(np.array(gi), -rot)   # k=-1 == DSL rot90 (CW)
    go = np.rot90(np.array(go), -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Rule: canvas grows to 3h x 3w; every seed (3 square-colour cells + 1 dot in a 2x2)
    becomes two 4x4 squares growing diagonally toward the dot, the picture anchored in
    the canvas corner opposite the dot direction.
    Unit of work: the whole new canvas as ONE background base layer, then each 4x4
    square as its OWN full rectangle (clipped to the canvas). Nothing reads O.
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
    blocks = []
    for (rd, cd) in dots:
        dr = 1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0
        dc = 1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0
        blocks.append((rd - dr, cd - dc, dr, dc))
    DR = Counter(b[2] for b in blocks).most_common(1)[0][0]
    DC = Counter(b[3] for b in blocks).most_common(1)[0][0]
    offr, offc = (1 - DR) * 2 * hi, (1 - DC) * 2 * wi
    sr, sc = 2 * DR - 1, 2 * DC - 1

    G = np.zeros((ho, wo), dtype=int)
    G[:hi, :wi] = I
    ops, sels = [33], [[0, 0, ho - 1, wo - 1]]          # grow canvas to 3h x 3w

    def paint_rect(col, r0, c0, r1, c1):
        r0, r1 = max(r0, 0), min(r1, ho - 1)
        c0, c1 = max(c0, 0), min(c1, wo - 1)
        if r0 > r1 or c0 > c1:
            return
        if np.all(G[r0:r1 + 1, c0:c1 + 1] == col):     # would change nothing now
            return
        G[r0:r1 + 1, c0:c1 + 1] = col
        # intended cells are exactly this full rectangle
        ops.append(col); sels.append([r0, c0, r1 - r0, c1 - c0])

    # base layer: the whole new canvas becomes background (padding and seeds alike)
    paint_rect(bgc, 0, 0, ho - 1, wo - 1)

    # each seed: its near 4x4 square, then its far 4x4 square, one rectangle each
    for (i, j, _, _) in blocks:
        ar = i + 1 - DR + offr
        ac = j + 1 - DC + offc
        for k in (0, 1):
            ra, ca = ar + 4 * k * sr, ac + 4 * k * sc
            rb, cb = ra + 3 * sr, ca + 3 * sc
            paint_rect(sqc, min(ra, rb), min(ca, cb), max(ra, rb), max(ca, cb))

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
