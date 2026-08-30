"""
ARC Task: c9f8e694 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------------------
# The generator draws a 1-cell "key line" (one colour per line) plus a set of
# solid squares of a single colour `sqc`, then rotates the whole thing.  So the
# key line ends up on one of the four edges:
#   identity -> left column,  rot90 -> top row,  rot180 -> right column,
#   rot270   -> bottom row.
# Rule: every square cell takes the colour of the key cell on its own
#       row (vertical key line) / column (horizontal key line).
# ---------------------------------------------------------------------------

_ROTS = ["identity", "rot90", "rot180", "rot270"]
# rot180 puts the key line on the right edge, rot270 on the bottom edge:
# those are the instances whose key line has to be reflected into the leading
# edge before the rule can be carried out.
_REFLECTED = ["rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    sqc = random.choice(cols)                      # colour of the squares
    rem = [c for c in cols if c != sqc]
    k = random.randint(3, min(7, len(rem)))
    palette = random.sample(rem, k)                # colours the key line uses

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_ROTS):
        examples = [{"rotf": r} for r in _ROTS]                       # cover all 4 edges
        examples += [{"rotf": random.choice(_REFLECTED)}
                     for _ in range(n_ex - len(_ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rotf": r} for r in random.sample(_ROTS, n_ex)]

    pool = [e for e in examples if e["rotf"] in _REFLECTED] or examples
    plan = examples + [dict(random.choice(pool))]  # test case is one of the shown ones
    return {"sqc": sqc, "palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, sqc, palette, rotf=None) -> dict:
    if rotf is None:
        rotf = random.choice(_ROTS)

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    bgc = 0
    swap = rotf in ("rot90", "rot270")             # rotation swaps the dimensions
    hcap = max(5, min(30, max_w if swap else max_h))
    wcap = max(5, min(30, max_h if swap else max_w))

    while True:
        h = unifint(diff_lb, diff_ub, (5, hcap))
        w = unifint(diff_lb, diff_ub, (5, wcap))
        gir = [[bgc] * (w - 1) for _ in range(h)]
        gil = [random.choice(palette) for _ in range(h)]
        nsq = unifint(diff_lb, diff_ub, (1, 8))
        succ, fails, maxfails = 0, 0, nsq * 5
        while succ < nsq and fails < maxfails:
            loci = random.randint(0, h - 3)
            locj = random.randint(0, w - 3)
            lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
            locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
            if locl <= w - 2:                       # backdrop must fit inside gir
                for r in range(loci, lock + 1):
                    for c in range(locj, locl + 1):
                        gir[r][c] = sqc
                succ += 1
            else:
                fails += 1
        if succ >= 1:
            break

    covered = {r for r in range(h) if any(v == sqc for v in gir[r])}
    gil = [gil[r] if r in covered else bgc for r in range(h)]
    gi = [[gil[r]] + list(gir[r]) for r in range(h)]
    go = [[gil[r]] + [gil[r] if v == sqc else bgc for v in gir[r]] for r in range(h)]

    def rot(g, kind):
        if kind == "identity":
            return [list(row) for row in g]
        if kind == "rot90":                          # clockwise
            return [list(row) for row in zip(*g[::-1])]
        if kind == "rot180":
            return [list(row)[::-1] for row in g[::-1]]
        return [list(row) for row in zip(*g)][::-1]  # rot270, counter-clockwise

    return {"input": rot(gi, rotf), "output": rot(go, rotf)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background is a hard 0 in this task; the squares are the dominant colour
    cnt = Counter(int(v) for v in I.flatten().tolist() if v != 0)
    sqc = cnt.most_common(1)[0][0]

    # the key line: the non-background cells that are not square cells
    strip = [(r, c) for r in range(h) for c in range(w)
             if I[r, c] != 0 and int(I[r, c]) != sqc]
    rows = {r for r, _ in strip}
    cols = {c for _, c in strip}

    ops, sels = [], []
    # whole-canvas rectangle: the reflection acts on the entire grid,
    # background included, so a bbox is exactly the intended cell set here.
    full = [0, 0, h - 1, w - 1]

    if len(cols) == 1:                      # key line is a column
        vertical = True
        far = (next(iter(cols)) != 0)       # sitting on the right edge?
    else:                                   # key line is a row
        vertical = False
        far = (next(iter(rows)) != 0)       # sitting on the bottom edge?

    G = I.copy()
    flip_op = None
    if far:
        flip_op = 26 if vertical else 27    # FlipH (l<->r) / FlipV (u<->d)
        ops.append(flip_op)
        sels.append(full)                   # reflect the key line onto the leading edge
        G = np.fliplr(G) if vertical else np.flipud(G)

    # In this reflected frame the key line leads: column 0 (or row 0).
    # Each key cell dyes its own line of square cells.
    if vertical:
        for r in range(h):
            key = int(G[r, 0])
            if key == 0:
                continue
            cells = [(r, c) for c in range(1, w) if int(G[r, c]) == sqc]
            if not cells:
                continue
            ops.append(key)
            sels.append(sel_of(cells))
            for rr, cc in cells:
                G[rr, cc] = key
    else:
        for c in range(w):
            key = int(G[0, c])
            if key == 0:
                continue
            cells = [(r, c) for r in range(1, h) if int(G[r, c]) == sqc]
            if not cells:
                continue
            ops.append(key)
            sels.append(sel_of(cells))
            for rr, cc in cells:
                G[rr, cc] = key

    if flip_op is not None:                 # reflect back into the original frame
        ops.append(flip_op)
        sels.append(full)

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
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
                                f"for task c9f8e694"
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
                    f"Failed to build a complete episode for task c9f8e694 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c9f8e694-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
