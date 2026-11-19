"""
ARC Task: 3bd67248 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (2, 4)]
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        return random.randint(min(lo, hi), max(lo, hi))

    if rot is None:
        rot = random.choice(ROTS)
    cap = max(3, min(15, max_h, max_w))
    h = unifint(diff_lb, diff_ub, (3, cap))
    w = unifint(diff_lb, diff_ub, (3, cap))
    fac = unifint(diff_lb, diff_ub, (1, max(1, min(max_h, max_w) // max(h, w))))
    gi = np.full((h, w), bgc, dtype=int)
    gi[:, 0] = linc
    go = gi.copy()
    go[h - 1, 1:] = 4
    r, c = h - 2, 1
    while r >= 0 and c < w:
        go[r, c] = 2
        r -= 1
        c += 1
    # rotf choice: identity, rot90 (CW), rot180, rot270 (CCW)
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    gi = np.kron(gi, np.ones((fac, fac), dtype=int))
    go = np.kron(go, np.ones((fac, fac), dtype=int))
    return {"input": tuple(tuple(int(x) for x in row) for row in gi),
            "output": tuple(tuple(int(x) for x in row) for row in go)}


def derive_operations(I, O, examples=None):
    # Reading (from the golf entry): in the frame where the line runs down the
    # left edge and the 4-band lies along the bottom, the task is a walk over
    # block-columns v >= 1. Each column gets two paints: the ray block v steps
    # up from the band turns 2, and the band block at the foot of the SAME
    # column turns 4. Walk the columns from the far end back toward the corner,
    # painting the ray block and then the band block in each column. The maker
    # paints the whole band in one op and then the ray; this pairs them column
    # by column. All of it comes from I (line edge, thickness = upscale factor).
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    cnt = Counter(I.flatten().tolist())

    # line colour: the minority colour (shown the same way in every demonstration)
    linc = None
    if examples:
        votes = Counter()
        for ei, _ in examples:
            e = np.asarray(ei, dtype=int)
            ec = Counter(e.flatten().tolist())
            votes[sorted(ec.items(), key=lambda kv: (kv[1], kv[0]))[0][0]] += 1
        cand = votes.most_common(1)[0][0]
        if cand in cnt:
            linc = cand
    if linc is None:
        linc = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    full = sel_of([(r, c) for r in range(H) for c in range(W)])

    if np.all(I[:, 0] == linc):
        edge = "left"
    elif np.all(I[:, W - 1] == linc):
        edge = "right"
    elif np.all(I[0, :] == linc):
        edge = "top"
    elif np.all(I[H - 1, :] == linc):
        edge = "bottom"
    else:
        return [34], [full]

    # line thickness == upscale factor
    fac = 0
    if edge == "left":
        while fac < W and np.all(I[:, fac] == linc):
            fac += 1
    elif edge == "right":
        while fac < W and np.all(I[:, W - 1 - fac] == linc):
            fac += 1
    elif edge == "top":
        while fac < H and np.all(I[fac, :] == linc):
            fac += 1
    else:
        while fac < H and np.all(I[H - 1 - fac, :] == linc):
            fac += 1
    fac = max(1, fac)
    while fac > 1 and (H % fac or W % fac):
        fac -= 1
    RB, CB = H // fac, W // fac

    # u = distance from the band edge, v = distance from the line edge.
    # Band edge = line edge turned 90 deg CCW.
    if edge == "left":
        U, V = RB, CB
        def blk(u, v): return (RB - 1 - u, v)
    elif edge == "top":
        U, V = CB, RB
        def blk(u, v): return (v, u)
    elif edge == "right":
        U, V = RB, CB
        def blk(u, v): return (u, CB - 1 - v)
    else:
        U, V = CB, RB
        def blk(u, v): return (RB - 1 - v, CB - 1 - u)

    def block_cells(rb, cb):
        return [(rb * fac + dr, cb * fac + dc)
                for dr in range(fac) for dc in range(fac)]

    ops, sels = [], []
    for v in range(V - 1, 0, -1):
        if v <= U - 1:
            # ray block in this column: v steps away from the band
            ops.append(2)
            sels.append(sel_of(block_cells(*blk(v, v))))
        # band block at the foot of the same column
        ops.append(4)
        sels.append(sel_of(block_cells(*blk(0, v))))

    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
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
                                f"for task 3bd67248"
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
                    f"Failed to build a complete episode for task 3bd67248 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bd67248-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
