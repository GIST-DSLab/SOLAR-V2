"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

ROTATIONS = [0, 1, 2, 3]  # number of 90-degree turns applied to (input, output)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]          # 8 is the fixed "gap" colour
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        examples = [{"rot": r} for r in ROTATIONS]
        examples += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTATIONS)
    # rotation by 90/270 swaps the canvas axes
    mh, mw = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    mh, mw = min(30, mh), min(30, mw)
    h = unifint(diff_lb, diff_ub, (3, mh))
    w = unifint(diff_lb, diff_ub, (3, mw))
    barth = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)

    gi = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc
    go = np.full((h, w), bgc, dtype=int)

    def fill_shift(g, col, dr, dc):
        a, b = 0 + dr, loci + dr
        c, d = locj + dc, locj + barth - 1 + dc
        a, b, c, d = max(0, a), min(h - 1, b), max(0, c), min(w - 1, d)
        if a <= b and c <= d:
            g[a:b + 1, c:d + 1] = col

    for k in range(16):
        fill_shift(go, fgc, -2 * k, 2 * k * barth)
        fill_shift(go, fgc, -2 * k, -2 * k * barth)
        fill_shift(go, 8, -(2 * k + 1), (2 * k + 1) * barth)
        fill_shift(go, 8, -(2 * k + 1), -(2 * k + 1) * barth)

    # np.rot90 k=3 is clockwise (DSL rot90)
    gi = np.rot90(gi, k=(4 - rot) % 4)
    go = np.rot90(go, k=(4 - rot) % 4)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): a staircase that grows OUT of the bar.
    # Each new strip comes from the strip just before it: move it one cell toward
    # the edge the bar touches and one bar-thickness sideways, then swap its colour
    # (fgc <-> 8). The maker shifts the ORIGINAL bar k times and alternates sides at
    # each k. Here the unit is "the previous strip", and each side is grown fully
    # outward before the other side starts.
    # Each strip is a full (clipped) rectangle, so the whole rectangle is painted.
    # Copy/Paste is not used: fgc may be 0, which Copy would skip, and a Paste
    # that a recolour then fully covers would be a wasted op.
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    mask = I != bgc
    if not mask.any():
        return [34], [[0, 0, hi - 1, wi - 1]]
    rs, cs = np.where(mask)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

    # the bar touches exactly one edge; the staircase climbs toward that edge
    if r0 == 0:
        along, perp = (-1, 0), (0, bw)
    elif r1 == hi - 1:
        along, perp = (1, 0), (0, bw)
    elif c0 == 0:
        along, perp = (0, -1), (bh, 0)
    else:
        along, perp = (0, 1), (bh, 0)

    ops, sels = [], []
    for sgn in (1, -1):                       # one flank, then the other
        a, b, c, d = r0, r1, c0, c1           # previous strip = the bar itself
        prev_col = fgc
        while True:
            dr = along[0] + sgn * perp[0]
            dc = along[1] + sgn * perp[1]
            a, b = max(0, a + dr), min(hi - 1, b + dr)
            c, d = max(0, c + dc), min(wi - 1, d + dc)
            if a > b or c > d:                # staircase has left the grid
                break
            col = 8 if prev_col == fgc else fgc
            ops.append(col)
            sels.append(sel_of([(r, q) for r in range(a, b + 1)
                                for q in range(c, d + 1)]))
            prev_col = col

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
