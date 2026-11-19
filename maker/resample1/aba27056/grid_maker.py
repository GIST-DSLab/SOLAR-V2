"""
ARC Task: aba27056 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of clockwise quarter turns applied to the canonical (hole-on-top) layout


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, rot=None, **kw) -> dict:
    def unifint(lb, ub):
        if ub <= lb:
            return lb
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, lb + int(round(d * (ub - lb)))))

    if bgc is None or sqc is None:
        bgc, sqc = random.sample([c for c in range(10) if c != 4], 2)
    if rot is None:
        rot = random.choice(ROTS)
    hmax, wmax = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    hmax = max(6, min(30, hmax))
    wmax = max(6, min(30, wmax))
    h = unifint(6, hmax)
    w = unifint(6, wmax)
    oh = random.randint(3, h)
    ow = unifint(5, w - 1)
    loci = unifint(0, h - oh)
    locj = random.randint(0, w - ow)
    r0, r1, c0, c1 = loci, loci + oh - 1, locj, locj + ow - 1

    gi = np.full((h, w), bgc, dtype=int)
    gi[r0, c0:c1 + 1] = sqc
    gi[r1, c0:c1 + 1] = sqc
    gi[r0:r1 + 1, c0] = sqc
    gi[r0:r1 + 1, c1] = sqc
    k = random.randint(0, (ow - 4) // 2)
    a, b = locj + 2 + k, locj + ow - 3 - k
    hl, hr = min(a, b), max(a, b)
    gi[loci, hl:hr + 1] = bgc

    go = np.full((h, w), bgc, dtype=int)
    go[r0:r1 + 1, c0:c1 + 1] = 4
    go[r0, c0:c1 + 1] = sqc
    go[r1, c0:c1 + 1] = sqc
    go[r0:r1 + 1, c0] = sqc
    go[r0:r1 + 1, c1] = sqc
    go[0:loci + 1, hl:hr + 1] = 4
    r, c = loci - 1, hr + 1
    while 0 <= r < h and 0 <= c < w:
        go[r, c] = 4
        r -= 1
        c += 1
    r, c = loci - 1, hl - 1
    while 0 <= r < h and 0 <= c < w:
        go[r, c] = 4
        r -= 1
        c -= 1

    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    colors = sorted(set(I.flatten().tolist()))

    # The frame colour is the one whose cells all lie on the border of their own bbox.
    sqc = None
    for c in colors:
        rs, cs = np.where(I == c)
        a, b, d, e = rs.min(), rs.max(), cs.min(), cs.max()
        if b - a >= 2 and e - d >= 2 and all(r in (a, b) or q in (d, e) for r, q in zip(rs, cs)):
            sqc = c
    bgc = [c for c in colors if c != sqc][0]
    rs, cs = np.where(I == sqc)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()

    ops, sels = [], []

    # 1. Fill the inside of the box with 4.
    inner = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    ops.append(4); sels.append(sel_of(inner))

    # 2. Find the gap in the frame and which way it faces.
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c)); border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0)); border.add((r, c1))
    hole = sorted(p for p in border if I[p] == bgc)
    if all(r == r0 for r, _ in hole):
        d = (-1, 0)
    elif all(r == r1 for r, _ in hole):
        d = (1, 0)
    elif all(c == c0 for _, c in hole):
        d = (0, -1)
    else:
        d = (0, 1)
    p = (abs(d[1]), abs(d[0]))  # perpendicular axis

    def shoot(start, v):
        out = []
        r, c = start
        while 0 <= r < H and 0 <= c < W:
            out.append((r, c))
            r += v[0]
            c += v[1]
        return out

    # 3. Draw the beam: the gap cells plus everything straight out from them to the edge.
    bar = []
    for cell in hole:
        bar += shoot(cell, d)
    ops.append(4); sels.append(sel_of(sorted(set(bar))))

    # 4. Draw the two diagonal rays that start just outside the ends of the gap.
    key = lambda q: q[0] * p[0] + q[1] * p[1]
    emax = max(hole, key=key)
    emin = min(hole, key=key)
    for e, s in ((emax, 1), (emin, -1)):
        v = (d[0] + s * p[0], d[1] + s * p[1])
        diag = shoot((e[0] + v[0], e[1] + v[1]), v)
        if diag:
            ops.append(4); sels.append(sel_of(diag))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(H) for c in range(W)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task aba27056"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task aba27056"
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
                                f"for task aba27056"
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
                    f"Failed to build a complete episode for task aba27056 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"aba27056-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
