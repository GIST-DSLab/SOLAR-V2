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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))

    if bgc is None or sqc is None:
        cols = [c for c in range(10) if c != 4]
        bgc, sqc = random.sample(cols, 2)
    m = max(6, min(30, max_h, max_w))   # rotation may swap h/w
    h = unifint(diff_lb, diff_ub, (6, m))
    w = unifint(diff_lb, diff_ub, (6, m))
    gi = np.full((h, w), bgc, dtype=int)
    oh = random.randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = random.randint(0, w - ow)
    r0, c0, r1, c1 = loci, locj, loci + oh - 1, locj + ow - 1
    k = random.randint(0, (ow - 4) // 2)
    a, b = sorted((locj + 2 + k, locj + ow - 3 - k))
    go = gi.copy()
    go[r0:r1 + 1, c0:c1 + 1] = 4
    for g in (gi, go):
        g[r0, c0:c1 + 1] = sqc
        g[r1, c0:c1 + 1] = sqc
        g[r0:r1 + 1, c0] = sqc
        g[r0:r1 + 1, c1] = sqc
    gi[r0, a:b + 1] = bgc
    go[0:r0 + 1, a:b + 1] = 4
    r, c = r0 - 1, b + 1
    while r >= 0 and c < w:
        go[r, c] = 4; r -= 1; c += 1
    r, c = r0 - 1, a - 1
    while r >= 0 and c >= 0:
        go[r, c] = 4; r -= 1; c -= 1
    kk = random.choice((0, 1, 2, 3))
    return {"input": np.rot90(gi, -kk).tolist(), "output": np.rot90(go, -kk).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the colour the demonstrations turn into 4 (background inside the box
    # always becomes 4). Fallback: most common colour.
    bgc = None
    for ex in (examples or []):
        EI = np.asarray(ex[0], dtype=int)
        EO = np.asarray(ex[1], dtype=int)
        if EI.shape != EO.shape:
            continue
        m = (EO == 4) & (EI != 4)
        if m.any():
            bgc = Counter(EI[m].tolist()).most_common(1)[0][0]
            break
    if bgc is None or not (I == bgc).any():
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    pts = np.argwhere(I != bgc)
    r0, c0 = pts.min(0).tolist()
    r1, c1 = pts.max(0).tolist()

    # 1) The box's backdrop minus its outline cells = interior AND the gap, taken as one
    #    region that the light floods.
    delta = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if I[r, c] == bgc]
    ops.append(4); sels.append(sel_of(delta))

    # 2) The gap: the box edge whose line shows background cells.
    edges = [
        ([(r0, c) for c in range(c0, c1 + 1)], (-1, 0), (0, 1)),   # top    -> shines up
        ([(r1, c) for c in range(c0, c1 + 1)], (1, 0), (0, 1)),    # bottom -> shines down
        ([(r, c0) for r in range(r0, r1 + 1)], (0, -1), (1, 0)),   # left   -> shines left
        ([(r, c1) for r in range(r0, r1 + 1)], (0, 1), (1, 0)),    # right  -> shines right
    ]
    hole, d, t, best = None, None, None, 0
    for line, dd, tt in edges:
        hc = [p for p in line if I[p] == bgc]
        if len(hc) > best:
            best, hole, d, t = len(hc), hc, dd, tt

    def ray(sr, sc, dr, dc):
        out = []
        r, c = sr, sc
        while 0 <= r < hi and 0 <= c < wi:
            out.append((r, c)); r += dr; c += dc
        return out

    # 3) Every gap cell shoots its own straight ray outward to the grid border.
    for (r, c) in hole:
        cells = ray(r + d[0], c + d[1], d[0], d[1])
        if cells:
            ops.append(4); sels.append(sel_of(cells))

    # 4) The gap's two ends shoot diagonal rays outward, each ray as one line.
    p1, p2 = hole[0], hole[-1]
    for (pr, pc), s in ((p1, -1), (p2, 1)):
        dr, dc = d[0] + s * t[0], d[1] + s * t[1]
        cells = ray(pr + dr, pc + dc, dr, dc)
        if cells:
            ops.append(4); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
