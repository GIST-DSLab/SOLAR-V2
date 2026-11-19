"""
ARC Task: 28e73c20 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    ncols = random.randint(1, 9)
    return {"ccols": random.sample(cols, ncols)}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kw) -> dict:
    if not ccols:
        ccols = random.sample([c for c in range(10) if c != 3], random.randint(1, 9))
    dm = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}

    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    h = unif(3, min(30, max_h))
    w = unif(4, min(30, max_w + 1))          # internal width; column 0 is cropped
    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    go = [row[:] for row in gi]
    for c in range(w):
        go[0][c] = 3
    sp = (0, w - 1)
    direc = (1, 0)
    lw, lh, ld = w, h, h
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        for k in range(ld):
            go[sp[0] + direc[0] * k][sp[1] + direc[1] * k] = 3
        ep = (sp[0] + direc[0] * (ld - 1), sp[1] + direc[1] * (ld - 1))
        direc = dm[direc]
        ld = lh if isverti else lw
        isverti = not isverti
        sp = ep
    gi = [row[1:] for row in gi]
    go = [row[1:] for row in go]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading: concentric rectangular frames at every other depth (0, 2, 4, ...)
    are the walls of the spiral. Each wall is drawn whole, then cut open at its
    seam (restoring the input colour there), and a connector bridges through the
    gap frame just inside into the next wall -- turning nested rings into one
    spiral. Everything is determined by I's shape alone; O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # spiral path implied by the shape (generator walk; internal col 0 cropped)
    H, W = hi, wi + 1
    dm = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    S = {(0, c) for c in range(W)}
    sp, direc = (0, W - 1), (1, 0)
    lw, lh, ld, iv = W, H, H, False
    while ld > 0:
        lw -= 1
        lh -= 1
        for k in range(ld):
            S.add((sp[0] + direc[0] * k, sp[1] + direc[1] * k))
        ep = (sp[0] + direc[0] * (ld - 1), sp[1] + direc[1] * (ld - 1))
        direc = dm[direc]
        ld = lh if iv else lw
        iv = not iv
        sp = ep
    S = {(r, c - 1) for r, c in S if c >= 1}

    def ring(d):
        r0, c0, r1, c1 = d, d, hi - 1 - d, wi - 1 - d
        if r0 > r1 or c0 > c1:
            return set()
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)}

    def depth(r, c):
        return min(r, c, hi - 1 - r, wi - 1 - c)

    rings = {}
    d = 0
    while True:
        rc = ring(d)
        if not rc:
            break
        rings[d] = rc
        d += 2
    allR = set().union(*rings.values()) if rings else set()

    ops, sels = [], []
    for d in sorted(rings):
        # 1. draw the whole wall frame at depth d
        ops.append(3)
        sels.append(sel_of(sorted(rings[d])))
        # 2. open its seam: restore the input colour there
        gap = sorted(rings[d] - S)
        bycol = {}
        for (r, c) in gap:
            bycol.setdefault(int(I[r, c]), []).append((r, c))
        for col in sorted(bycol):
            ops.append(col)
            sels.append(sel_of(bycol[col]))
        # 3. bridge into the next wall through the gap frame at depth d+1
        conn = sorted(x for x in S - allR if depth(*x) == d + 1)
        if conn:
            ops.append(3)
            sels.append(sel_of(conn))

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
                        f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
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
                                f"for task 28e73c20"
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
                    f"Failed to build a complete episode for task 28e73c20 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"28e73c20-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
