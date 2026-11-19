"""
ARC Task: 5c0a986e (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
import numpy as np
from maker.sel_helpers import sel_of

try:
    from dsl import *  # noqa
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    # 1 and 2 are hardcoded rule colours; only the background is sampled.
    cols = [c for c in range(10) if c not in (1, 2)]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c not in (1, 2)])
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 10)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    inds = asindices(gi)
    fullinds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: 0 < ij[0] <= h - 3 and 0 < ij[1] <= w - 3)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        col = choice((1, 2))
        sq = {(loc), add(loc, (0, 1)), add(loc, (1, 0)), add(loc, (1, 1))}
        if col == 1:
            obj = sq | (shoot(loc, (-1, -1)) & fullinds)
        else:
            obj = sq | (shoot(loc, (1, 1)) & fullinds)
        if obj.issubset(inds):
            succ += 1
            inds = (inds - obj) - mapply(dneighbors, sq)
            gi = fill(gi, col, sq)
            go = fill(go, col, obj)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: every 2x2 square grows a diagonal line from its corner, one
    # cell per step -- 2-squares toward the lower right and 1-squares toward
    # the upper left. All squares grow at the same time. A background cell
    # gets colour x when the next two cells back along the diagonal are
    # already x. Each wave is one Color op per colour, applied to every
    # line's current tip. A tip is chosen by looking at the grid as it is
    # now: the two cells behind it are x and the cell itself is in bounds.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    others = [v for v in I.flatten().tolist() if v not in (1, 2)]
    bgc = max(set(others), key=others.count) if others else 0

    # Find the squares (4-connected components of colour 1 or 2).
    visited = np.zeros((hi, wi), dtype=bool)
    heads = []  # [colour, (r, c) of the line's current end, (dr, dc)]
    for r in range(hi):
        for c in range(wi):
            col = int(I[r, c])
            if col not in (1, 2) or visited[r, c]:
                continue
            comp = []
            dq = deque([(r, c)])
            visited[r, c] = True
            while dq:
                cr, cc = dq.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not visited[nr, nc] and I[nr, nc] == col:
                        visited[nr, nc] = True
                        dq.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if col == 2:
                heads.append([2, (max(rs), max(cs)), (1, 1)])
            else:
                heads.append([1, (min(rs), min(cs)), (-1, -1)])

    G = I.copy()
    ops, sels = [], []
    while True:
        grew = False
        for col in (2, 1):
            tips = []
            for hd in heads:
                if hd[0] != col:
                    continue
                (r, c), (dr, dc) = hd[1], hd[2]
                nr, nc = r + dr, c + dc
                pr, pc = r - dr, c - dc
                if (0 <= nr < hi and 0 <= nc < wi and G[nr, nc] == bgc
                        and G[r, c] == col and G[pr, pc] == col):
                    tips.append((nr, nc))
                    hd[1] = (nr, nc)
            if tips:
                for (r, c) in tips:
                    G[r, c] = col
                ops.append(col)
                sels.append(sel_of(tips))
                grew = True
        if not grew:
            break

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
                        f"num_examples+1 ({num_examples + 1}) for task 5c0a986e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5c0a986e"
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
                                f"for task 5c0a986e"
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
                    f"Failed to build a complete episode for task 5c0a986e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5c0a986e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
