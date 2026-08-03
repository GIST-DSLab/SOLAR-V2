"""
ARC Task: a9f96cdd (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # generator samples bgc and fgc from {0..9} minus the four mark colors {3,6,7,8}
    cols = [c for c in range(10) if c not in (3, 6, 7, 8)]
    bgc = choice(cols)
    fgc = choice(remove(bgc, cols))
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for k in range(noccs):
        if len(locs) == 0:
            break
        loc = choice(totuple(locs))
        locs = locs - mapply(neighbors, neighbors(loc))
        plcd = {loc}
        gi = fill(gi, fgc, plcd)
        go = fill(go, 3, shift(plcd, (-1, -1)))
        go = fill(go, 7, shift(plcd, (1, 1)))
        go = fill(go, 8, shift(plcd, (1, -1)))
        go = fill(go, 6, shift(plcd, (-1, 1)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule read off I/O: I is a plain background with isolated single-cell dots of one
    rare color.  Each dot vanishes (back to background) and its four DIAGONAL
    neighbours get marked.  Which colour goes on which diagonal is MEASURED from the
    I/O pair (never assumed), then applied dot by dot.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- roles from I alone -------------------------------------------------
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]                       # canvas colour
    others = [c for c in cnt if c != bgc]
    if not others:
        return [34], [[0, 0, hi - 1, wi - 1]]
    fgc = min(others, key=lambda c: cnt[c])              # rarest colour = the dots

    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == fgc]

    # --- measure the diagonal-offset -> mark-colour correspondence from I/O --
    DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    votes = {d: Counter() for d in DIAGS}
    for (r, c) in dots:
        for d in DIAGS:
            rr, cc = r + d[0], c + d[1]
            if 0 <= rr < hi and 0 <= cc < wi:
                votes[d][int(O[rr, cc])] += 1
    mark = {d: votes[d].most_common(1)[0][0] for d in DIAGS if votes[d]}

    # --- apply the measured rule, one whole dot at a time --------------------
    G = I.copy()
    ops, sels = [], []
    for (r, c) in dots:
        for d in DIAGS:                                  # the dot's four arms
            if d not in mark:
                continue
            rr, cc = r + d[0], c + d[1]
            if not (0 <= rr < hi and 0 <= cc < wi):
                continue                                 # arm falls off the canvas
            col = mark[d]
            if G[rr, cc] != col:
                ops.append(int(col))
                sels.append([rr, cc, 0, 0])
                G[rr, cc] = col
        if G[r, c] != bgc:                               # the dot itself is consumed
            ops.append(int(bgc))
            sels.append([r, c, 0, 0])
            G[r, c] = bgc

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task a9f96cdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a9f96cdd"
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
                                f"for task a9f96cdd"
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
                    f"Failed to build a complete episode for task a9f96cdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a9f96cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
