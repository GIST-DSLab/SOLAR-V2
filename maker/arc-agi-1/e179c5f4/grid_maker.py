"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice([c for c in cols if c != 0])
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, linc: int) -> dict:
    # both orientations can occur after the mirror step -> bound by the tighter limit
    lim = min(max_h, max_w)
    w = unifint(diff_lb, diff_ub, (2, min(10, lim - 1)))
    h = unifint(diff_lb, diff_ub, (w + 1, min(30, lim)))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    go = replace(go, bgc, 8)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # --- read the rule off I -------------------------------------------------
    # I is a plain canvas with marker pixels sitting on 1..4 of its corners.
    corners = {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}
    bgc = None
    seed = None
    for r in range(H):
        for c in range(W):
            if (r, c) not in corners:
                bgc = int(I[r, c])
                seed = (r, c)
                break
        if seed is not None:
            break
    linc = None
    for (r, c) in corners:
        if int(I[r, c]) != bgc:
            linc = int(I[r, c])
            break

    ops, sels = [], []

    # 1. the canvas turns into 8; the corner markers survive.
    #    one FloodFill8 on the single connected background region does exactly that.
    ops.append(18)
    sels.append([seed[0], seed[1], 0, 0])

    # 2. every marker shoots a 45-degree ray that bounces between the two long
    #    walls and runs the whole length of the grid.
    portrait = H > W
    Hn, Wn = (H, W) if portrait else (W, H)     # long axis first
    P = 2 * Wn - 2                              # bounce period across the short axis

    def to_actual(r, c):
        return (r, c) if portrait else (c, r)

    def fold(k):
        m = k % P
        return m if m <= Wn - 1 else P - m

    ray_of = [
        ((0, 0),             lambda k: (k, fold(k))),
        ((0, Wn - 1),        lambda k: (k, Wn - 1 - fold(k))),
        ((Hn - 1, 0),        lambda k: (Hn - 1 - k, fold(k))),
        ((Hn - 1, Wn - 1),   lambda k: (Hn - 1 - k, Wn - 1 - fold(k))),
    ]

    painted = {(r, c) for r in range(H) for c in range(W) if int(I[r, c]) == linc}

    for start, path in ray_of:
        ar, ac = to_actual(*start)
        if int(I[ar, ac]) != linc:
            continue
        # walk this ray from its own marker outward, one line at a time
        for k in range(Hn):
            cell = to_actual(*path(k))
            if cell in painted:
                continue
            painted.add(cell)
            ops.append(linc)
            sels.append([cell[0], cell[1], 0, 0])

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
