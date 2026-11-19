"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the "seeds on top row" layout


def sample_colors(num_examples=None) -> dict:
    bgc = random.randrange(10)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    a = lo + int(round(d * (hi - lo) * 0.0))
    lo2 = lo + int(diff_lb * (hi - lo))
    hi2 = lo + int(round(diff_ub * (hi - lo)))
    lo2 = max(lo, min(lo2, hi))
    hi2 = max(lo2, min(hi2, hi))
    return random.randint(lo2, hi2)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # dims in the canonical (seeds on top row) frame; rotation swaps them
    if rot % 2 == 0:
        Hmax, Wmax = max_h, max_w
    else:
        Hmax, Wmax = max_w, max_h
    Hmax = max(3, min(30, Hmax))
    Wmax = max(3, min(30, Wmax))
    h = _unif(diff_lb, diff_ub, 3, Hmax)
    w = _unif(diff_lb, diff_ub, 3, Wmax)
    remcols = [c for c in range(10) if c != bgc]
    nlocs = _unif(diff_lb, diff_ub, 1, max(1, (w - 2) // 3))
    locopts = list(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if x < locj - 2 or x > locj + 2]
        col = random.choice(remcols)
        gi[0, locj] = col
        for p in range(0, h, 2):
            go[p, locj] = col
        for p in range(1, h, 2):
            go[p, locj - 1] = col
            go[p, locj + 1] = col
    # rot90 (CW) applied `rot` times
    gi = np.rot90(gi, k=-rot)
    go = np.rot90(go, k=-rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (lane carving): each seed on a border owns three parallel lanes running
    straight into the grid. Its OWN lane is coloured at every even depth (0,2,4,...);
    each of the two NEIGHBOUR lanes is coloured at every odd depth (1,3,5,...).
    So per seed: one Color op paints the centre lane's even-depth cells, then one
    Color op per side lane paints that lane's odd-depth cells. Every selection is
    "every other cell of a line through the seed", named from I alone. Color ops
    paint colour 0 just as well, so seeds of colour 0 need no special case.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    ops, sels = [], []
    if seeds:
        if all(r == 0 for r, _ in seeds):
            u, v, D = (1, 0), (0, 1), hi
        elif all(r == hi - 1 for r, _ in seeds):
            u, v, D = (-1, 0), (0, 1), hi
        elif all(c == 0 for _, c in seeds):
            u, v, D = (0, 1), (1, 0), wi
        else:
            u, v, D = (0, -1), (1, 0), wi

        def cell(s, d, t):
            return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

        def inside(p):
            return 0 <= p[0] < hi and 0 <= p[1] < wi

        for s in seeds:
            col = int(I[s[0], s[1]])
            # centre lane: every even depth (seed itself already coloured)
            centre = [cell(s, d, 0) for d in range(0, D, 2)]
            if len(centre) > 1:
                ops.append(col); sels.append(sel_of(centre))
            # the two neighbour lanes: every odd depth
            for t in (-1, 1):
                lane = [cell(s, d, t) for d in range(1, D, 2)]
                lane = [p for p in lane if inside(p)]
                if lane:
                    ops.append(col); sels.append(sel_of(lane))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
