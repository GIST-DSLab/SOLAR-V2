"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, dotc: int) -> dict:
    cols = interval(0, 10, 1)
    hcap = max(2, min(4, max_h // 2))
    wcap = max(2, min(4, max_w // 2))
    h = unifint(diff_lb, diff_ub, (2, hcap))
    w = unifint(diff_lb, diff_ub, (2, wcap))
    remcols = remove(bgc, cols)
    remcols = remove(dotc, remcols)
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)

    lo_h = min(min(h, w) * 2, max_h)
    lo_w = min(min(h, w) * 2, max_w)
    gih = unifint(diff_lb, diff_ub, (lo_h, max_h))
    giw = unifint(diff_lb, diff_ub, (lo_w, max_w))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    maxtrtot = 1000
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            break
        maxtrtot += 1
        if maxtrtot < 1000:
            break
        maxtr = int(maxtr * 1.5)
        gih = randint(min(gih, max_h), max_h)
        giw = randint(min(giw, max_w), max_w)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    I holds several scattered fragments; every fragment carries one copy of the
    same marker colour (dotc).  Overlaying all fragments on that marker rebuilds
    the small output grid.  So: crop the canvas onto the frame where one
    fragment already sits correctly, then paint in the remaining fragments'
    cells (one Color op per colour).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    cnt_I = Counter(I.flatten().tolist())
    cnt_O = Counter(O.flatten().tolist())
    bgc = cnt_I.most_common(1)[0][0]

    # marker colour: appears once per fragment in I but only once in O
    diffs = {c: cnt_I.get(c, 0) - cnt_O[c] for c in cnt_O}
    best = max(diffs.values()) if diffs else 0
    if best > 0:
        dotc = sorted([c for c in diffs if diffs[c] == best])[0]
    else:
        singles = sorted([c for c in cnt_O if cnt_O[c] == 1])
        dotc = singles[0] if singles else sorted(cnt_O)[0]

    # marker position inside O
    dp = [(r, c) for r in range(ho) for c in range(wo) if O[r, c] == dotc]
    pr, pc = dp[0] if dp else (0, 0)

    dot_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]

    # for each fragment colour, align its nearest marker cell onto (pr, pc)
    frames = []
    for c in sorted(cnt_O):
        if c == dotc:
            continue
        cells = [(r, cc) for r in range(hi) for cc in range(wi) if I[r, cc] == c]
        if not cells or not dot_cells:
            continue
        d = min(dot_cells,
                key=lambda p: min(abs(p[0] - r) + abs(p[1] - cc) for r, cc in cells))
        R, C = d[0] - pr, d[1] - pc
        if 0 <= R and R + ho <= hi and 0 <= C and C + wo <= wi:
            frames.append((len(cells), R, C))

    if frames:
        frames.sort(reverse=True)
        R, C = frames[0][1], frames[0][2]
    else:
        R, C = 0, 0
    R = max(0, min(R, hi - ho))
    C = max(0, min(C, wi - wo))

    ops, sels = [], []
    # 1. crop canvas onto the output frame (the anchor fragment is already right)
    ops.append(33); sels.append([R, C, ho - 1, wo - 1])

    cropped = I[R:R + ho, C:C + wo]
    # 2. paint the remaining fragments, one op per colour
    for col in sorted(cnt_O):
        cells = [(r, c) for r in range(ho) for c in range(wo)
                 if O[r, c] == col and cropped[r, c] != col]
        if cells:
            ops.append(int(col)); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
