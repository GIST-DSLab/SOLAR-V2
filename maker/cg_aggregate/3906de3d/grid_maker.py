"""
ARC Task: 3906de3d (RE-ARC) — LLM-generated grid_maker
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

# The four orientations the generator can produce (k = number of CW quarter-turns)
ROTATIONS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, linc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        examples = [{"rot": r} for r in ROTATIONS]
        examples += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _rot_cw(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTATIONS)
    # an odd number of quarter-turns swaps the axes
    hb, wb = (max_w, max_h) if rot % 2 else (max_h, max_w)
    h = _unif(diff_lb, diff_ub, (5, max(5, min(30, hb))))
    w = _unif(diff_lb, diff_ub, (5, max(5, min(30, wb))))
    oh = _unif(diff_lb, diff_ub, (2, h // 2))
    ow = _unif(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)
    gi = [[bgc] * w for _ in range(h)]
    for r in range(oh):
        for c in range(locj, locj + ow):
            gi[r][c] = boxc
    rng = list(range(locj, locj + ow))
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):
        for r in range(co, oh):
            gi[r][jj] = bgc
    numlns = _unif(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = random.sample(rng, numlns)
    go = [row[:] for row in gi]
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            for r in range(h - lineh, h):
                gi[r][jj] = linc
            for r in range(co, co + lineh):
                go[r][jj] = linc
    for _ in range(rot):
        gi = _rot_cw(gi)
        go = _rot_cw(go)
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """Each 1-wide line slides along its lane toward the bar block on the opposite
    edge until it sits against the end of the bar in that lane.

    The golf reading (per-lane stable 'gravity': non-background cells compact toward
    the bar edge while order is kept) is the same as the maker's here. In each lane
    the bar is a run that starts at the edge, and the line is one contiguous run, so
    compacting the lane is exactly moving the line as one unit up to the bar. Lanes
    stop at different distances, so each line gets its own grab and slide. There is
    nothing to change from the maker's route."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # In the un-rotated layout the side columns are pure background,
    # so a uniform first row means the lanes run horizontally.
    horizontal = len(set(I[0].tolist())) == 1

    if horizontal:
        e0, e1 = I[:, 0].tolist(), I[:, -1].tolist()
    else:
        e0, e1 = I[0].tolist(), I[-1].tolist()

    common = set(e0) & set(e1)
    if len(common) == 1:
        bgc = next(iter(common))
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    n0 = sum(1 for v in e0 if v != bgc)
    n1 = sum(1 for v in e1 if v != bgc)
    box_first = n0 >= n1          # the bar edge holds more non-background cells

    box_edge = e0 if box_first else e1
    lin_edge = e1 if box_first else e0
    bc = [v for v in box_edge if v != bgc]
    lc = [v for v in lin_edge if v != bgc]
    if not bc or not lc:
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels
    boxc = Counter(bc).most_common(1)[0][0]
    linc = Counter(lc).most_common(1)[0][0]

    if horizontal:
        move_op = 23 if box_first else 22      # MoveL / MoveR
        lanes = range(h)
        L = w
    else:
        move_op = 20 if box_first else 21      # MoveU / MoveD
        lanes = range(w)
        L = h

    def lane_vec(k):
        v = I[k, :].tolist() if horizontal else I[:, k].tolist()
        return v if box_first else v[::-1]

    def coord(t, k):
        if horizontal:
            return (k, t) if box_first else (k, w - 1 - t)
        return (t, k) if box_first else (h - 1 - t, k)

    for k in lanes:
        vec = lane_vec(k)
        co = 0
        while co < L and vec[co] == boxc:      # length of the bar in this lane
            co += 1
        idxs = [t for t, v in enumerate(vec) if v == linc]
        if not idxs:
            continue
        steps = min(idxs) - co                 # slide until it touches the bar
        if steps <= 0:
            continue

        src = [coord(t, k) for t in idxs]
        dst = [coord(t - steps, k) for t in idxs]
        hole = sorted(set(src) - set(dst))

        if linc != 0:
            ops.append(move_op); sels.append(sel_of(src))        # grab the line
            for _ in range(steps - 1):
                ops.append(move_op); sels.append(sel_of([]))     # keep sliding it
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))       # repair start footprint
        else:
            # colour 0 is "nothing" to ARCLE object ops: paint the line at its new place
            ops.append(0); sels.append(sel_of(dst))
            if hole:
                ops.append(bgc); sels.append(sel_of(hole))

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
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
                                f"for task 3906de3d"
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
                    f"Failed to build a complete episode for task 3906de3d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3906de3d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
