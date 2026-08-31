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


# ---------------------------------------------------------------- colors ----
ROTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxc = random.choice([c for c in cols if c != bgc])
    # keep the moving line non-zero so it is a real ARCLE "object"
    linc_pool = [c for c in cols if c not in (bgc, boxc) and c != 0]
    linc = random.choice(linc_pool)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [dict(v) for v in ROTS]
        examples += [dict(random.choice(ROTS)) for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


# -------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    if rot % 2 == 1:              # a 90/270 rotation swaps the dimensions
        hmax, wmax = max(5, max_w), max(5, max_h)
    else:
        hmax, wmax = max(5, max_h), max(5, max_w)

    h = unifint(diff_lb, diff_ub, (5, hmax))
    w = unifint(diff_lb, diff_ub, (5, wmax))
    oh = unifint(diff_lb, diff_ub, (2, h // 2))
    ow = unifint(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)

    bx = backdrop(frozenset({(0, locj), (oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, boxc, bx)
    rng = range(locj, locj + ow)
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):
        gi = fill(gi, bgc, connect((co, jj), (oh - 1, jj)))

    numlns = unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = random.sample(list(rng), numlns)
    go = tuple(e for e in gi)
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            linei = connect((h - lineh, jj), (h - 1, jj))
            lineo = connect((co, jj), (co + lineh - 1, jj))
            gi = fill(gi, linc, linei)
            go = fill(go, linc, lineo)

    rotf = [identity, rot90, rot180, rot270][rot]
    return {"input": rotf(gi), "output": rotf(go)}


# ------------------------------------------------------- derive_operations ---
def derive_operations(I, O):
    """Each 1-wide line hanging off one edge slides toward the bar-chart block
    on the opposite edge until it touches the end of the bar in its own lane."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- orientation: in the un-rotated layout the two side columns are pure
    #     background, so a uniform first ROW means the motion axis is horizontal
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
    box_first = n0 >= n1                      # block edge holds more non-bg cells

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
        steps = min(idxs) - co                 # slide until adjacent to the bar
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
                ops.append(bgc); sels.append(sel_of(hole))       # repair footprint
        else:
            # colour 0 is "nothing" to ARCLE's object ops: paint the line instead
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
