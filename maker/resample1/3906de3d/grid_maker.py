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
from maker.sel_helpers import sel_of

ROTS = ["id", "r90", "r180", "r270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    boxc = random.choice([c for c in cols if c != bgc])
    # line colour is non-zero so the sliding lines can be expressed with ARCLE Move
    linc = random.choice([c for c in cols if c not in (bgc, boxc, 0)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    if rot in ("r90", "r270"):
        mh, mw = max_w, max_h
    else:
        mh, mw = max_h, max_w
    h = unifint(diff_lb, diff_ub, (5, max(5, mh)))
    w = unifint(diff_lb, diff_ub, (5, max(5, mw)))
    oh = unifint(diff_lb, diff_ub, (2, h // 2))
    ow = unifint(diff_lb, diff_ub, (3, w - 2))
    locj = random.randint(1, w - ow - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[0:oh, locj:locj + ow] = boxc
    rng = list(range(locj, locj + ow))
    cutoffs = [random.randint(1, oh - 1) for _ in rng]
    for jj, co in zip(rng, cutoffs):
        gi[co:oh, jj] = bgc
    numlns = unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = random.sample(rng, numlns)
    go = gi.copy()
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = random.randint(1, h - co - 1)
            gi[h - lineh:h, jj] = linc
            go[co:co + lineh, jj] = linc
    k = {"id": 0, "r90": 3, "r180": 2, "r270": 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = int(I[0, 0])  # corners are always background
    edges = {
        "top": [(0, c) for c in range(w)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
        "right": [(r, w - 1) for r in range(h)],
    }
    counts = {k: sum(1 for (r, c) in v if I[r, c] != bgc) for k, v in edges.items()}
    box_edge = max(counts, key=lambda k: counts[k])
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    line_edge = opposite[box_edge]
    d, mop = {"top": ((-1, 0), 20), "bottom": ((1, 0), 21),
              "left": ((0, -1), 23), "right": ((0, 1), 22)}[box_edge]
    dr, dc = d
    ops, sels = [], []
    for (r0, c0) in edges[line_edge]:
        if I[r0, c0] == bgc:
            continue
        linc = int(I[r0, c0])
        cells = []
        r, c = r0, c0
        while 0 <= r < h and 0 <= c < w and I[r, c] == linc:
            cells.append((r, c))
            r += dr; c += dc
        k = 0
        while 0 <= r < h and 0 <= c < w and I[r, c] == bgc:
            k += 1
            r += dr; c += dc
        if k == 0:
            continue
        dst = [(rr + dr * k, cc + dc * k) for rr, cc in cells]
        if linc != 0:
            ops.append(mop); sels.append(sel_of(cells))   # grab the line
            for _ in range(k - 1):
                ops.append(mop); sels.append(sel_of([]))  # keep sliding same line
            hole = sorted(set(cells) - set(dst))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
        else:
            # 0-coloured object cannot be grabbed by Move; repaint instead
            vac = sorted(set(cells) - set(dst))
            new = sorted(set(dst) - set(cells))
            if vac:
                ops.append(bgc); sels.append(sel_of(vac))
            if new:
                ops.append(0); sels.append(sel_of(new))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
