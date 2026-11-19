"""
ARC Task: a61ba2ce (RE-ARC) — LLM-generated grid_maker
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
import itertools
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4) -> dict:
    from utils import unifint
    from dsl import connect, canvas, fill, asindices, sfilter, shift, totuple
    mh = max(4, min(15, max_h // 2))
    mw = max(4, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (4, mh))
    w = unifint(diff_lb, diff_ub, (4, mw))
    lociL = random.randint(2, h - 2)
    lociR = random.randint(2, h - 2)
    locjT = random.randint(2, w - 2)
    locjB = random.randint(2, w - 2)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(diff_lb, diff_ub, (2 * h, max(2 * h, max_h)))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max(2 * w, max_w)))
    gi = canvas(bgc, (fullh, fullw))
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    while True:
        inds = asindices(gi)
        locs = []
        for o, c in zip(objs, ocols):
            cands = sfilter(inds, lambda ij: shift(o, ij).issubset(inds))
            if len(cands) == 0:
                break
            loc = random.choice(totuple(cands))
            locs.append(loc)
            inds = inds - shift(o, loc)
        if len(locs) == 4:
            break
    for o, c, l in zip(objs, ocols, locs):
        gi = fill(gi, c, shift(o, l))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])

    # one L-shaped corner piece per non-background colour
    pieces = {}
    for col in vals:
        col = int(col)
        if col == bgc:
            continue
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        cs = set(cells)
        # the bbox corner the L does NOT occupy tells which frame corner it is
        if (r1, c1) not in cs:
            kind = "UL"
        elif (r1, c0) not in cs:
            kind = "UR"
        elif (r0, c1) not in cs:
            kind = "LL"
        else:
            kind = "LR"
        pieces[kind] = dict(col=col, cells=cells, r0=r0, c0=c0,
                            h=r1 - r0 + 1, w=c1 - c0 + 1)

    ho = pieces["UL"]["h"] + pieces["LL"]["h"]
    wo = pieces["UL"]["w"] + pieces["UR"]["w"]
    rel = {"UL": (0, 0),
           "UR": (0, wo - pieces["UR"]["w"]),
           "LL": (ho - pieces["LL"]["h"], 0),
           "LR": (ho - pieces["LR"]["h"], wo - pieces["LR"]["w"])}
    kinds = ["UL", "UR", "LL", "LR"]
    orig = {k: set(pieces[k]["cells"]) for k in kinds}

    def plan_at(R0, C0):
        dest, disp = {}, {}
        for k in kinds:
            p = pieces[k]
            dr = R0 + rel[k][0] - p["r0"]
            dc = C0 + rel[k][1] - p["c0"]
            disp[k] = (dr, dc)
            dest[k] = {(r + dr, c + dc) for r, c in p["cells"]}
        return dest, disp

    # choose where to assemble the frame and the order of moves so that no
    # piece lands on a piece that has not been moved yet
    best = None
    for R0 in range(hi - ho + 1):
        for C0 in range(wi - wo + 1):
            dest, disp = plan_at(R0, C0)
            cost = sum(abs(a) + abs(b) for a, b in disp.values())
            if best is not None and cost >= best[0]:
                continue
            for order in itertools.permutations(kinds):
                ok = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        if dest[order[i]] & orig[order[j]]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best = (cost, R0, C0, order, disp, dest)
                    break
    if best is None:
        dest, disp = plan_at(0, 0)
        best = (0, 0, 0, tuple(kinds), disp, dest)
    cost, R0, C0, order, disp, dest = best

    ops, sels = [], []
    zeroed = set()      # cells currently 0 because a piece left them
    for k in order:
        p = pieces[k]
        dr, dc = disp[k]
        if dr == 0 and dc == 0:
            continue
        if p["col"] == 0:
            # colour 0 cannot be grabbed by Move: draw it at its corner
            todo = [x for x in dest[k] if x not in orig[k]]
            ops.append(0); sels.append(sel_of(todo))
            zeroed |= orig[k] - dest[k]
            zeroed -= dest[k]
            continue
        cur = list(p["cells"])
        first = True
        steps = [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr) + \
                [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
        for op, (sr, sc) in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))  # grab once, then continue
            first = False
            cur = [(r + sr, c + sc) for r, c in cur]
        if bgc != 0:
            zeroed |= orig[k] - dest[k]
        zeroed -= dest[k]

    # repair vacated (0) cells that lie inside the frame region; outside it the crop drops them
    if bgc != 0:
        hole = [(r, c) for r, c in zeroed
                if R0 <= r < R0 + ho and C0 <= c < C0 + wo]
        if hole:
            ops.append(bgc); sels.append(sel_of(hole))

    # crop to the assembled frame (whole rectangle intended)
    if not (R0 == 0 and C0 == 0 and ho == hi and wo == wi):
        ops.append(33); sels.append([R0, C0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
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
                                f"for task a61ba2ce"
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
                    f"Failed to build a complete episode for task a61ba2ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a61ba2ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
