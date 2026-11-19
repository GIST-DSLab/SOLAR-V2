"""
ARC Task: b0c4d837 (RE-ARC) — LLM-generated grid_maker
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
    bgc, boxc, fillc = random.sample(range(10), 3)
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc}


def _rot90(g):
    # clockwise
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxc=None, fillc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None or boxc is None or fillc is None:
        bgc, boxc, fillc = random.sample(range(10), 3)
    max_h = max(5, min(30, max_h))
    max_w = max(5, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    subg = [[boxc] * ow for _ in range(oh)]
    subg2 = [[fillc] * (ow - 2) for _ in range(oh - 1)]
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        for c in range(ow - 2):
            subg2[j][c] = bgc
    for r in range(oh - 1):
        for c in range(ow - 2):
            subg[r][c + 1] = subg2[r][c]
    gi = [[bgc] * w for _ in range(h)]
    for r in range(oh):
        for c in range(ow):
            gi[loci + r][locj + c] = subg[r][c]
    go = [fillc] * ntofill + [bgc] * (9 - ntofill)
    go = [go[:3], go[3:6][::-1], go[6:]]
    k = random.randint(0, 3)
    for _ in range(k):
        gi = _rot90(gi)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by maker and golf entry): the count n is the number of
    empty background rows inside the open-ended box (= box depth - fill depth - 1).
    The 3x3 answer is a tally: fillc laid n times along a snake path
    (row0 L->R, row1 R->L, row2 L->R), background elsewhere.
    The golf entry treats the answer as a fixed template whose cells switch on
    with the count; here each row of the template is one unit: lay the
    background base on the 3x3 canvas, then paint each row's tally segment
    as one strip.  (Colour ops, so colour 0 needs no special case.)
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # colour roles by bounding-box area (like the verifier's partition)
    info = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        info.append((int((r1 - r0 + 1) * (c1 - c0 + 1)), int(col),
                     (int(r0), int(c0), int(r1), int(c1))))
    info.sort(key=lambda t: -t[0])
    bgc = info[0][1]
    boxc, bbox = info[1][1], info[1][2]
    fillc, fbox = info[-1][1], info[-1][2]

    r0, c0, r1, c1 = bbox
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    if left_full and right_full:
        n = (r1 - r0 + 1) - (fbox[2] - fbox[0] + 1) - 1
    else:
        n = (c1 - c0 + 1) - (fbox[3] - fbox[1] + 1) - 1
    n = max(0, min(9, n))

    ops, sels = [], []
    out_cells = [(r, c) for r in range(3) for c in range(3)]

    # 1. reframe canvas to the 3x3 answer (full rectangle)
    ops.append(33); sels.append(sel_of(out_cells))
    cur = I[0:3, 0:3].copy()

    # 2. lay the background base over the whole 3x3 (skip if already so)
    if not np.all(cur == bgc):
        ops.append(bgc); sels.append(sel_of(out_cells))
        cur[:, :] = bgc

    # 3. tally: one strip per row of the snake path
    segs = []
    k0 = min(n, 3)
    if k0 > 0:
        segs.append([(0, c) for c in range(k0)])
    k1 = min(max(n - 3, 0), 3)
    if k1 > 0:
        segs.append([(1, 2 - i) for i in range(k1)])
    k2 = min(max(n - 6, 0), 3)
    if k2 > 0:
        segs.append([(2, c) for c in range(k2)])
    for seg in segs:
        if any(cur[r, c] != fillc for r, c in seg):
            ops.append(fillc); sels.append(sel_of(seg))
            for r, c in seg:
                cur[r, c] = fillc

    ops.append(34); sels.append(sel_of(out_cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
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
                                f"for task b0c4d837"
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
                    f"Failed to build a complete episode for task b0c4d837 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b0c4d837-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
