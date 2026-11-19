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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, boxc, fillc = random.sample(cols, 3)
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + int(round((hi - lo) * max(0.0, min(1.0, diff_lb))))
    b = lo + int(round((hi - lo) * max(0.0, min(1.0, diff_ub))))
    a, b = min(a, b), max(a, b)
    a = max(lo, min(hi, a))
    b = max(lo, min(hi, b))
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc) -> dict:
    h = _unifint(diff_lb, diff_ub, 5, max(5, max_h))
    w = _unifint(diff_lb, diff_ub, 5, max(5, max_w))
    oh = _unifint(diff_lb, diff_ub, 3, h - 1)
    ow = _unifint(diff_lb, diff_ub, 3, w - 1)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    subg = [[boxc] * ow for _ in range(oh)]
    for i in range(oh - 1):
        for j in range(1, ow - 1):
            subg[i][j] = fillc
    ntofill = _unifint(diff_lb, diff_ub, 1, min(9, oh - 2))
    for i in range(ntofill):
        for j in range(1, ow - 1):
            subg[i][j] = bgc
    gi = [[bgc] * w for _ in range(h)]
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = subg[i][j]
    go = [fillc] * ntofill + [bgc] * (9 - ntofill)
    go = [go[:3], go[3:6][::-1], go[6:]]
    k = random.choice([0, 1, 2, 3])
    for _ in range(k):
        gi = [list(r) for r in zip(*gi[::-1])]  # rot90 CW
    return {"input": [list(r) for r in gi], "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = sorted(set(I.flatten().tolist()))

    def bbox(c):
        rs, cs = np.where(I == c)
        return rs.min(), cs.min(), rs.max(), cs.max()

    # background = the colour whose cells span the whole grid (margin row+col always exist)
    bgc = None
    for c in colors:
        r0, c0, r1, c1 = bbox(c)
        if r0 == 0 and c0 == 0 and r1 == hi - 1 and c1 == wi - 1:
            bgc = c
            break
    others = [c for c in colors if c != bgc]
    areas = {}
    for c in others:
        r0, c0, r1, c1 = bbox(c)
        areas[c] = (r1 - r0 + 1) * (c1 - c0 + 1)
    boxc = max(others, key=lambda c: areas[c])
    fillc = [c for c in others if c != boxc][0]

    br0, bc0, br1, bc1 = bbox(boxc)
    fr0, fc0, fr1, fc1 = bbox(fillc)
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    fh, fw = fr1 - fr0 + 1, fc1 - fc0 + 1
    top_full = bool(np.all(I[br0, bc0:bc1 + 1] == boxc))
    bot_full = bool(np.all(I[br1, bc0:bc1 + 1] == boxc))
    if not top_full or not bot_full:
        n = (bh - 1) - fh   # opening on top/bottom: count empty rows
    else:
        n = (bw - 1) - fw   # opening on left/right: count empty columns

    ops, sels = [], []
    # Crop canvas to 3x3 (top-left region of the input)
    ops.append(33); sels.append([0, 0, 2, 2])
    cur = I[0:3, 0:3].copy()
    cur[cur == 0] = 0
    # Lay a background base on cells not already bgc
    base = [(r, c) for r in range(3) for c in range(3) if cur[r, c] != bgc]
    if base:
        ops.append(bgc); sels.append(sel_of(base))
        for r, c in base:
            cur[r, c] = bgc
    # Fill n cells in snake order with the fill colour
    cells = []
    for k in range(n):
        r = k // 3
        c = (2 - k % 3) if r == 1 else (k % 3)
        cells.append((r, c))
    cells = [(r, c) for r, c in cells if cur[r, c] != fillc]
    if cells:
        ops.append(fillc); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, 2, 2])
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
