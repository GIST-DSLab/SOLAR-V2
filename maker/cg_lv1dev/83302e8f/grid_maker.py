"""
ARC Task: 83302e8f (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c not in (3, 4)]
        bgc, linc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (3, max(3, max_h // (h + 1))))
    nw = unifint(diff_lb, diff_ub, (3, max(3, max_w // (w + 1))))
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h + 1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w + 1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    ofc = ofcolor(gi, linc)
    dots = sfilter(ofc, lambda ij: dneighbors(ij).issubset(ofc))
    tmp = fill(gi, bgc, dots)
    lns = apply(toindices, colorfilter(objects(tmp, T, F, F), linc))
    dts = apply(initset, dots)
    cands = lns | dts
    nbreaks = unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    breaklocs = set()
    for breakobj in random.sample(totuple(cands), nbreaks):
        breaklocs.add(random.choice(totuple(breakobj)))
    gi = fill(gi, bgc, breaklocs)
    objs = colorfilter(objects(gi, T, F, F), bgc)
    objs = sfilter(objs, lambda o: len(o) == h * w)
    res = toindices(merge(objs))
    go = fill(gi, 3, res)
    go = replace(go, bgc, 4)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])  # (0,0) is always inside a lattice cell, never a line
    is_bg = (I == bgc)
    # lattice lines: rows/cols mostly made of line colour (at most one break per
    # segment, so a line row is >=2/3 line colour; other rows are <=1/3)
    line_rows = {r for r in range(hi) if (~is_bg[r]).sum() * 2 > wi}
    line_cols = {c for c in range(wi) if (~is_bg[:, c]).sum() * 2 > hi}

    # 4-connected background regions of I
    seen = np.zeros_like(is_bg)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if is_bg[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; comp = []
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < hi and 0 <= y < wi and is_bg[x, y] and not seen[x, y]:
                            seen[x, y] = True; q.append((x, y))
                comps.append(comp)

    ops, sels = [], []
    # 1. base layer: every background cell is first treated as an enclosed room (3)
    ops.append(3); sels.append(sel_of(list(zip(*np.nonzero(is_bg)))))
    # 2. every gap in a line leaks: flood 4 from the gap through whatever it joins
    for comp in comps:
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        if len(comp) > 1 and area == len(comp):
            continue  # intact lattice cell stays 3
        gaps = [p for p in comp if p[0] in line_rows or p[1] in line_cols]
        seed = min(gaps) if gaps else min(comp)
        ops.append(14); sels.append([seed[0], seed[1], 0, 0])  # FloodFill4 from the gap
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
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
                                f"for task 83302e8f"
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
                    f"Failed to build a complete episode for task 83302e8f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"83302e8f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
