"""
ARC Task: 88a62173 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule (take the odd value out at each position) does not depend on colour.
    # Only the background/separator colour is fixed for the whole episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    v = int(round(random.uniform(a, b)))
    return max(lo, min(hi, v))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    hmax = max(1, min(14, (max_h - 1) // 2))
    wmax = max(1, min(14, (max_w - 1) // 2))
    h = _unifint(diff_lb, diff_ub, 1, hmax)
    w = _unifint(diff_lb, diff_ub, 1, wmax)
    remcols = [c for c in range(10) if c != bgc]
    gib = [[bgc] * w for _ in range(h)]
    numc = _unifint(diff_lb, diff_ub, 1, min(9, h * w))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = _unifint(diff_lb, diff_ub, 1, max(1, len(inds) // numc))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gib[i][j] = col
        cs = set(chos)
        inds = [p for p in inds if p not in cs]
    numchinv = _unifint(diff_lb, diff_ub, 0, h * w - 1)
    numch = h * w - numchinv
    allinds = [(i, j) for i in range(h) for j in range(w)]
    subs = random.sample(allinds, numch)
    go = [row[:] for row in gib]
    banned = set(colsch + [bgc])
    for (x, y) in subs:
        opts = [c for c in range(10) if c != go[x][y] and c not in banned]
        if not opts:
            opts = [c for c in range(10) if c != go[x][y]]
        go[x][y] = random.choice(opts)
    gi = [[bgc] * (2 * w + 1) for _ in range(2 * h + 1)]
    idxes = [(0, 0), (h + 1, w + 1), (h + 1, 0), (0, w + 1)]
    trgloc = random.choice(idxes)
    for (oi, oj) in idxes:
        src = go if (oi, oj) == trgloc else gib
        for i in range(h):
            for j in range(w):
                gi[oi + i][oj + j] = src[i][j]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Per-cell reading (as in the code-golf entry): the input is four (h, w) panels
    # separated by a background row/column. At each position (i, j), compare the
    # four panel values and keep the least common one. The cell is the unit, not
    # the whole panel.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    h = (hi - 1) // 2
    w = (wi - 1) // 2
    origins = [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]

    ops, sels = [], []
    # 1. Crop to the top-left panel. This is a fixed rectangle, named before any
    #    answer is known. It becomes the base canvas.
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    cur = I[0:h, 0:w].copy()

    # 2. At every position where the four panels disagree, find the per-cell
    #    least common value. If the base canvas does not already hold it, plan to
    #    paint it. Group the cells by target colour: each group ("disagreement
    #    cells whose odd value is c") can be named from the input alone.
    by_color = {}
    for i in range(h):
        for j in range(w):
            vals = [int(I[r + i, c + j]) for (r, c) in origins]
            cnt = Counter(vals)
            if len(cnt) == 1:
                continue
            odd = min(vals, key=lambda v: cnt[v])
            if cur[i, j] != odd:
                by_color.setdefault(odd, []).append((i, j))

    for col in sorted(by_color):
        cells = by_color[col]
        ops.append(int(col)); sels.append(sel_of(cells))
        for (i, j) in cells:
            cur[i, j] = col

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 88a62173"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 88a62173"
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
                                f"for task 88a62173"
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
                    f"Failed to build a complete episode for task 88a62173 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"88a62173-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
