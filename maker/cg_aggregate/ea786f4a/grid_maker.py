"""
ARC Task: ea786f4a (RE-ARC) — LLM-generated grid_maker
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
    # Only the line / centre colour has a role in the rule; the noise colours
    # are the remaining colours, sampled per instance.
    linc = random.choice(range(10))
    return {"linc": linc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + (hi - lo) * max(0.0, min(1.0, diff_lb))
    b = lo + (hi - lo) * max(0.0, min(1.0, diff_ub))
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    a = max(lo, min(hi, a))
    b = max(lo, min(hi, b))
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, linc=None, **kwargs) -> dict:
    if linc is None:
        linc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != linc]

    hmax = max(1, min(14, (max_h - 1) // 2))
    wmax = max(1, min(14, (max_w - 1) // 2))
    hh = _unifint(diff_lb, diff_ub, 1, hmax)
    ww = _unifint(diff_lb, diff_ub, 1, wmax)
    h, w = 2 * hh + 1, 2 * ww + 1
    cr, cc = hh, ww

    ncols = _unifint(diff_lb, diff_ub, 1, 9)
    ccols = random.sample(remcols, ncols)

    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    gi[cr][cc] = linc

    go = [row[:] for row in gi]
    for dr, dc in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
        r, c = cr + dr, cc + dc
        while 0 <= r < h and 0 <= c < w:
            go[r][c] = linc
            r += dr
            c += dc
    return {"input": gi, "output": go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # The grid has odd size (2h+1) x (2w+1); its exact centre cell holds the
    # marker colour. The X through the centre is drawn ROW BY ROW (the
    # reading the golf solution uses): in each row r at distance k = |r - cr|
    # from the centre row, the X touches the two cells (r, cc-k) and (r, cc+k).
    # Each row's X-pair is painted as one unit, sweeping from top to bottom.
    cr, cc = hi // 2, wi // 2
    marker = int(I[cr, cc])

    ops, sels = [], []
    for r in range(hi):
        if r == cr:
            continue  # the centre row's only X cell is the marker itself
        k = abs(r - cr)
        cells = []
        for c in (cc - k, cc + k):
            if 0 <= c < wi and int(I[r, c]) != marker:
                cells.append((r, c))
        if cells:
            ops.append(marker)
            sels.append(sel_of(sorted(set(cells))))

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
                        f"num_examples+1 ({num_examples + 1}) for task ea786f4a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ea786f4a"
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
                                f"for task ea786f4a"
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
                    f"Failed to build a complete episode for task ea786f4a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ea786f4a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
