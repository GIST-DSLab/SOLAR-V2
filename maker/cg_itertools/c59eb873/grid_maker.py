"""
ARC Task: c59eb873 (RE-ARC) — LLM-generated grid_maker
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
    # Rule (2x upscale) is colour-independent; only the background is fixed.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        lo = a + int(round((b - a) * max(0.0, lb)))
        hi_ = a + int(round((b - a) * min(1.0, ub)))
        lo, hi_ = min(lo, hi_), max(lo, hi_)
        return random.randint(max(a, lo), min(b, max(lo, hi_)))

    if bgc is None:
        bgc = random.choice(range(10))
    cap_h = max(1, min(15, max_h // 2))
    cap_w = max(1, min(15, max_w // 2))
    h = unifint(diff_lb, diff_ub, (1, cap_h))
    w = unifint(diff_lb, diff_ub, (1, cap_w))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = random.sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        if not inds:
            break
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = random.sample(inds, num)
        for (i, j) in chos:
            gi[i][j] = col
        cs = set(chos)
        inds = [x for x in inds if x not in cs]
    go = []
    for row in gi:
        wide = []
        for v in row:
            wide += [v, v]
        go.append(list(wide))
        go.append(list(wide))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: each row is stretched horizontally (every cell doubled across),
    # and then that stretched row is duplicated beneath itself.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 2 * hi, 2 * wi
    ops, sels = [], []

    # 1. grow canvas to twice I's size (bbox = exactly the full target rectangle)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    # 2. bottom-up, so input rows not yet consumed are never overwritten
    for r in range(hi - 1, -1, -1):
        row = 2 * r
        # 2a. horizontal stretch: paint input row r into output row 2r,
        #     one run of equal colours at a time (run covers 2*len cells)
        c = 0
        while c < wi:
            col = int(I[r, c]); e = c
            while e + 1 < wi and int(I[r, e + 1]) == col:
                e += 1
            cells = [(row, x) for x in range(2 * c, 2 * e + 2)]
            if any(cur[a, b] != col for a, b in cells):
                ops.append(col); sels.append(sel_of(cells))
                for a, b in cells:
                    cur[a, b] = col
            c = e + 1
        # 2b. vertical duplication: copy the stretched row onto the row below
        src = cur[row].copy()
        nz = src != 0
        after = cur[row + 1].copy()
        after[nz] = src[nz]
        if np.any(nz) and not np.array_equal(after, cur[row + 1]):
            ops.append(29); sels.append(sel_of([(row, x) for x in range(wo)]))
            ops.append(30); sels.append(sel_of([(row + 1, 0)]))
            cur[row + 1] = after
        # 2c. Paste is transparent for 0: clear cells of the copied row that the
        #     copy could not write (source cell is 0) but still hold stale content
        stale = [(row + 1, x) for x in range(wo)
                 if src[x] == 0 and cur[row + 1, x] != 0]
        if stale:
            ops.append(0); sels.append(sel_of(stale))
            for a, b in stale:
                cur[a, b] = 0

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
                        f"num_examples+1 ({num_examples + 1}) for task c59eb873"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c59eb873"
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
                                f"for task c59eb873"
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
                    f"Failed to build a complete episode for task c59eb873 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c59eb873-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
