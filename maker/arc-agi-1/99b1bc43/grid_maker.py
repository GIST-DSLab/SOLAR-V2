"""
ARC Task: 99b1bc43 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np


VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, acol, bcol, mirror=None) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    if mirror:
        hub = min(30, max_w)
        wub = min(14, (max_h - 1) // 2)
    else:
        hub = min(30, max_h)
        wub = min(14, (max_w - 1) // 2)
    h = unifint(diff_lb, diff_ub, (2, max(2, hub)))
    w = unifint(diff_lb, diff_ub, (2, max(2, wub)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numbdev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(bset) - set(aset)) | (set(aset) - set(bset))
    go = fill(c, 3, res)
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # --- split I at the separator line into the two panels -----------------
    if hi == ho and wi == 2 * wo + 1:          # vertical bar at col wo
        A = I[:, :wo]
        B = I[:, wo + 1:2 * wo + 1]
    else:                                      # horizontal bar at row ho
        A = I[:ho, :]
        B = I[ho + 1:2 * ho + 1, :]

    # shared color of both panels = background; each panel's other color = its marks
    pa = Counter(A.reshape(-1).tolist())
    pb = Counter(B.reshape(-1).tolist())
    common = [c for c in pa if c in pb]
    bgc = max(common, key=lambda c: pa[c] + pb[c])

    afg = (A != bgc)                 # marks of panel A
    bfg = (B != bgc)                 # marks of panel B
    both = afg & bfg                 # marks present in BOTH panels -> cancel out

    def comps(mask):
        seen = np.zeros(mask.shape, dtype=bool)
        out = []
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        y, x = stack.pop()
                        cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] \
                                    and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                    cells.sort()
                    out.append(cells)
        return out

    ops, sels = [], []
    cur = np.full(A.shape, bgc)      # simulated state of panel A region

    # --- step 1: recolor every mark-object of panel A ----------------------
    # object fully re-marked by B -> it cancels, erase it to bgc
    # otherwise -> it becomes a 3 mark (B-overlapped cells fixed in step 2)
    for cells in sorted(comps(afg), key=lambda cs: (-len(cs), cs[0])):
        r, c = cells[0]
        if all(both[y, x] for (y, x) in cells):
            ops.append(10 + bgc)
            sels.append([r, c, 0, 0])
            for (y, x) in cells:
                cur[y, x] = bgc
        else:
            ops.append(13)
            sels.append([r, c, 0, 0])
            for (y, x) in cells:
                cur[y, x] = 3

    # --- step 2: overlay each mark-object of panel B ------------------------
    # over an A mark -> the two agree, cancel to bgc; over empty -> new 3 mark
    for cells in sorted(comps(bfg), key=lambda cs: (-len(cs), cs[0])):
        rows = {}
        for (y, x) in cells:
            tgt = bgc if afg[y, x] else 3
            if cur[y, x] == tgt:
                continue
            rows.setdefault(y, []).append((x, tgt))
        for y in sorted(rows):
            run = sorted(rows[y])
            i = 0
            while i < len(run):
                j = i
                while j + 1 < len(run) and run[j + 1][0] == run[j][0] + 1 \
                        and run[j + 1][1] == run[i][1]:
                    j += 1
                x0, tgt = run[i][0], run[i][1]
                ops.append(int(tgt))
                sels.append([y, x0, 0, run[j][0] - x0])
                for xx in range(x0, run[j][0] + 1):
                    cur[y, xx] = tgt
                i = j + 1

    # --- panel B no longer needed: keep only panel A ------------------------
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 99b1bc43"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 99b1bc43"
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
                                f"for task 99b1bc43"
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
                    f"Failed to build a complete episode for task 99b1bc43 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"99b1bc43-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
