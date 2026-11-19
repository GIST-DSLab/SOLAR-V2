"""
ARC Task: d687bc17 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = random.uniform(lo, hi)
    return max(a, min(b, int(round(v))))


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3, c4 = random.sample(range(10), 5)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, **kw) -> dict:
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[0][j] = c1
    for i in range(h):
        gi[i][0] = c2
    for i in range(h):
        gi[i][w - 1] = c3
    for j in range(w):
        gi[h - 1][j] = c4
    for (i, j) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        gi[i][j] = bgc
    go = [row[:] for row in gi]
    cands = [(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]
    ndots = _unifint(diff_lb, diff_ub, (1, min(len(cands), 2 * h + 2 * w)))
    dots = random.sample(cands, ndots)
    dots = [(random.choice((c1, c2, c3, c4)), ij) for ij in dots]

    def place(col, axis, size, target):
        n = [ij for c, ij in dots if c == col]
        cov = sorted(set(ij[axis] for ij in n))
        if len(cov) == size - 4 and size > 5:
            cov.remove(random.choice(cov))
        for k in cov:
            other = random.choice([ij[1 - axis] for ij in n if ij[axis] == k])
            if axis == 1:
                gi[other][k] = col
                go[target][k] = col
            else:
                gi[k][other] = col
                go[k][target] = col

    place(c1, 1, w, 1)
    place(c2, 0, h, 1)
    place(c4, 1, w, h - 2)
    place(c3, 0, h, w - 2)
    noisecands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    noisecols = [c for c in cols if c not in (bgc, c1, c2, c3, c4)]
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = _unifint(diff_lb, diff_ub, (0, max(0, ub)))
    nnoise = min(nnoise, len(noisecands))
    for (i, j) in random.sample(noisecands, nnoise):
        gi[i][j] = random.choice(noisecols)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # border sides are never touched by noise
    c1, c2, c3, c4 = int(I[0, 1]), int(I[1, 0]), int(I[1, w - 1]), int(I[h - 1, 1])
    border = {c1, c2, c3, c4}
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    region = corners + [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    # background: the dominant non-border colour of interior + corners
    # (noise may hit corners, so a single corner is not reliable)
    cnt = Counter(int(I[r, c]) for r, c in region if int(I[r, c]) not in border)
    bgc = cnt.most_common(1)[0][0]
    ops, sels = [], []

    # 1. erase noise: interior/corner cells that are neither bgc nor a border colour
    noise = [(r, c) for r, c in region
             if int(I[r, c]) != bgc and int(I[r, c]) not in border]
    if noise:
        ops.append(bgc); sels.append(sel_of(noise))

    # 2. each dot of a border colour slides toward that border until adjacent
    #    (c1 -> top: row 1, c2 -> left: col 1, c3 -> right: col w-2, c4 -> bottom: row h-2)
    dirs = {c1: (20, -1, 0), c2: (23, 0, -1), c3: (22, 0, 1), c4: (21, 1, 0)}
    dots = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if int(I[r, c]) in border]
    for color in (c1, c2, c3, c4):
        op, dr, dc = dirs[color]
        for (r, c) in dots:
            if int(I[r, c]) != color:
                continue
            if dr == -1:
                steps = r - 1
            elif dr == 1:
                steps = (h - 2) - r
            elif dc == -1:
                steps = c - 1
            else:
                steps = (w - 2) - c
            if steps <= 0:
                continue
            dest = (r + dr * steps, c + dc * steps)
            if color == 0:
                # 0 cells cannot be grabbed by Move (object buffer ignores 0): repaint
                ops.append(0); sels.append(sel_of([dest]))
                ops.append(bgc); sels.append(sel_of([(r, c)]))
                continue
            ops.append(op); sels.append(sel_of([(r, c)]))   # grab the dot
            for _ in range(steps - 1):
                ops.append(op); sels.append(sel_of([]))     # continue same grab
            if bgc != 0:
                ops.append(bgc); sels.append(sel_of([(r, c)]))  # repair vacated source

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
                        f"num_examples+1 ({num_examples + 1}) for task d687bc17"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d687bc17"
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
                                f"for task d687bc17"
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
                    f"Failed to build a complete episode for task d687bc17 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d687bc17-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
