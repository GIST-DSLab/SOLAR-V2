"""
ARC Task: f8b3ba0a (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the lattice colour and the plain block colour are fixed for the episode.
    # The marker colours do not change the rule: the output ranks colours by how
    # many blocks they occupy. So each instance picks its own marker colours.
    cols = list(range(10))
    fullbgc, bgc = random.sample(cols, 2)
    return {"fullbgc": fullbgc, "bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, fullbgc=None, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi_))))

    cols = list(range(10))
    if fullbgc is None or bgc is None:
        fullbgc, bgc = random.sample(cols, 2)

    h = unifint(diff_lb, diff_ub, (1, 5))
    w = unifint(diff_lb, diff_ub, (1, 5))
    while (max_h - 1) // (h + 1) < 3 and h > 1:
        h -= 1
    while (max_w - 1) // (w + 1) < 3 and w > 1:
        w -= 1
    nh = unifint(diff_lb, diff_ub, (3, max(3, (max_h - 1) // (h + 1))))
    nw = unifint(diff_lb, diff_ub, (3, max(3, (max_w - 1) // (w + 1))))
    fullh = (h + 1) * nh + 1
    fullw = (w + 1) * nw + 1

    remcols = [c for c in cols if c != fullbgc and c != bgc]
    gi = np.full((fullh, fullw), fullbgc, dtype=int)
    locs = []
    for a in range(nh):
        for b in range(nw):
            r0, c0 = a * (h + 1) + 1, b * (w + 1) + 1
            locs.append((r0, c0))
            gi[r0:r0 + h, c0:c0 + w] = bgc

    numc = unifint(diff_lb, diff_ub, (1, max(1, (nh * nw) // 2 - 1)))
    stack = []
    nn = numc + 1
    while nn > 1 and numc > 0 and len(remcols) > 0:
        nn3 = int(0.5 * (8 * numc + 1) ** 0.5 - 1)
        nn = min(max(1, nn3), nn - 1)
        if nn < 1:
            break
        col = random.choice(remcols)
        remcols.remove(col)
        numc -= nn
        stack.append((col, nn))

    for col, n in stack:
        chosen = random.sample(locs, n)
        for (r0, c0) in chosen:
            gi[r0:r0 + h, c0:c0 + w] = col
        locs = [l for l in locs if l not in chosen]

    go = [[c] for c, _ in stack]
    return {"input": gi.tolist(), "output": go}


def derive_operations(I, O=None, examples=None):
    # The golf entry and the maker read the task the same way. Each colour is
    # one unit. The units are ranked by how often they occur. The most common
    # unit is dropped, and the rest are written as a column in decreasing order.
    # The golf entry counts pixels, which only works for the original ARC task.
    # Here the border can outnumber a 2-block colour, so we count BLOCKS. That is
    # the same quantity the maker holds fixed, so the ops follow the maker.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    border = int(I[0, 0])  # row 0 / col 0 are always lattice lines

    def block_starts(is_sep):
        res, start = [], None
        for i, s in enumerate(is_sep):
            if not s and start is None:
                start = i
                res.append(i)
            elif s:
                start = None
        return res

    row_reps = block_starts([bool(np.all(I[r, :] == border)) for r in range(hi)])
    col_reps = block_starts([bool(np.all(I[:, c] == border)) for c in range(wi)])

    cnt = Counter()
    for r in row_reps:
        for c in col_reps:
            cnt[int(I[r, c])] += 1

    ranked = sorted(cnt.items(), key=lambda kv: -kv[1])
    target = [col for col, _ in ranked[1:]]   # drop the dominant block colour
    k = max(1, len(target))

    ops, sels = [], []
    # Shrink the canvas to a k x 1 strip: the full rectangle of the leftmost lattice column.
    ops.append(33); sels.append([0, 0, k - 1, 0])
    # A transparent crop keeps nonzero cells only: rows inside I hold the border colour (0 if border==0).
    for r, col in enumerate(target):
        cur = border if r < hi else 0
        if col != cur:
            ops.append(int(col)); sels.append(sel_of([(r, 0)]))

    ops.append(34); sels.append([0, 0, k - 1, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8b3ba0a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8b3ba0a"
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
                                f"for task f8b3ba0a"
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
                    f"Failed to build a complete episode for task f8b3ba0a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8b3ba0a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
