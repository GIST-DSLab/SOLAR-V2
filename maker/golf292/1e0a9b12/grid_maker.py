"""
ARC Task: 1e0a9b12 (RE-ARC) — LLM-generated grid_maker
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
    # background is the only colour role; foreground colours are free per cell
    return {"bgc": random.choice(list(range(10)))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)

    def unif(lo, hi):
        hi = max(lo, hi)
        lb = max(0.0, min(1.0, diff_lb))
        ub = max(lb, min(1.0, diff_ub))
        d = random.uniform(lb, ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))

    while True:
        h = unif(3, min(30, max_h))
        w = unif(3, min(30, max_w))
        nc = unif(1, w)
        gi = [[bgc] * w for _ in range(h)]
        remcols = [c for c in cols if c != bgc]
        scols = [random.choice(remcols) for _ in range(nc)]
        slocs = random.sample(range(w), nc)
        for c, l in zip(scols, slocs):
            nc2 = random.randint(1, h - 1)
            for r in random.sample(range(h), nc2):
                gi[r][l] = c
        go = [[bgc] * w for _ in range(h)]
        for c in range(w):
            stack = [gi[r][c] for r in range(h) if gi[r][c] != bgc]
            for i, v in enumerate(stack):
                go[h - len(stack) + i][c] = v
        cnt = Counter(v for row in gi for v in row)
        others = [k for k in cnt if k != bgc]
        if not others or cnt[bgc] > max(cnt[k] for k in others):
            break
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Background: the generator paints at most h-1 cells per column, so every
    # column keeps at least one background cell. Find the colours present in
    # every column (narrowed further by the demonstrations), then take the most
    # frequent of those.
    cand = set(I[:, 0].tolist())
    for c in range(1, w):
        cand &= set(I[:, c].tolist())
    if examples:
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            for c in range(ei.shape[1]):
                nxt = cand & set(ei[:, c].tolist())
                if nxt:
                    cand = nxt
    cnt = Counter(I.flatten().tolist())
    bgc = max(cand, key=lambda k: cnt[k]) if cand else cnt.most_common(1)[0][0]

    G = I.copy()  # working grid as ARCLE holds it
    ops, sels = [], []

    for c in range(w):
        # this column's non-background cells, top to bottom (gravity keeps their order)
        stack = [int(I[r, c]) for r in range(h) if I[r, c] != bgc]
        k = len(stack)
        if k == 0:
            continue
        top = h - k  # the stack will rest in rows top..h-1
        if all(I[r, c] != bgc for r in range(top, h)):
            continue  # column has already settled

        # 1) clear the column's empty top: every row above where the stack will rest
        region = [(r, c) for r in range(top)]
        ops.append(bgc)
        sels.append(sel_of(region))
        for r, cc in region:
            G[r, cc] = bgc

        # 2) lay the settled stack at the bottom, one same-colour run at a time
        i = 0
        while i < k:
            j = i
            while j + 1 < k and stack[j + 1] == stack[i]:
                j += 1
            cells = [(top + t, c) for t in range(i, j + 1)]
            col = stack[i]
            if any(G[r, cc] != col for r, cc in cells):
                ops.append(col)
                sels.append(sel_of(cells))
                for r, cc in cells:
                    G[r, cc] = col
            i = j + 1

    # Submit: the whole grid rectangle, background included
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1e0a9b12"
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
                                f"for task 1e0a9b12"
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
                    f"Failed to build a complete episode for task 1e0a9b12 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1e0a9b12-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
