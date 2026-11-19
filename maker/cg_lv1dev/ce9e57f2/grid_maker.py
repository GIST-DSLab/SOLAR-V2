"""
ARC Task: ce9e57f2 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque, Counter
import numpy as np
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical (top-anchored) grid


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 8)
    colss = random.sample(remcols, numc)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "colss": colss, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, colss=None, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if colss is None:
        rem = [c for c in cols if c != bgc]
        colss = random.sample(rem, random.randint(1, 8))
    colss = [c for c in colss if c != bgc and c != 8] or [c for c in cols if c != bgc][:1]
    if rot is None:
        rot = random.choice(ROTS)

    # for odd rotations the canonical grid dims swap
    if rot % 2 == 0:
        hmax, wmax = max_h, max_w
    else:
        hmax, wmax = max_w, max_h
    h = unifint(diff_lb, diff_ub, (6, max(6, min(30, hmax))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, wmax))))
    nbars = unifint(diff_lb, diff_ub, (2, max(2, (w - 2) // 2)))
    locopts = list(range(1, w - 1))
    barlocs = []
    for _ in range(nbars):
        if not locopts:
            break
        loc = random.choice(locopts)
        barlocs.append(loc)
        locopts = [x for x in locopts if x not in (loc, loc - 1, loc + 1)]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for j in barlocs:
        barloci = unifint(diff_lb, diff_ub, (1, h - 2))
        halfend = barloci // 2 if barloci % 2 == 1 else (barloci - 1) // 2
        barcol = random.choice(colss)
        gi[0:barloci + 1, j] = barcol
        go[0:barloci + 1, j] = barcol
        go[0:halfend + 1, j] = 8
    # rot quarter turns clockwise
    gi = np.rot90(gi, k=-rot)
    go = np.rot90(go, k=-rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour; bars are thin lines so bgc dominates
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    visited = np.zeros((hi, wi), dtype=bool)
    bars = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or visited[r, c]:
                continue
            col = I[r, c]
            comp = []
            dq = deque([(r, c)])
            visited[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == col:
                        visited[ny, nx] = True
                        dq.append((ny, nx))
            bars.append(comp)

    ops, sels = [], []
    for comp in bars:
        rows = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cs), max(cs)
        # each bar is a line rooted at one grid edge; the half nearest that edge turns 8
        if c0 == c1 and (r0 == 0 or r1 == hi - 1) and r1 > r0:
            L = r1 - r0 + 1
            half = L // 2
            if r0 == 0:
                cells = [(r, c0) for r in range(0, half)]
            else:
                cells = [(r, c0) for r in range(r1 - half + 1, r1 + 1)]
        elif r0 == r1 and (c0 == 0 or c1 == wi - 1):
            L = c1 - c0 + 1
            half = L // 2
            if c0 == 0:
                cells = [(r0, c) for c in range(0, half)]
            else:
                cells = [(r0, c) for c in range(c1 - half + 1, c1 + 1)]
        else:
            continue
        if not cells:
            continue
        ops.append(8)
        sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task ce9e57f2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce9e57f2"
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
                                f"for task ce9e57f2"
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
                    f"Failed to build a complete episode for task ce9e57f2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce9e57f2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
