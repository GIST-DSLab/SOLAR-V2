"""
ARC Task: 8d510a79 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    barcol = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barcol": barcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, barcol=5, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        d = random.uniform(diff_lb, diff_ub)
        return max(lb, min(ub, lb + round(d * (ub - lb))))

    hmax = max_w if transpose else max_h
    wmax = max_h if transpose else max_w
    h = unifint(5, min(30, hmax))
    w = unifint(3, min(30, wmax))
    barloci = random.randint(2, h - 3)
    gi = [[bgc] * w for _ in range(h)]
    for j in range(w):
        gi[barloci][j] = barcol
    go = [row[:] for row in gi]
    numtop = unifint(1, w - 1)
    numbot = unifint(1, w - 1)
    tops = random.sample(range(w), numtop)
    bots = random.sample(range(w), numbot)
    for t in tops:
        loci = random.randint(0, barloci - 2)
        col = random.choice((1, 2))
        gi[loci][t] = col
        rng = range(0, loci + 1) if col == 1 else range(loci, barloci)
        for r in rng:
            go[r][t] = col
    for t in bots:
        loci = random.randint(barloci + 2, h - 1)
        col = random.choice((1, 2))
        gi[loci][t] = col
        rng = range(loci, h) if col == 1 else range(barloci + 1, loci + 1)
        for r in rng:
            go[r][t] = col
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading taken from the golf entry: the task is a row-by-row SWEEP, not a
    # per-dot ray. Moving through the rows on each side of the bar, a front grows
    # one row at a time: 2s move toward the bar and 1s move away from it toward the
    # grid edge. Each op paints one row-step of one front. It selects the bg cells
    # that sit right past a cell of that colour in the row just painted, so every
    # selection can be read off the current grid.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour shared by every input in the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = [c for c, _ in cnt.most_common() if c not in (1, 2)][0]

    # the bar: a full row (or a full column) of one colour that is not bg and not a dot colour
    def find_bar(G):
        for r in range(G.shape[0]):
            v = int(G[r, 0])
            if v not in (bgc, 1, 2) and bool(np.all(G[r, :] == v)):
                return r
        return None

    transposed = False
    b = find_bar(I)
    G = I.copy()
    if b is None:
        G = I.T.copy()
        b = find_bar(G)
        transposed = True
    H = G.shape[0]

    ops, sels = [], []

    def emit(col, cells):
        if not cells:
            return
        for rr, cc in cells:
            G[rr, cc] = col
        out = [(cc, rr) for rr, cc in cells] if transposed else cells
        ops.append(col)
        sels.append(sel_of(out))

    def front(r_from, r_to, col):
        # the advancing front: bg cells in row r_to directly past a col-cell in row r_from
        return [(r_to, c) for c in range(G.shape[1])
                if G[r_from, c] == col and G[r_to, c] == bgc]

    # upper side: 2s move down toward the bar, one row at a time
    for r in range(0, b - 1):
        emit(2, front(r, r + 1, 2))
    # upper side: 1s move up to the top edge, one row at a time
    for r in range(b - 1, 0, -1):
        emit(1, front(r, r - 1, 1))
    # lower side: 2s move up toward the bar
    for r in range(H - 1, b + 1, -1):
        emit(2, front(r, r - 1, 2))
    # lower side: 1s move down to the bottom edge
    for r in range(b + 1, H - 1):
        emit(1, front(r, r + 1, 1))

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
                        f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
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
                                f"for task 8d510a79"
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
                    f"Failed to build a complete episode for task 8d510a79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d510a79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
