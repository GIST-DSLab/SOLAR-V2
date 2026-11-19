"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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

ORIENT_VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ORIENT_VARIANTS):
        examples = [dict(v) for v in ORIENT_VARIANTS]
        examples += [dict(random.choice(ORIENT_VARIANTS)) for _ in range(n_ex - len(ORIENT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ORIENT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, mirror=None, **kwargs) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    lim_h = min(30, max_w if mirror else max_h)   # gi rows (transposed when mirror)
    lim_w = min(30, max_h if mirror else max_w)
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    nmax = min(8, lim_h // 3, lim_w - 3)
    if nmax < 3:
        raise ValueError("grid too small")
    nbars = unifint(diff_lb, diff_ub, (3, nmax))
    ccols = list(sample(remcols, nbars))
    w = unifint(diff_lb, diff_ub, (nbars + 3, lim_w))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, lim_h - nbars))
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = randint(0, len(go) - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = randint(1, h2 - oh - 2)
    locj = randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the golf entry): strip away the background and the fat
    square; what is left are the 1-thick bars, and they close up against each
    other into an n x n block, keeping their order.
    Ops, per bar in order (uppermost / leftmost first):
      1. if the square cut a gap into the bar where it will be kept, repaint the
         bar's whole line in its colour (the bar is a full line in the rule);
      2. slide the bar into slot k: one grab with the bar's cells, then empty-
         selection Moves until it sits on line k (ARCLE restores the path);
         a 0-coloured bar cannot be grabbed, so its slot line is painted 0;
    then crop the n x n corner block.  The vacated original lines are never
    repaired: each lies either below the kept block or on a slot a later bar
    lands on, so a repair would not change the submitted grid.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    bars_h, bars_v, amb = [], [], []
    for v in np.unique(I):
        rs, cs = np.where(I == v)
        nr, nc = len(set(rs.tolist())), len(set(cs.tolist()))
        if nr == 1 and nc == 1:
            amb.append((int(rs[0]), int(cs[0]), int(v)))
        elif nr == 1:
            bars_h.append((int(rs[0]), int(v)))
        elif nc == 1:
            bars_v.append((int(cs[0]), int(v)))
    vertical = len(bars_v) > len(bars_h)
    if vertical:
        bars = bars_v + [(c, v) for (r, c, v) in amb]
    else:
        bars = bars_h + [(r, v) for (r, c, v) in amb]
    bars.sort()
    n = len(bars)

    G = I.copy()   # current grid, tracked in Python
    ops, sels = [], []

    def line_cells(p):
        return [(t, p) for t in range(H)] if vertical else [(p, t) for t in range(W)]

    for k, (pos, col) in enumerate(bars):
        cells = line_cells(pos)
        kept = cells[:n]                     # the part of the line that survives the crop
        # 1. complete the bar where the square interrupted it
        if any(G[r, c] != col for r, c in kept) and col != 0:
            ops.append(int(col))
            sels.append([0, pos, H - 1, 0] if vertical else [pos, 0, 0, W - 1])  # full line
            for r, c in cells:
                G[r, c] = col
        steps = pos - k
        if col == 0:
            # colour 0 is "nothing" to Move: paint the slot line instead
            slot = line_cells(k)
            if any(G[r, c] != 0 for r, c in slot[:n]):
                ops.append(0)
                sels.append([0, k, H - 1, 0] if vertical else [k, 0, 0, W - 1])  # full line
                for r, c in slot:
                    G[r, c] = 0
            continue
        if steps <= 0:
            continue
        # 2. slide the bar toward the front: grab once, then empty selections
        mop = 23 if vertical else 20         # MoveL / MoveU
        src = [(r, c) for r, c in cells if G[r, c] == col]
        snap = G.copy()
        for r, c in src:
            snap[r, c] = 0
        cur = list(src)
        ops.append(mop); sels.append(sel_of(cur))
        for s in range(steps):
            if s > 0:
                ops.append(mop); sels.append(sel_of([]))
            cur = [(r, c - 1) if vertical else (r - 1, c) for r, c in cur]
        G = snap
        for r, c in cur:
            G[r, c] = col

    # 3. keep only the assembled block (full rectangle)
    ops.append(33); sels.append([0, 0, n - 1, n - 1])
    ops.append(34); sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
