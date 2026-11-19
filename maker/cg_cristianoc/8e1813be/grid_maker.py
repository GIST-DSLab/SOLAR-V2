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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    b = max(a, b)
    d = random.uniform(lb, ub)
    return min(b, max(a, round(a + d * (b - a))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, transpose=None, **kw):
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(range(10), 2)
    if transpose is None:
        transpose = random.choice([True, False])
    lim_w = max_h if transpose else max_w   # width before transposition
    lim_h = max_w if transpose else max_h
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    nmax = min(8, lim_h // 3, lim_w - 3)
    if nmax < 3:
        raise ValueError("grid too small")
    nbars = _unifint(diff_lb, diff_ub, (3, nmax))
    ccols = random.sample(remcols, nbars)
    w = _unifint(diff_lb, diff_ub, (nbars + 3, lim_w))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, lim_h - nbars))
    go = [[c] * nbars for c in ccols]
    gi = [[c] * w for c in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(go) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    h2 = nbars + hmarg
    n = nbars
    loci = random.randint(1, h2 - n - 2)
    locj = random.randint(1, w - n - 2)
    for i in range(loci - 1, loci + n + 1):
        for j in range(locj - 1, locj + n + 1):
            inside = loci <= i < loci + n and locj <= j < locj + n
            gi[i][j] = sqc if inside else bgc
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the golf entry, differs from the maker): the fat block is an
    interruption laid over a stack of full-length 1-thick bars.  Restore each bar through
    the block, then close up the background gaps so the bars sit side by side in their
    original order, and keep the n x n stack.
    Per bar, in order: repaint the bar's whole line if the block/outbox cut it inside the
    kept n x n window; then SLIDE the bar (one grab, then empty-selection Moves) toward the
    leading edge until it lands in slot k.  Colour-0 bars cannot be grabbed by ARCLE's
    Move (only nonzero cells are carried), so such a bar is laid into its slot line directly.
    Finally crop to the n x n block.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
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
        bars = sorted(bars_v + [(c, v) for (r, c, v) in amb])
        line_len = hi
        cell = lambda a, b: (b, a)  # a = column index, b = position along it
        move_op = 23                # MoveL
    else:
        bars = sorted(bars_h + [(r, v) for (r, c, v) in amb])
        line_len = wi
        cell = lambda a, b: (a, b)  # a = row index, b = position along it
        move_op = 20                # MoveU
    n = len(bars)
    G = I.copy()
    ops, sels = [], []

    def line_vals(a, g):
        return [int(g[cell(a, b)]) for b in range(line_len)]

    for k, (pos, col) in enumerate(bars):
        if col == 0:
            # colour-0 bar: nothing grabbable, lay it into its slot line directly
            if any(x != 0 for x in line_vals(k, G)[:n]):
                cells = [cell(k, b) for b in range(line_len)]
                ops.append(0); sels.append(sel_of(cells))
                for c_ in cells:
                    G[c_] = 0
            continue
        # restore the bar through the block when it is cut inside the kept window
        if any(x != col for x in line_vals(pos, G)[:n]):
            cells = [cell(pos, b) for b in range(line_len)]
            ops.append(col); sels.append(sel_of(cells))
            for c_ in cells:
                G[c_] = col
        # slide the bar to slot k: grab once, then empty-selection Moves
        steps = pos - k
        if steps > 0:
            cur = [cell(pos, b) for b in range(line_len) if G[cell(pos, b)] == col]
            snap = G.copy()
            for c_ in cur:
                snap[c_] = 0
            ops.append(move_op); sels.append(sel_of(cur))
            for _ in range(steps - 1):
                ops.append(move_op); sels.append(sel_of([]))
            dr, dc = cell(-steps, 0)
            G = snap
            for (r, c) in cur:
                G[r + dr, c + dc] = col
        # the vacated original line lies outside the crop or under a later bar: no repair needed
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
