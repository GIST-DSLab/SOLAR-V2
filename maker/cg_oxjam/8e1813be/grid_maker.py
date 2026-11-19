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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants)) for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, transpose=None) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            raise ValueError
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, int(round(lo + d * (hi - lo)))))
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(range(10), 2)
    if transpose is None:
        transpose = random.choice((True, False))
    HH, WW = (max_w, max_h) if transpose else (max_h, max_w)
    HH, WW = min(HH, 30), min(WW, 30)
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    nbars = unifint(3, min(8, HH // 3, WW - 3))
    ccols = random.sample(remcols, nbars)
    w = unifint(nbars + 3, WW)
    hmarg = unifint(2 * nbars, HH - nbars)
    gi = [[c] * w for c in ccols]
    go = [[c] * nbars for c in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(go) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    gi = np.array(gi)
    h2 = nbars + hmarg
    loci = random.randint(1, h2 - nbars - 2)
    locj = random.randint(1, w - nbars - 2)
    gi[loci - 1:loci + nbars + 1, locj - 1:locj + nbars + 1] = bgc
    gi[loci:loci + nbars, locj:locj + nbars] = sqc
    go = np.array(go)
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule: I holds n one-cell-thick bars (all cells of a colour in one row, or one column),
    a fat square block and background.  Output = the bars alone, squeezed side by side in
    their original order into an n x n block.  The golf entry reads the task the same way
    (the single-colour lines, listed in order, repeated n times), so the plan is unchanged:
    paint each bar's colour onto its line of the top-left n x n block, in bar order
    (skipped if that line already holds it), then crop to the block.
    """
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    bars_h, bars_v, amb = [], [], []
    for v in np.unique(I):
        rs, cs = np.where(I == v)
        nr, nc = len(set(rs.tolist())), len(set(cs.tolist()))
        if nr == 1 and nc == 1:            # bar cut down to one cell by the square
            amb.append((int(rs[0]), int(cs[0]), int(v)))
        elif nr == 1:
            bars_h.append((int(rs[0]), int(v)))
        elif nc == 1:
            bars_v.append((int(cs[0]), int(v)))
    if bars_h or bars_v:
        vertical = len(bars_v) > len(bars_h)
    else:                                  # all bars cut to single cells: they share a line
        vertical = len({r for r, c, v in amb}) == 1
    if vertical:
        bars = sorted(bars_v + [(c, v) for r, c, v in amb])
    else:
        bars = sorted(bars_h + [(r, v) for r, c, v in amb])
    n = len(bars)
    ops, sels = [], []
    for k, (_, col) in enumerate(bars):
        if vertical:
            cells = [(r, k) for r in range(n)]
        else:
            cells = [(k, c) for c in range(n)]
        if any(I[r, c] != col for r, c in cells):
            ops.append(int(col)); sels.append(sel_of(cells))
    ops.append(33); sels.append([0, 0, n - 1, n - 1])   # crop to the full n x n block
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
