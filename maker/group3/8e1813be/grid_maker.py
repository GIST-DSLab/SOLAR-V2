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
    bgc, sqc = random.sample(range(10), 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS] + [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    lim = min(max_h, max_w)
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    nbars = _unif(diff_lb, diff_ub, (3, max(3, min(8, lim // 3))))
    ccols = random.sample(remcols, nbars)
    w = _unif(diff_lb, diff_ub, (nbars + 3, lim))
    hmarg = _unif(diff_lb, diff_ub, (2 * nbars, lim - nbars))
    go = [[col] * nbars for col in ccols]
    gi = [[col] * w for col in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(go) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    h2 = nbars + hmarg
    n = nbars
    loci = random.randint(1, h2 - n - 2)
    locj = random.randint(1, w - n - 2)
    gi = np.array(gi, dtype=int)
    gi[loci - 1:loci + n + 1, locj - 1:locj + n + 1] = bgc
    gi[loci:loci + n, locj:locj + n] = sqc
    go = np.array(go, dtype=int)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Rule: the 1-thick bars of I, kept in their order, are pushed together side by side
    into an n*n block (n = number of bars).
    Unit of work: each bar is COPIED out of the input. CopyI takes an unbroken stretch
    of the bar, and Paste drops it into the bar's slot in the top-left n*n block. If the
    square cut the bar shorter than n, the same piece is pasted again until the slot is
    full. A colour-0 bar cannot be copied (0 is transparent), so it is painted instead.
    Last, the canvas is cropped to the block.
    """
    I = np.asarray(I, dtype=int)
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

    # Work with each bar running along axis 0, then map back when emitting.
    A = I if vertical else I.T

    def cell(p, q):  # p = along the bar, q = across
        return (p, q) if vertical else (q, p)

    cur = A.copy()
    ops, sels = [], []
    for k, (pos, col) in enumerate(bars):
        if np.all(cur[0:n, k] == col):
            continue
        if col == 0:
            ops.append(0)
            sels.append(sel_of([cell(p, k) for p in range(n)]))
            cur[0:n, k] = 0
            continue
        # unbroken stretches of this bar along its line
        line = A[:, pos]
        runs, s = [], None
        for p in range(len(line) + 1):
            if p < len(line) and line[p] == col:
                if s is None:
                    s = p
            elif s is not None:
                runs.append((s, p - s))
                s = None
        s, L = max(runs, key=lambda t: t[1])      # longest unbroken stretch of the bar
        ln = min(L, n)
        clip_len = None
        p = 0
        while p < n:
            seg = min(ln, n - p)
            if not np.all(cur[p:p + seg, k] == col):
                if clip_len != seg:                  # copy once, reuse for each paste
                    ops.append(28)
                    sels.append(sel_of([cell(t, pos) for t in range(s, s + seg)]))
                    clip_len = seg
                ops.append(30)
                r0, c0 = cell(p, k)
                sels.append([r0, c0, 0, 0])
                cur[p:p + seg, k] = col
            p += seg
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
