"""
ARC Task: 0962bcdd (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(2, 7)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (3, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None or len(ccols) < 2:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(2, 7))
    ccols = list(ccols)

    def unifint(lo, hi):
        lo_v = lo + int(round(diff_lb * (hi - lo)))
        hi_v = lo + int(round(diff_ub * (hi - lo)))
        lo_v, hi_v = max(lo, min(lo_v, hi)), max(lo, min(hi_v, hi))
        if hi_v < lo_v:
            hi_v = lo_v
        return random.randint(lo_v, hi_v)

    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    num = unifint(1, max(1, (h * w) // 25))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    subs = [(i, j) for i in range(h) for j in range(w) if i < h - 5 and j < w - 5]
    while succ < num and tr <= maxtrials:
        if not indss or not subs:
            break
        loci, locj = random.choice(subs)
        bd = {(i, j) for i in range(loci, loci + 5) for j in range(locj, locj + 5)}
        if bd.issubset(indss):
            ca, cb = random.sample(ccols, 2)
            cr, cc = loci + 2, locj + 2
            for k in range(5):
                go[cr, locj + k] = cb
                go[loci + k, cc] = cb
            for k in range(5):
                go[loci + k, locj + k] = ca
                go[loci + 4 - k, locj + k] = ca
            gi[cr, cc] = ca
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                gi[cr + dr, cc + dc] = cb
            succ += 1
            indss -= bd
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # background: canvas colour the generator paints first (always the clear majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # find every plus: centre colour ca, 4 orthogonal arms of one other colour cb
    plusses = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            ca = int(I[r, c])
            if ca == bgc:
                continue
            arms = {int(I[r - 1, c]), int(I[r + 1, c]), int(I[r, c - 1]), int(I[r, c + 1])}
            if len(arms) == 1:
                cb = next(iter(arms))
                if cb != bgc and cb != ca:
                    plusses.append((r, c, ca, cb))

    ops, sels = [], []

    # Stage 1, one colour at a time over the whole grid: lengthen the arms of every
    # plus whose arm colour is cb by one cell, all in one stroke.
    arm_cols = sorted({p[3] for p in plusses})
    for cb in arm_cols:
        tips = []
        for (r, c, ca, b) in plusses:
            if b == cb:
                tips += [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        ops.append(cb)
        sels.append(sel_of(tips))

    # Stage 2, one colour at a time: draw the full X (both diagonals) through every
    # plus whose centre colour is ca.
    ctr_cols = sorted({p[2] for p in plusses})
    for ca in ctr_cols:
        xs = []
        for (r, c, a, cb) in plusses:
            if a == ca:
                for d in (1, 2):
                    xs += [(r - d, c - d), (r - d, c + d), (r + d, c - d), (r + d, c + d)]
        ops.append(ca)
        sels.append(sel_of(xs))

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
                        f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0962bcdd"
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
                                f"for task 0962bcdd"
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
                    f"Failed to build a complete episode for task 0962bcdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0962bcdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
