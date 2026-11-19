"""
ARC Task: 98cf29f8 (RE-ARC) — LLM-generated grid_maker
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


# ─────────────────────────────────────────────────────────────────────────────
# Reading taken from the golf entry: the connector is never erased as its own
# step. The block slides toward the anchor rectangle, and each unit step covers
# one more stem cell, so the stem is absorbed into the block's path. The maker
# instead erased the stem first and then moved the block. Both readings use the
# same unit (the block) and the same distance (the stem length). The difference
# is that here the stem disappears because the block moves over it, with no
# separate erase op.
# ─────────────────────────────────────────────────────────────────────────────


def sample_colors(num_examples=None) -> dict:
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, otherc=2) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    objh = _unifint(diff_lb, diff_ub, (2, h - 5))
    objw = _unifint(diff_lb, diff_ub, (2, w - 5))
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + objh, locj:locj + objw] = objc
    bmarg = h - (loci + objh)
    rmarg = w - (locj + objw)
    tmarg = loci
    lmarg = locj
    margs = (bmarg, rmarg, tmarg, lmarg)
    options = [i for i, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = np.rot90(gi, 3)  # dsl rot90 = clockwise
    h, w = gi.shape
    rr, cc = np.where(gi == objc)
    top, bot, left, right = rr.min(), rr.max(), cc.min(), cc.max()
    locis = random.randint(bot + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, right))
    locje = random.randint(max(locjs + 1, left), w - 1)
    ub = min(right, locje)
    lb = max(left, locjs)
    jloc = random.randint(lb, ub)
    ln_rows = list(range(bot + 1, locis))
    gib = gi.copy()
    gi = gi.copy()
    gi[locis:locie + 1, locjs:locje + 1] = otherc
    for r in ln_rows:
        gi[r, jloc] = otherc
    go = gib.copy()
    L = len(ln_rows)
    go[locis - L:locie + 1 - L, locjs:locje + 1] = otherc
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    # background = dominant colour of the grid border (as in the verifier)
    border = np.concatenate([I[0, :], I[-1, :], I[:, 0], I[:, -1]]).tolist()
    bg = Counter(border).most_common(1)[0][0]

    colors = [c for c in set(I.flatten().tolist()) if c != bg]
    if len(colors) != 2:
        ops.append(34); sels.append(submit_sel)
        return ops, sels

    def is_rect(c):
        rr, cc = np.where(I == c)
        return len(rr) == (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)

    # the anchor is a solid rectangle; the block+stem colour is not
    if is_rect(colors[0]) and not is_rect(colors[1]):
        anchorc, movc = colors
    elif is_rect(colors[1]) and not is_rect(colors[0]):
        movc, anchorc = colors
    else:
        ops.append(34); sels.append(submit_sel)
        return ops, sels

    mcells = [(int(r), int(c)) for r, c in zip(*np.where(I == movc))]
    rowcnt = Counter(r for r, _ in mcells)
    colcnt = Counter(c for _, c in mcells)
    stem_rows = {r for r, n in rowcnt.items() if n == 1}
    stem_cols = {c for c, n in colcnt.items() if n == 1}

    if stem_rows:      # vertical stem: block moves up/down
        stem = [p for p in mcells if p[0] in stem_rows]
        block = [p for p in mcells if p[0] not in stem_rows]
        dr = -1 if min(stem_rows) < min(r for r, _ in block) else 1
        dc = 0
    else:              # horizontal stem: block moves left/right
        stem = [p for p in mcells if p[1] in stem_cols]
        block = [p for p in mcells if p[1] not in stem_cols]
        dc = -1 if min(stem_cols) < min(c for _, c in block) else 1
        dr = 0
    nsteps = len(stem)
    block = sorted(block)
    dst = [(r + dr * nsteps, c + dc * nsteps) for r, c in block]

    if movc != 0:
        # Slide the block toward the anchor. Each step covers one more stem cell,
        # so the stem is absorbed by the block's own path.
        move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
        ops.append(move_op); sels.append(sel_of(block))   # grab the block
        for _ in range(nsteps - 1):
            ops.append(move_op); sels.append(sel_of([]))  # keep same object
        # the block's original footprint it no longer covers is left at 0
        hole = sorted(set(block) - set(dst))
        if bg != 0 and hole:
            ops.append(int(bg)); sels.append(sel_of(hole))
    else:
        # colour-0 content cannot be carried by ARCLE's object buffer
        vacated = sorted(set(block) - set(dst))
        if vacated:
            ops.append(int(bg)); sels.append(sel_of(vacated))
        ops.append(0); sels.append(sel_of(dst))

    ops.append(34); sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
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
                                f"for task 98cf29f8"
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
                    f"Failed to build a complete episode for task 98cf29f8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"98cf29f8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
