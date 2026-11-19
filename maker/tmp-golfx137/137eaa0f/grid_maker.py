"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if dotc is None:
        dotc = choice(remove(bgc, cols))
    lim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(4, lim // 2))))
    remcols = [c for c in cols if c not in (bgc, dotc)]
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    lo = min(h, w) * 2
    gih = unifint(diff_lb, diff_ub, (lo, max(lo, max_h)))
    giw = unifint(diff_lb, diff_ub, (lo, max(lo, max_w)))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    for _ in range(60):
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                if gih - oh < 0 or giw - ow < 0:
                    break
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            return {'input': gi, 'output': go}
        maxtr = min(int(maxtr * 1.5) + 1, 500)
        gih = randint(gih, max(gih, max_h))
        giw = randint(giw, max(giw, max_w))
    raise ValueError("placement failed")


def _assign(dots, groups):
    """Pair every colour group with its own dot so that, laid over one another
    on their dots, the groups tile one rectangle (at most 4x4) exactly once."""
    order = sorted(range(len(groups)), key=lambda k: -len(groups[k]))
    best = [None]

    def near(cells):
        return sorted(dots, key=lambda d: (min(abs(d[0] - r) + abs(d[1] - c)
                                                for r, c in cells), d))

    def rec(i, used, taken, pick):
        if best[0] is not None:
            return
        if i == len(order):
            rs = [p[0] for p in taken]; cs = [p[1] for p in taken]
            hh = max(rs) - min(rs) + 1; ww = max(cs) - min(cs) + 1
            if hh * ww == len(taken):
                best[0] = dict(pick)
            return
        k = order[i]
        for d in near(groups[k]):
            if d in used:
                continue
            rel = {(r - d[0], c - d[1]) for r, c in groups[k]}
            if rel & taken:
                continue
            allc = taken | rel
            rs = [p[0] for p in allc]; cs = [p[1] for p in allc]
            if max(rs) - min(rs) >= 4 or max(cs) - min(cs) >= 4:
                continue
            pick[k] = d
            rec(i + 1, used | {d}, allc, pick)
            del pick[k]

    rec(0, frozenset(), {(0, 0)}, {})
    return best[0]


def _read_fragments(I):
    """Find the marker dots and, for each other colour, the dot it hangs from."""
    H, W = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = sorted(set(cnt) - {bgc})
    # one dot per fragment and one colour per fragment: the marker colour
    # has exactly as many cells as there are other colours
    for dotc in sorted(fg, key=lambda c: cnt[c]):
        if cnt[dotc] != len(fg) - 1:
            continue
        dots = [(r, c) for r in range(H) for c in range(W) if I[r, c] == dotc]
        cols = [c for c in fg if c != dotc]
        groups = [[(r, c) for r in range(H) for c in range(W) if I[r, c] == col]
                  for col in cols]
        pick = _assign(dots, groups)
        if pick is None:
            continue
        frags = [{"color": cols[k], "cells": groups[k], "dot": pick[k]}
                 for k in range(len(cols))]
        return bgc, dotc, frags
    raise ValueError("137eaa0f: could not pair the colours with their dots")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    bgc, dotc, frags = _read_fragments(I)

    # every dot is the same point: the frame is the union of each dot's
    # surroundings, measured relative to its dot
    rel = [(0, 0)]
    for f in frags:
        dr0, dc0 = f["dot"]
        rel += [(r - dr0, c - dc0) for r, c in f["cells"]]
    r_lo = min(p[0] for p in rel); r_hi = max(p[0] for p in rel)
    c_lo = min(p[1] for p in rel); c_hi = max(p[1] for p in rel)
    fh, fw = r_hi - r_lo + 1, c_hi - c_lo + 1
    lr, lc = -r_lo, -c_lo                     # where the dot sits in the frame

    ops, sels = [], []
    # a new canvas the size of that frame; it carries over whatever sat in
    # the input's top-left corner, so empty it out before stamping
    ops.append(33); sels.append([0, 0, fh - 1, fw - 1])
    leftover = [(r, c) for r in range(fh) for c in range(fw) if I[r, c] != 0]
    if leftover:
        ops.append(0); sels.append(sel_of(leftover))

    # stamp each dot together with the colour hanging from it into the frame,
    # lining its dot up on the frame's dot cell.  A fragment painted 0 is
    # already present: its cells are the frame's empty cells.
    order = sorted(frags, key=lambda f: f["dot"])
    for f in order:
        if f["color"] == 0:
            continue
        unit = f["cells"] + [f["dot"]]
        br = min(p[0] for p in unit); bc = min(p[1] for p in unit)
        dr0, dc0 = f["dot"]
        ops.append(28); sels.append(sel_of(unit))                       # CopyI
        ops.append(30); sels.append([br - dr0 + lr, bc - dc0 + lc, 0, 0])  # Paste

    ops.append(34); sels.append([0, 0, fh - 1, fw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
