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
    """Pair every colour group with its own anchor dot so that, laid over one
    another on their dots, the groups tile one rectangle (at most 4x4) once."""
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
            if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(taken):
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


def _read_fragments(I, examples=None):
    """Background, anchor colour, and for each other colour the anchor it hangs from."""
    H, W = I.shape
    cnt = Counter(I.flatten().tolist())
    # the background is shared by the whole episode: read it from every input
    tot = Counter(cnt)
    common = set(cnt)
    for ex in (examples or []):
        EI = np.asarray(ex[0], dtype=int)
        c2 = Counter(EI.flatten().tolist())
        tot.update(c2)
        common &= set(c2)
    bgc = max(common or set(cnt), key=lambda c: tot[c])
    fg = sorted(set(cnt) - {bgc})
    # the anchor colour is also episode-wide: it fills the example outputs'
    # anchor cell, so prefer colours present in every example output
    def pref(c):
        inall = all(c in set(np.asarray(ex[1]).flatten().tolist())
                    for ex in (examples or []))
        return (0 if inall else 1, cnt[c])
    for dotc in sorted(fg, key=pref):
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
    raise ValueError("137eaa0f: could not pair the colours with their anchors")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    bgc, dotc, frags = _read_fragments(I, examples)

    # Reading: every anchor dot is the same point.  The answer is a fresh frame
    # around that point, and each anchor's surroundings (anchor + its colour)
    # are stamped into it, anchor on anchor.
    rel = [(0, 0)]
    for f in frags:
        dr0, dc0 = f["dot"]
        rel += [(r - dr0, c - dc0) for r, c in f["cells"]]
    r_lo = min(p[0] for p in rel); r_hi = max(p[0] for p in rel)
    c_lo = min(p[1] for p in rel); c_hi = max(p[1] for p in rel)
    fh, fw = r_hi - r_lo + 1, c_hi - c_lo + 1
    lr, lc = -r_lo, -c_lo                     # anchor cell inside the frame

    ops, sels = [], []
    # 1. open the frame: a canvas of the frame's size (it keeps whatever the
    #    input had in its top-left corner, which the stamps then overwrite)
    ops.append(33); sels.append([0, 0, fh - 1, fw - 1])
    cur = np.zeros((fh, fw), dtype=int)
    for r in range(fh):
        for c in range(fw):
            cur[r, c] = I[r, c]

    # 2. stamp each anchor's neighbourhood into the frame, anchor on anchor
    order = sorted(frags, key=lambda f: f["dot"])
    for f in order:
        dr0, dc0 = f["dot"]
        land = [(r - dr0 + lr, c - dc0 + lc) for r, c in f["cells"]]
        if f["color"] == 0:
            # 0 is "nothing" to copy/paste: this colour's landing cells are
            # emptied directly
            todo = [p for p in land if cur[p] != 0]
            if todo:
                ops.append(0); sels.append(sel_of(todo))
                for p in todo:
                    cur[p] = 0
            continue
        unit = f["cells"] + [f["dot"]]
        br = min(p[0] for p in unit); bc = min(p[1] for p in unit)
        tgt = land + [(lr, lc)]
        if all(cur[p] == I[q] or I[q] == 0 for p, q in zip(tgt, unit)):
            continue                          # already standing there
        ops.append(28); sels.append(sel_of(unit))                         # CopyI
        ops.append(30); sels.append([br - dr0 + lr, bc - dc0 + lc, 0, 0])  # Paste
        for p, q in zip(tgt, unit):
            if I[q] != 0:                     # 0 never travels on the clipboard
                cur[p] = I[q]

    # the anchor cell itself, when no stamp could carry it: an anchor painted
    # 0 is "nothing" to copy/paste, so it is emptied directly; otherwise only
    # a 0-coloured neighbourhood hung from it, so the anchor is stamped alone
    if cur[lr, lc] != dotc and dotc == 0:
        ops.append(0); sels.append(sel_of([(lr, lc)]))
    elif cur[lr, lc] != dotc:
        d = frags[0]["dot"]
        ops.append(28); sels.append(sel_of([d]))
        ops.append(30); sels.append([lr, lc, 0, 0])

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
