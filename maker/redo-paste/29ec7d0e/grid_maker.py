"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols) -> dict:
    for _attempt in range(60):
        rotf = random.choice((identity, rot90, rot180, rot270))
        # a quarter turn swaps the axes: sample inside the bounds it will land in
        glim_h, glim_w = (max_w, max_h) if rotf in (rot90, rot270) else (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (min(10, glim_h), glim_h))
        w = unifint(diff_lb, diff_ub, (min(10, glim_w), glim_w))
        if h < 6 or w < 6:
            continue
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
        go = canvas(bgc, (h, w))
        locs = set()
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci = (a + 1) + hp * a
                locj = (b + 1) + wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
        numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ = 0
        tr = 0
        maxtr = 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = random.randint(2, 6)
            pw = random.randint(2, 6)
            loci = random.randint(0, h - ph)
            locj = random.randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                        len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        if succ == 0:
            continue    # every patch was rejected: nothing to repair, not a task
        return {'input': rotf(gi), 'output': rotf(go)}
    raise ValueError("generation failed")


def _sel_cells(sel):
    if isinstance(sel, dict):
        return [(int(r), int(c)) for r, c in sel.get("cells", [])]
    r, c, dh, dw = sel
    return [(rr, cc) for rr in range(r, r + dh + 1) for cc in range(c, c + dw + 1)]


def _simulate(ops, sels, I):
    """Replay ops the way ARCLE does (Copy/Paste ignore 0 cells)."""
    h, w = I.shape
    M = 30
    inp = np.zeros((M, M), dtype=int)
    inp[:h, :w] = I
    g = inp.copy()
    clip = None
    for op, sel in zip(ops, sels):
        if op == 34:
            continue
        cells = _sel_cells(sel)
        if not cells:
            continue
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if 0 <= op <= 9:
            for (r, c) in cells:
                if 0 <= r < M and 0 <= c < M:
                    g[r, c] = op
        elif op == 28:
            clip = inp[r0:r1 + 1, c0:c1 + 1].copy()
        elif op == 29:
            clip = g[r0:r1 + 1, c0:c1 + 1].copy()
        elif op == 30 and clip is not None:
            ch, cw = clip.shape
            for i in range(ch):
                for j in range(cw):
                    if clip[i, j] != 0 and r0 + i < M and c0 + j < M:
                        g[r0 + i, c0 + j] = clip[i, j]
    return g[:h, :w]


def derive_operations(I, O):
    """The grid is one small tile repeated forever; blots of a single foreign
    colour have covered parts of it.  Repair = re-stamp every damaged tile with
    an intact copy of itself taken a whole number of periods away."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    diff = (I != O)
    if not diff.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # the blot colour: the one colour of I that the repeating pattern never uses
    extra = sorted(set(I.flatten().tolist()) - set(O.flatten().tolist()))
    noise = None
    if len(extra) == 1:
        cand = (I == extra[0])
        if np.array_equal(cand, diff):
            noise = cand
    if noise is None:
        noise = diff
    clean = ~noise

    # periods of the repetition: smallest shift every intact cell agrees with
    def find_period(axis):
        n = I.shape[axis]
        for p in range(1, n):
            if axis == 0:
                a, b = I[p:, :], I[:-p, :]
                both = clean[p:, :] & clean[:-p, :]
                oa, ob = O[p:, :], O[:-p, :]
            else:
                a, b = I[:, p:], I[:, :-p]
                both = clean[:, p:] & clean[:, :-p]
                oa, ob = O[:, p:], O[:, :-p]
            if (a[both] == b[both]).all() and np.array_equal(oa, ob):
                return p
        return n

    ph = find_period(0)
    pw = find_period(1)

    # nearest intact block of the same size, a whole number of periods away
    def find_src(r, c, a, b):
        best = None
        for r2 in range(r % ph, h - a + 1, ph):
            for c2 in range(c % pw, w - b + 1, pw):
                if clean[r2:r2 + a, c2:c2 + b].all():
                    d = abs(r2 - r) + abs(c2 - c)
                    if best is None or d < best[0]:
                        best = (d, r2, c2)
        return None if best is None else (best[1], best[2])

    tasks = []

    def plan(r, c, a, b):
        if not noise[r:r + a, c:c + b].any():
            return                                   # this tile survived intact
        src = find_src(r, c, a, b)
        if src is not None:
            tasks.append(('rect', (r, c, a, b), src))
            return
        if a == 1 and b == 1:                        # no intact twin anywhere
            tasks.append(('cell', r, c, int(O[r, c])))
            return
        if a >= b:                                   # halve; each half may find one
            a1 = a // 2
            plan(r, c, a1, b); plan(r + a1, c, a - a1, b)
        else:
            b1 = b // 2
            plan(r, c, a, b1); plan(r, c + b1, a, b - b1)

    for r in range(0, h, ph):
        for c in range(0, w, pw):
            plan(r, c, min(ph, h - r), min(pw, w - c))

    # one clipboard load per stamp; smallest stamps first so no load is spare
    groups = {}
    for kind, dest, src in [t for t in tasks if t[0] == 'rect']:
        groups.setdefault((dest[2], dest[3], src[0], src[1]), []).append((dest[0], dest[1]))
    order = sorted(groups, key=lambda k: (k[0] * k[1], k))

    cur = I.copy()
    for key in order:
        a, b, r2, c2 = key
        block = I[r2:r2 + a, c2:c2 + b]
        has_ink = bool((block != 0).any())
        if has_ink:
            ops.append(28); sels.append([r2, c2, a - 1, b - 1])   # exactly this rectangle
        for (r, c) in sorted(groups[key]):
            zm = (block == 0)
            # Paste cannot carry the stamp's 0 cells -> clear the tile first
            if zm.any() and (cur[r:r + a, c:c + b][zm] != 0).any():
                ops.append(0); sels.append([r, c, a - 1, b - 1])  # exactly this rectangle
                cur[r:r + a, c:c + b] = 0
            if has_ink:
                ops.append(30); sels.append([r, c, 0, 0])
                tgt = cur[r:r + a, c:c + b]
                m = block != 0
                tgt[m] = block[m]

    for t in tasks:                                  # lone cells with no intact twin
        if t[0] == 'cell' and cur[t[1], t[2]] != t[3]:
            ops.append(t[3]); sels.append(sel_of([(t[1], t[2])]))
            cur[t[1], t[2]] = t[3]

    rem = {}
    for r in range(h):
        for c in range(w):
            if cur[r, c] != O[r, c]:
                rem.setdefault(int(O[r, c]), []).append((r, c))
    for col in sorted(rem):
        ops.append(col); sels.append(sel_of(rem[col]))
        for (r, c) in rem[col]:
            cur[r, c] = col

    i = 0                                            # drop anything not needed
    while i < len(ops):
        t_ops, t_sels = ops[:i] + ops[i + 1:], sels[:i] + sels[i + 1:]
        if np.array_equal(_simulate(t_ops, t_sels, I), O):
            ops, sels = t_ops, t_sels
        else:
            i += 1

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
