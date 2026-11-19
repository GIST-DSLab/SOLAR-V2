"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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
    # Background is the only episode-level colour role; ring colours are free
    # (the rule depends on ring size, not on which colour a ring has).
    bgc = random.choice(range(10))
    variants = [{"with_center": True}, {"with_center": False}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, with_center=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if with_center is None:
        with_center = choice((True, False))
    lim = min(max_h, max_w, 30)
    dmax = max(2, min(7, lim // 4))
    while True:
        d = unifint(diff_lb, diff_ub, (2, dmax))
        h, w = d, d
        fullh = unifint(diff_lb, diff_ub, (4 * d, min(30, max_h)))
        fullw = unifint(diff_lb, diff_ub, (4 * d, min(30, max_w)))
        remcols = remove(bgc, cols)
        ccols = sample(remcols, d)
        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w - idx + 1)
            quad = fill(quad, c, (connect((idx, idx), (idx + linlen - 1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx + linlen - 1))))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if with_center:
            go = fill(go, choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            d = width(obj)
            dh = max(0, d // 2 - 1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - d and ij[1] <= fullw - d)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1
                loc = choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    """Rule (read from I only): every non-background colour is one square ring
    of four corner-L's (possibly clipped by the grid border); a lone single cell,
    if any, is the centre. Rings have distinct odd sides 3,5,...,2d+1; stack them
    concentrically on a (2d+1)x(2d+1) background canvas. Each ring's visible
    fragment is copied from the input and pasted at its place in the frame; the
    parts the border had clipped are completed by the ring's 4-fold symmetry."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    colors = sorted(set(I.flatten().tolist()) - {bgc})

    def ring(s, L):
        cells = set()
        for j in range(L):
            for (a, b) in ((0, j), (j, 0)):
                cells |= {(a, b), (a, s - 1 - b), (s - 1 - a, b), (s - 1 - a, s - 1 - b)}
        return cells

    vis_of = {col: {(int(r), int(c)) for r, c in zip(*np.nonzero(I == col))} for col in colors}

    # For each colour: every (side, arm, ring origin) whose ring, clipped by the
    # grid, reproduces exactly the visible cells.
    fits = {}
    for col in colors:
        vis = vis_of[col]
        top = min(vis)
        opts = []
        for s in range(3, 2 * len(colors) + 2, 2):
            for L in range(2, (s + 1) // 2 + 1):
                R = ring(s, L)
                for p in sorted(R):
                    dr, dc = top[0] - p[0], top[1] - p[1]
                    sh = {(r + dr, c + dc) for r, c in R}
                    if {(r, c) for r, c in sh if 0 <= r < hi and 0 <= c < wi} == vis:
                        opts.append((s, L, dr, dc))
                        break
        fits[col] = opts

    def solve(centre):
        rings = [c for c in colors if c != centre]
        d = len(rings)
        sides = {2 * (d - i) + 1 for i in range(d)}
        order_c = sorted(rings, key=lambda c: len({f[0] for f in fits[c]}))
        assign = {}

        def bt(k, used):
            if k == len(order_c):
                return True
            col = order_c[k]
            for f in fits[col]:
                if f[0] in sides and f[0] not in used:
                    assign[col] = f
                    if bt(k + 1, used | {f[0]}):
                        return True
            return False
        return (rings, d, assign) if bt(0, frozenset()) else None

    unfit = [c for c in colors if not fits[c]]
    one = [c for c in colors if len(vis_of[c]) == 1]
    tries = unfit[:1] if unfit else one + [None]
    res, centre = None, None
    for centre in tries:
        res = solve(centre)
        if res is not None:
            break
    rings, d, assign = res
    S = 2 * d + 1

    ops, sels = [], []
    # 1. Resize the canvas to the S x S output frame (full rectangle).
    ops.append(33); sels.append([0, 0, S - 1, S - 1])
    # crop is transparent: zeros stay zero, nonzero cells of I's corner survive
    cur = np.where(I[:S, :S] != 0, I[:S, :S], 0)
    # 2. Lay the background over the whole frame (full rectangle).
    if (cur != bgc).any():
        ops.append(bgc); sels.append([0, 0, S - 1, S - 1])
        cur[:, :] = bgc
    # 3. Each ring, outermost first: copy its visible fragment from the input,
    #    paste it at the ring's place in the frame, then complete clipped parts.
    for col in sorted(rings, key=lambda c: -assign[c][0]):
        s, L, dr, dc = assign[col]
        off = (S - s) // 2
        full = {(r + off, c + off) for r, c in ring(s, L)}
        vis = vis_of[col]
        placed = {(r - dr + off, c - dc + off) for r, c in vis}
        if col != 0:
            r0 = min(r for r, _ in vis); c0 = min(c for _, c in vis)
            ops.append(28); sels.append(sel_of(vis))
            ops.append(30); sels.append(sel_of([(r0 - dr + off, c0 - dc + off)]))
            for r, c in placed:
                cur[r, c] = col
            missing = sorted(full - placed)
        else:
            missing = sorted(full)   # colour 0 cannot travel through the clipboard
        if missing:
            ops.append(col); sels.append(sel_of(missing))
            for r, c in missing:
                cur[r, c] = col
    # 4. The single-cell colour, if present, goes in the centre.
    if centre is not None:
        if centre != 0:
            (r, c), = vis_of[centre]
            ops.append(28); sels.append(sel_of([(r, c)]))
            ops.append(30); sels.append(sel_of([(d, d)]))
        else:
            ops.append(0); sels.append(sel_of([(d, d)]))
        cur[d, d] = centre
    ops.append(34); sels.append([0, 0, S - 1, S - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
