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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"has_centre": True}, {"has_centre": False}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    a = lb + (ub - lb) * diff_lb
    b = lb + (ub - lb) * diff_ub
    lo = int(round(min(a, b))); hi = int(round(max(a, b)))
    lo = max(lb, min(ub, lo)); hi = max(lo, min(ub, hi))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, has_centre=None, **kw) -> dict:
    if has_centre is None:
        has_centre = random.choice((True, False))
    cols = list(range(10))
    dmax = max(2, min(7, max_h // 4, max_w // 4))
    for _ in range(500):
        d = _unifint(diff_lb, diff_ub, 2, dmax)
        fullh = _unifint(diff_lb, diff_ub, 4 * d, max(4 * d, max_h))
        fullw = _unifint(diff_lb, diff_ub, 4 * d, max(4 * d, max_w))
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, d)
        quad = np.full((d + 1, d + 1), bgc, dtype=int)
        for idx, c in enumerate(ccols):
            linlen = random.randint(2, d - idx + 1)
            quad[idx:idx + linlen, idx] = c
            quad[idx, idx:idx + linlen] = c
        go = np.full((d + 1, 2 * d + 1), bgc, dtype=int)
        go[:, :d + 1] = quad
        go[:, d:] = quad[:, ::-1]
        go = np.vstack([go, go[::-1][1:]])
        if has_centre:
            go[d, d] = random.choice([c for c in remcols if c not in ccols])
        objs = []
        for c in sorted(set(go.flatten().tolist()) - {bgc}):
            objs.append([(int(r), int(q)) for r, q in zip(*np.nonzero(go == c))])

        def width(o):
            return max(q for _, q in o) - min(q for _, q in o) + 1
        random.shuffle(objs)
        objs.sort(key=width)
        gi = np.full((fullh, fullw), bgc, dtype=int)
        free = {(r, q) for r in range(fullh) for q in range(fullw)}
        ok = True
        for o in objs:
            r0 = min(r for r, _ in o); c0 = min(q for _, q in o)
            objn = [(r - r0, q - c0) for r, q in o]
            col = int(go[o[0]])
            w = width(o)
            dh = max(0, w // 2 - 1)
            base = {(i, j) for i in range(fullh) for j in range(fullw)
                    if i <= fullh - w and j <= fullw - w}
            cands = set(base)
            for (a, b) in ((-dh, 0), (0, -dh), (dh, 0), (0, dh)):
                cands |= {(i + a, j + b) for i, j in base}
            if not cands:
                ok = False; break
            cands = list(cands)
            succ = False
            vis = set()
            for _t in range(10):
                loc = random.choice(cands)
                sh = {(r + loc[0], q + loc[1]) for r, q in objn}
                vis = {p for p in sh if 0 <= p[0] < fullh and 0 <= p[1] < fullw}
                if vis.issubset(free):
                    succ = True
                    break
            if not succ:
                ok = False; break
            for p in vis:
                gi[p] = col
            free -= vis
        if not ok:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: the output is one corner quadrant
    (each colour contributes a single corner-L: its ring index fixes where the
    L sits, its arm length how long the arms are), and the rest of the picture
    is that quadrant mirrored left-right and then top-bottom.
    Route: open the S x S canvas laid with background, draw one corner-L per
    colour in the top-left quadrant, put the centre cell, then CopyO+Paste+Flip
    to mirror the left half to the right, then the top half to the bottom.
    Colour 0 cannot travel through Copy/Paste, so when 0 is a ring colour on a
    non-zero background its ring is drawn whole after the mirroring.
    O is never read."""
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

    fits = {}
    for col in colors:
        vis = {(int(r), int(c)) for r, c in zip(*np.nonzero(I == col))}
        top = min(vis)
        opts = []
        for s in range(3, 2 * len(colors) + 2, 2):
            for L in range(2, (s + 1) // 2 + 1):
                R = ring(s, L)
                for p in R:
                    dr, dc = top[0] - p[0], top[1] - p[1]
                    sh = {(r + dr, c + dc) for r, c in R}
                    if {(r, c) for r, c in sh if 0 <= r < hi and 0 <= c < wi} == vis:
                        opts.append((s, L))
                        break
        fits[col] = opts

    def solve(centre):
        rings = [c for c in colors if c != centre]
        d = len(rings)
        sides = {2 * (d - i) + 1 for i in range(d)}
        order_c = sorted(rings, key=lambda c: len({s for s, _ in fits[c]}))
        assign = {}

        def bt(k, used):
            if k == len(order_c):
                return True
            col = order_c[k]
            for s, L in fits[col]:
                if s in sides and s not in used:
                    assign[col] = (s, L)
                    if bt(k + 1, used | {s}):
                        return True
            return False
        return (rings, d, dict(assign)) if bt(0, frozenset()) else None

    unfit = [c for c in colors if not fits[c]]
    one = [c for c in colors if int((I == c).sum()) == 1]
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
    cur = np.where(I[:S, :S] != 0, I[:S, :S], 0)
    # 2. Lay the background over the whole frame (full rectangle).
    if (cur != bgc).any():
        ops.append(bgc); sels.append([0, 0, S - 1, S - 1])
        cur[:, :] = bgc

    # Colour 0 on a non-zero background cannot be carried by CopyO/Paste.
    late = [c for c in rings if c == 0 and bgc != 0]
    early = [c for c in rings if c not in late]

    # 3. One corner-L per colour in the top-left quadrant, outermost first.
    for col in sorted(early, key=lambda c: -assign[c][0]):
        s, L = assign[col]
        off = (S - s) // 2
        cells = {(off, off + j) for j in range(L)} | {(off + j, off) for j in range(L)}
        ops.append(col); sels.append(sel_of(cells))
        for p in cells:
            cur[p] = col
    # 4. Centre cell (lies on the mirror axes, so it is never copied).
    if centre is not None:
        ops.append(centre); sels.append(sel_of([(d, d)]))
        cur[d, d] = centre

    # 5. Mirror left columns 0..d-1 onto the right: copy, paste beside the
    #    axis column, flip that pasted block (full rectangles).
    ops.append(29); sels.append([0, 0, d, d - 1])
    ops.append(30); sels.append([0, d + 1, 0, 0])
    ops.append(26); sels.append([0, d + 1, d, d - 1])
    blk = cur[0:d + 1, 0:d]
    cur[0:d + 1, d + 1:] = blk[:, ::-1]
    # 6. Mirror top rows 0..d-1 onto the bottom the same way.
    ops.append(29); sels.append([0, 0, d - 1, S - 1])
    ops.append(30); sels.append([d + 1, 0, 0, 0])
    ops.append(27); sels.append([d + 1, 0, d - 1, S - 1])
    cur[d + 1:, :] = cur[0:d, :][::-1]

    # 7. A colour-0 ring (non-zero background) is drawn whole afterwards.
    for col in late:
        s, L = assign[col]
        off = (S - s) // 2
        cells = [(r + off, c + off) for r, c in ring(s, L)]
        ops.append(col); sels.append(sel_of(cells))
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
