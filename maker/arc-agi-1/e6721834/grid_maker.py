"""
ARC Task: e6721834 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc1, bgc2, sqc = random.sample(cols, 3)
    return {"bgc1": bgc1, "bgc2": bgc2, "sqc": sqc}


# ── rule helpers (shared by generate's well-posedness check and derive) ───────
# I holds two panels. One panel ("legend") shows template boxes: a solid sqc
# rectangle carrying a few coloured marks. The other panel ("canvas") shows the
# same marks, bare, at other places. O = the canvas panel with each matching
# box stamped back around its marks.

def _e67_most(a):
    return Counter(np.asarray(a).flatten().tolist()).most_common(1)[0][0]


def _e67_halves(I):
    """Split I into its two panels; return (canvas_rect, legend_rect)."""
    h, w = I.shape
    opts = []
    if h % 2 == 0:
        opts.append(((0, 0, h // 2, w), (h // 2, 0, h // 2, w)))
    if w % 2 == 0:
        opts.append(((0, 0, h, w // 2), (0, w // 2, h, w // 2)))
    sub = lambda R: I[R[0]:R[0] + R[2], R[1]:R[1] + R[3]]
    good = []
    for A, B in opts:
        a, b = sub(A), sub(B)
        ca, cb = _e67_most(a), _e67_most(b)
        # true split: each panel's dominant colour is wholly absent from the other
        if ca != cb and not (a == cb).any() and not (b == ca).any():
            good.append((A, B))
    if len(good) != 1:
        nrow = sum(1 for r in range(h) if len(set(I[r].tolist())) == 1)
        ncol = sum(1 for c in range(w) if len(set(I[:, c].tolist())) == 1)
        good = [o for o in opts if (o[0][2] == h // 2) == (nrow > ncol)] or opts
    A, B = good[0]
    # legend carries the extra box colour -> strictly more distinct colours
    if len(set(sub(A).flatten().tolist())) > len(set(sub(B).flatten().tolist())):
        A, B = B, A
    return A, B


def _e67_comps(g, bg):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    out = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg or seen[r, c]:
                continue
            st, cells = [(r, c)], []
            seen[r, c] = True
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            out.append(cells)
    return out


def _e67_boxes(lg, cv):
    """Legend background = the colour (absent from the canvas panel) whose
    complement is exactly a set of solid, two-coloured rectangles = the boxes.
    Never assume it is the majority colour: dense boxes can outnumber it."""
    lgc = set(lg.flatten().tolist())
    cvc = set(cv.flatten().tolist())
    for bg in sorted(lgc - cvc, key=lambda c: -int((lg == c).sum())):
        comps = _e67_comps(lg, bg)
        if not comps:
            continue
        ok = True
        for cells in comps:
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            if (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1) != len(cells):
                ok = False
                break
            if len({int(lg[y, x]) for y, x in cells}) < 2:
                ok = False
                break
        if ok:
            return comps
    return []


def _e67_find(cv, bg, rh, rw, obj):
    """Where this box's marks sit on the canvas: marks matching, and every other
    cell of the box footprint plus its surrounding ring still plain background."""
    H, W = cv.shape
    hits = []
    for dr in range(H - rh + 1):
        for dc in range(W - rw + 1):
            ok = True
            for i in range(rh):
                for j in range(rw):
                    if cv[dr + i, dc + j] != obj.get((i, j), bg):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for i in range(-1, rh + 1):
                for j in range(-1, rw + 1):
                    if 0 <= i < rh and 0 <= j < rw:
                        continue
                    rr, cc = dr + i, dc + j
                    if 0 <= rr < H and 0 <= cc < W and cv[rr, cc] != bg:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hits.append((dr, dc))
                if len(hits) > 1:
                    return hits
    return hits


def _e67_plan(I):
    I = np.asarray(I, dtype=int)
    C, L = _e67_halves(I)
    cv = I[C[0]:C[0] + C[2], C[1]:C[1] + C[3]]
    lg = I[L[0]:L[0] + L[2], L[1]:L[1] + L[3]]
    bg_cv = _e67_most(cv)
    comps = _e67_boxes(lg, cv)
    cnt = Counter(int(lg[y, x]) for cells in comps for y, x in cells)
    sqc = cnt.most_common(1)[0][0] if cnt else bg_cv   # box fill dominates the boxes
    stamps, amb = [], False
    for cells in comps:
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        r0, c0 = min(ys), min(xs)
        rh, rw = max(ys) - r0 + 1, max(xs) - c0 + 1
        obj = {(y - r0, x - c0): int(lg[y, x]) for y, x in cells if lg[y, x] != sqc}
        if not obj:
            continue
        hits = _e67_find(cv, bg_cv, rh, rw, obj)
        if len(hits) > 1:
            amb = True
        if hits:
            stamps.append((hits[0][0], hits[0][1], L[0] + r0, L[1] + c0, rh, rw, obj))
    stamps.sort(key=lambda s: (s[0], s[1]))
    return C, sqc, stamps, amb


def _e67_solve(I):
    I = np.asarray(I, dtype=int)
    C, sqc, stamps, amb = _e67_plan(I)
    if amb:
        return None
    out = I[C[0]:C[0] + C[2], C[1]:C[1] + C[3]].copy()
    for dr, dc, sr, sc, rh, rw, obj in stamps:
        out[dr:dr + rh, dc:dc + rw] = sqc
        for (i, j), v in obj.items():
            out[dr + i, dc + j] = v
    return out


def _e67_build(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc):
    cols = interval(0, 10, 1)
    lim = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (6, max(6, min(15, lim // 2))))
    w = unifint(diff_lb, diff_ub, (8, max(8, min(30, lim))))
    remcols = difference(cols, (bgc1, bgc2, sqc))
    gi1 = canvas(bgc1, (h, w))
    gi2 = canvas(bgc2, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    gi1inds = asindices(gi1)
    gi2inds = asindices(gi2)
    go = canvas(bgc2, (h, w))
    seen = []
    while tr < maxtr and succ < noccs:
        tr += 1
        oh = randint(2, min(6, h // 2))
        ow = randint(2, min(6, w // 2))
        cands = sfilter(gi1inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        bounds = shift(asindices(canvas(-1, (oh, ow))), loc)
        ncells = unifint(diff_lb, diff_ub, (1, max(1, (oh * ow) // 2)))
        obj = set(sample(totuple(bounds), ncells))
        objc = choice(remcols)
        objn = normalize(obj)
        if (objn, objc) in seen:
            continue
        seen.append(((objn, objc)))
        if bounds.issubset(gi1inds):
            succ += 1
            gi1inds = (gi1inds - bounds) - mapply(neighbors, bounds)
            gi1 = fill(gi1, sqc, bounds)
            gi1 = fill(gi1, objc, obj)
            cands2 = sfilter(gi2inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands2) == 0:
                continue
            loc2 = choice(totuple(cands2))
            bounds2 = shift(shift(bounds, invert(loc)), loc2)
            obj2 = shift(shift(obj, invert(loc)), loc2)
            if bounds2.issubset(gi2inds):
                gi2inds = (gi2inds - bounds2) - mapply(neighbors, bounds2)
                gi2 = fill(gi2, objc, obj2)
                go = fill(go, sqc, bounds2)
                go = fill(go, objc, obj2)
    gi = vconcat(gi1, gi2)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return gi, go


def generate(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc) -> dict:
    # a mark pattern that fits the canvas in two places makes the instance
    # unsolvable-in-principle -> resample until the rule pins one answer
    for _ in range(200):
        gi, go = _e67_build(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc)
        got = _e67_solve(gi)
        gon = np.array(go, dtype=int)
        if got is not None and got.shape == gon.shape and np.array_equal(got, gon):
            return {'input': gi, 'output': go}
    raise ValueError("no unambiguous instance")


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []
    C, sqc, stamps, _amb = _e67_plan(I)
    # 1. keep the canvas panel, drop the legend panel
    ops.append(33); sels.append([C[0], C[1], C[2] - 1, C[3] - 1])
    # 2. per matched box: copy that box out of the input, paste it over its marks.
    #    Paste is mark-transparent, so marks already on the canvas stay put.
    #    Only when the box fill is 0 (which Paste cannot write) clear it first.
    for dr, dc, sr, sc, rh, rw, obj in stamps:
        if sqc == 0:
            ops.append(0); sels.append([dr, dc, rh - 1, rw - 1])
        ops.append(28); sels.append([sr, sc, rh - 1, rw - 1])
        ops.append(30); sels.append([dr, dc, 0, 0])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task e6721834"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e6721834"
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
                                f"for task e6721834"
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
                    f"Failed to build a complete episode for task e6721834 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e6721834-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
