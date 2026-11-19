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
from maker.sel_helpers import sel_of
try:
    from dsl import *
    from utils import *
except Exception:
    pass

LAYOUTS = ["top", "bottom", "left", "right"]


def sample_colors(num_examples=None) -> dict:
    bgc1, bgc2, sqc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(LAYOUTS):
        ex = list(LAYOUTS) + [random.choice(LAYOUTS) for _ in range(n_ex - len(LAYOUTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(LAYOUTS, n_ex)
    plan = [{"layout": l} for l in ex] + [{"layout": random.choice(ex)}]
    return {"bgc1": bgc1, "bgc2": bgc2, "sqc": sqc, "instance_plan": plan}


def _layout_of(fns, h, w):
    mk = tuple(tuple(1 for _ in range(w)) for _ in range(h)) + \
         tuple(tuple(2 for _ in range(w)) for _ in range(h))
    for fn in fns:
        mk = fn(mk)
    a = np.array(mk)
    if a[0, 0] == a[-1, 0]:  # split left/right
        return "left" if a[0, -1] == 2 else "right"
    return "top" if a[-1, 0] == 2 else "bottom"


def generate(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc, layout=None) -> dict:
    if layout is None:
        layout = random.choice(LAYOUTS)
    while True:
        res = _gen_once(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc, layout)
        if res is not None:
            return res


def _gen_once(diff_lb, diff_ub, max_h, max_w, bgc1, bgc2, sqc, layout):
    cols = interval(0, 10, 1)
    m = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (6, max(6, min(15, m // 2))))
    w = unifint(diff_lb, diff_ub, (8, max(8, min(30, m))))
    remcols = difference(cols, (bgc1, bgc2, sqc))
    gi1 = canvas(bgc1, (h, w))
    gi2 = canvas(bgc2, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
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
        ncells = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
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
    for _ in range(200):
        nmfs = choice((1, 2))
        fns = sample(mfs, nmfs)
        if _layout_of(fns, h, w) == layout:
            break
    for fn in fns:
        gi = fn(gi)
        go = fn(go)
    # reject instances whose answer is ambiguous (an unplaced template that
    # still occurs in the plain half)
    if _predict(gi) != go:
        return None
    return {"input": gi, "output": go}


def _predict(gi):
    I = np.asarray(gi, dtype=int)
    ops, sels = derive_operations(I, None)
    cur = None
    clip = None
    for op, sel in zip(ops, sels):
        if op == 33:
            r, c, h, w = sel
            cur = I[r:r + h + 1, c:c + w + 1].copy()
        elif op == 28:
            r, c, h, w = sel
            clip = I[r:r + h + 1, c:c + w + 1]
        elif op == 30:
            r, c = sel[0], sel[1]
            reg = cur[r:r + clip.shape[0], c:c + clip.shape[1]]
            reg[clip != 0] = clip[clip != 0]
        elif op == 0:
            for r, c in sel["cells"]:
                cur[r, c] = 0
    return tuple(tuple(int(v) for v in row) for row in cur)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # candidate splits into two equal halves; (kind, half A, half B, offset of B)
    cands = []
    if H % 2 == 0:
        hh = H // 2
        cands.append(("v", I[:hh], I[hh:], (hh, 0)))
    if W % 2 == 0:
        hw = W // 2
        cands.append(("h", I[:, :hw], I[:, hw:], (0, hw)))

    def mc(a):
        return Counter(a.flatten().tolist()).most_common(1)[0][0]

    def ok(A, B):
        ca, cb = mc(A), mc(B)
        return ca != cb and ca not in set(B.flatten().tolist()) and cb not in set(A.flatten().tolist())

    good = [c for c in cands if ok(c[1], c[2])]
    if good:
        split = good[0]
    else:  # fallback: verifier's frontier rule
        hl = sum(1 for r in range(H) if len(set(I[r].tolist())) == 1)
        vl = sum(1 for c in range(W) if len(set(I[:, c].tolist())) == 1)
        split = [c for c in cands if c[0] == ("v" if hl > vl else "h")][0]
    _, A, B, offB = split
    nA, nB = len(set(A.flatten().tolist())), len(set(B.flatten().tolist()))
    if nA < nB:
        tgt, tmpl, toff = A, B, (0, 0)
    else:
        tgt, tmpl, toff = B, A, offB
    th, tw = tgt.shape
    bg1, bg2 = mc(tmpl), mc(tgt)
    tgt_cols = set(tgt.flatten().tolist())
    cnt = Counter(int(v) for v in tmpl.flatten().tolist() if v != bg1)
    sq_c = [c for c in cnt if c not in tgt_cols] or list(cnt)
    sqc = max(sq_c, key=lambda c: cnt[c])

    # template boxes in tmpl: 4-connected comps of non-bg1 cells (rectangles)
    hh, ww = tmpl.shape
    seen = np.zeros_like(tmpl, dtype=bool)
    boxes = []
    for r in range(hh):
        for c in range(ww):
            if tmpl[r, c] == bg1 or seen[r, c]:
                continue
            stack = [(r, c)]; seen[r, c] = True; comp = []
            while stack:
                y, x = stack.pop(); comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hh and 0 <= nx < ww and not seen[ny, nx] and tmpl[ny, nx] != bg1:
                        seen[ny, nx] = True; stack.append((ny, nx))
            r0 = min(p[0] for p in comp); c0 = min(p[1] for p in comp)
            r1 = max(p[0] for p in comp); c1 = max(p[1] for p in comp)
            boxes.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))

    # where does each template's pattern occur in the target half (with bg2 around)
    matches = []
    for (r0, c0, bh, bw) in boxes:
        box = tmpl[r0:r0 + bh, c0:c0 + bw]
        pat = [(dr, dc) for dr in range(bh) for dc in range(bw) if box[dr, dc] != sqc]
        if not pat:
            continue
        for R in range(th - bh + 1):
            for C in range(tw - bw + 1):
                good_ = True
                for dr in range(-1, bh + 1):
                    for dc in range(-1, bw + 1):
                        y, x = R + dr, C + dc
                        if not (0 <= y < th and 0 <= x < tw):
                            continue
                        inside = 0 <= dr < bh and 0 <= dc < bw
                        want = int(box[dr, dc]) if inside and box[dr, dc] != sqc else bg2
                        if tgt[y, x] != want:
                            good_ = False; break
                    if not good_:
                        break
                if good_:
                    matches.append(((r0, c0, bh, bw), (R, C), box))

    # larger patterns first; a target pixel belongs to at most one box
    matches.sort(key=lambda m: -int((m[2] != sqc).sum()))
    claimed, chosen = set(), []
    for m in matches:
        (r0, c0, bh, bw), (R, C), box = m
        cells = {(R + dr, C + dc) for dr in range(bh) for dc in range(bw)}
        if cells & claimed:
            continue
        claimed |= cells
        chosen.append(m)
    matches = chosen

    ops, sels = [], []
    # 1. crop the canvas down to the half that lacks the framing boxes
    ops.append(33); sels.append([toff[0], toff[1], th - 1, tw - 1])  # full rectangular half
    cur = tgt.copy()
    # 2. stamp each matching template box around its pattern
    tr0 = offB if toff == (0, 0) else (0, 0)  # template half offset in I
    for (r0, c0, bh, bw), (R, C), box in matches:
        if sqc != 0:
            # whole template box rectangle (frame + pattern) copied from input
            ops.append(28); sels.append([tr0[0] + r0, tr0[1] + c0, bh - 1, bw - 1])
            ops.append(30); sels.append([R, C, 0, 0])
        else:
            # frame colour 0 cannot be pasted: paint the box's background cells
            cells = [(R + dr, C + dc) for dr in range(bh) for dc in range(bw)
                     if cur[R + dr, C + dc] == bg2]
            ops.append(0); sels.append(sel_of(cells))
        cur[R:R + bh, C:C + bw] = box
    ops.append(34); sels.append([0, 0, th - 1, tw - 1])
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
