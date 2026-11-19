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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc1, bgc2, sqc = random.sample(range(10), 3)
    return {"bgc1": bgc1, "bgc2": bgc2, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc1=None, bgc2=None, sqc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc1 is None or bgc2 is None or sqc is None:
        bgc1, bgc2, sqc = sample(cols, 3)
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
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def _e67_mostcolor(a):
    return Counter(a.flatten().tolist()).most_common(1)[0][0]


def _e67_panels(I):
    """Split I into its two equal panels (canvas = fewer colours, legend = more)."""
    H, W = I.shape
    urows = sum(1 for r in range(H) if len(set(I[r].tolist())) == 1)
    ucols = sum(1 for c in range(W) if len(set(I[:, c].tolist())) == 1)
    cands = []
    if H % 2 == 0:
        h2 = H // 2
        cands.append((urows, [(0, 0, h2, W), (h2, 0, h2, W)]))
    if W % 2 == 0:
        w2 = W // 2
        cands.append((ucols, [(0, 0, H, w2), (0, w2, H, w2)]))

    def ok(pan):
        a = I[pan[0][0]:pan[0][0] + pan[0][2], pan[0][1]:pan[0][1] + pan[0][3]]
        b = I[pan[1][0]:pan[1][0] + pan[1][2], pan[1][1]:pan[1][1] + pan[1][3]]
        return _e67_mostcolor(a) != _e67_mostcolor(b)
    cands.sort(key=lambda t: (ok(t[1]), t[0]), reverse=True)
    pan = cands[0][1]
    subs = [I[p[0]:p[0] + p[2], p[1]:p[1] + p[3]] for p in pan]
    order = sorted(range(2), key=lambda k: len(set(subs[k].flatten().tolist())))
    return pan[order[0]], subs[order[0]], pan[order[1]], subs[order[1]]


def _e67_plan(I):
    Cp, Cg, Lp, Lg = _e67_panels(I)
    cbg = _e67_mostcolor(Cg)
    lh, lw = Lg.shape
    ch, cw = Cg.shape

    def comps(bg):
        seen = np.zeros_like(Lg, dtype=bool)
        out = []
        for r in range(lh):
            for c in range(lw):
                if Lg[r, c] == bg or seen[r, c]:
                    continue
                comp, st = [], [(r, c)]
                seen[r, c] = True
                while st:
                    y, x = st.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < lh and 0 <= nx < lw and not seen[ny, nx] and Lg[ny, nx] != bg:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append(comp)
        return out

    def rect(comp):
        ys = [y for y, _ in comp]; xs = [x for _, x in comp]
        return len(comp) == (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)

    # legend background: the colour whose complement splits into solid rectangles (boxes)
    boxes = None
    for bg, _ in Counter(Lg.flatten().tolist()).most_common():
        cand = comps(bg)
        if all(rect(b) for b in cand):
            boxes = cand
            break
    if boxes is None:
        boxes = comps(_e67_mostcolor(Lg))
    if not boxes:
        return Cp, None, []
    # box fill = the colour shared by every box; prefer one never seen on the canvas
    common = set.intersection(*[set(int(Lg[y, x]) for (y, x) in b) for b in boxes])
    allc = Counter(int(Lg[y, x]) for b in boxes for (y, x) in b)
    cset = set(Cg.flatten().tolist())
    sqc = max(common or allc, key=lambda k: (k not in cset, allc[k]))
    infos = []
    for b in boxes:
        r0 = min(y for y, _ in b); c0 = min(x for _, x in b)
        rh = max(y for y, _ in b) - r0 + 1; rw = max(x for _, x in b) - c0 + 1
        box = Lg[r0:r0 + rh, c0:c0 + rw]
        infos.append((int((box != sqc).sum()), r0, c0, rh, rw, box))
    # most distinctive stencils (most marks) first; each stamp is checked against
    # the canvas as it stands after the earlier stamps
    infos.sort(key=lambda t: (-t[0], t[1], t[2]))
    P = np.full((ch + 2, cw + 2), cbg, dtype=int)
    P[1:-1, 1:-1] = Cg
    stamps = []
    for nmarks, r0, c0, rh, rw, box in infos:
        stencil = np.where(box == sqc, cbg, box)
        pat = np.full((rh + 2, rw + 2), cbg, dtype=int)
        pat[1:-1, 1:-1] = stencil
        hit = None
        for i in range(ch - rh + 1):
            for j in range(cw - rw + 1):
                if np.array_equal(P[i:i + rh + 2, j:j + rw + 2], pat):
                    hit = (i, j); break
            if hit:
                break
        if hit is None:
            continue
        P[hit[0] + 1:hit[0] + 1 + rh, hit[1] + 1:hit[1] + 1 + rw] = box
        stamps.append((nmarks, hit[0], hit[1], Lp[0] + r0, Lp[1] + c0, rh, rw))
    return Cp, sqc, stamps


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    ops, sels = [], []
    Cp, sqc, stamps = _e67_plan(I)
    ch, cw = Cp[2], Cp[3]
    # 1. keep the canvas panel, drop the legend panel (exact full rectangle -> bbox)
    ops.append(33); sels.append([Cp[0], Cp[1], ch - 1, cw - 1])
    # 2. each legend box whose marks are found on the canvas (most marks first, matched
    #    against the canvas as already stamped): copy the box (an exact full rectangle
    #    of the input) and paste it onto the canvas over its marks.
    for _, dr, dc, sr, sc, rh, rw in stamps:
        if sqc == 0:
            # Paste cannot write 0: lay the box fill on the landing rectangle first
            ops.append(0); sels.append([dr, dc, rh - 1, rw - 1])
        ops.append(28); sels.append([sr, sc, rh - 1, rw - 1])
        ops.append(30); sels.append([dr, dc, 0, 0])
    ops.append(34); sels.append([0, 0, ch - 1, cw - 1])
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
