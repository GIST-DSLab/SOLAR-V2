"""
ARC Task: 72322fa7 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    """Only the background colour is an episode-level role here: the rule is purely
    structural (complete every partial copy of a two-coloured template), so the object
    colours may vary freely per instance.  Foreground colours are kept non-zero inside
    generate() so that 0 always means 'background' for Copy/Paste."""
    import random as _r
    return {"bgc": _r.choice(list(range(10)))}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc=0, **kwargs) -> dict:
    # ---- helper: rebuild the output from the input with the task rule -----------------
    def _recon(gr, bg):
        Hh, Ww = len(gr), len(gr[0])
        seen = [[False] * Ww for _ in range(Hh)]
        comps = []
        for r in range(Hh):
            for c in range(Ww):
                if gr[r][c] != bg and not seen[r][c]:
                    st = [(r, c)]
                    seen[r][c] = True
                    comp = []
                    while st:
                        x, y = st.pop()
                        comp.append((x, y))
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < Hh and 0 <= ny < Ww and not seen[nx][ny] and gr[nx][ny] != bg:
                                    seen[nx][ny] = True
                                    st.append((nx, ny))
                    comps.append(comp)
        out = [[gr[r][c] for c in range(Ww)] for r in range(Hh)]
        ok_bounds = True
        ntemp = 0
        for comp in comps:
            colors = sorted({gr[r][c] for r, c in comp})
            if len(colors) != 2:
                continue
            ntemp += 1
            r0 = min(r for r, _ in comp)
            c0 = min(c for _, c in comp)
            objn = {(r - r0, c - c0): gr[r][c] for r, c in comp}
            oh = max(r for r, _ in objn) + 1
            ow = max(c for _, c in objn) + 1
            places = set()
            for col in colors:
                part = [(r, c) for (r, c), v in objn.items() if v == col]
                pr = min(r for r, _ in part)
                pc = min(c for _, c in part)
                pn = [(r - pr, c - pc) for r, c in part]
                ph = max(r for r, _ in pn) + 1
                pw = max(c for _, c in pn) + 1
                for i in range(Hh - ph + 1):
                    for j in range(Ww - pw + 1):
                        good = True
                        for dr, dc in pn:
                            if gr[i + dr][j + dc] != col:
                                good = False
                                break
                        if good:
                            places.add((i - pr, j - pc))
            for (tr, tc) in places:
                if tr < 0 or tc < 0 or tr + oh > Hh or tc + ow > Ww:
                    ok_bounds = False
                for (dr, dc), v in objn.items():
                    rr, cc = tr + dr, tc + dc
                    if 0 <= rr < Hh and 0 <= cc < Ww:
                        out[rr][cc] = v
        return out, ok_bounds, ntemp

    cols = interval(0, 10, 1)
    last = None
    for _attempt in range(64):
        h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
        # foreground colours never use 0 (0 stays reserved as "nothing" for Copy/Paste)
        remcols = tuple(c for c in cols if c != bgc and c != 0)
        nobjs = unifint(diff_lb, diff_ub, (1, min(4, len(remcols) // 2)))
        ccols = sample(remcols, 2 * nobjs)
        cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
        objs = []
        gi = canvas(bgc, (h, w))
        inds = asindices(gi)
        try:
            for ca, cb in cpairs:
                oh = unifint(diff_lb, diff_ub, (1, 4))
                ow = unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
                if choice((True, False)):
                    oh, ow = ow, oh
                bounds = asindices(canvas(-1, (oh, ow)))
                obj = {choice(totuple(bounds))}
                ncells = randint(2, oh * ow)
                for k in range(ncells - 1):
                    obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
                objn = normalize(obj)
                objt = totuple(objn)
                apart = sample(objt, randint(1, len(objt) - 1))
                bpart = difference(objt, apart)
                obj = recolor(ca, set(apart)) | recolor(cb, set(bpart))
                oh, ow = shape(obj)
                cands = sfilter(inds, lambda ij: shift(objn, ij).issubset(inds))
                if len(cands) == 0:
                    continue
                loc = choice(totuple(cands))
                plcd = shift(obj, loc)
                gi = paint(gi, plcd)
                plcdi = toindices(plcd)
                inds = (inds - plcdi) - mapply(neighbors, plcdi)
                objs.append(obj)
            if len(objs) == 0:
                continue
            avgs = sum([len(o) for o in objs]) / len(objs)
            ub = max(1, int((h * w) // (avgs * 2)))
            noccs = unifint(diff_lb, diff_ub, (1, ub))
            succ = 0
            tr = 0
            maxtr = 5 * noccs
            go = tuple(e for e in gi)
            while tr < maxtr and succ < noccs:
                tr += 1
                obj = choice(objs)
                ca, cb = list(palette(obj))
                oh, ow = shape(obj)
                cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
                if len(cands) == 0:
                    continue
                loc = choice(totuple(cands))
                plcd = shift(obj, loc)
                plcdi = toindices(plcd)
                if plcdi.issubset(inds):
                    succ += 1
                    inds = (inds - plcdi) - mapply(neighbors, plcdi)
                    go = paint(go, plcd)
                    col = choice((ca, cb))
                    gi = paint(gi, sfilter(plcd, lambda cij: cij[0] == col))
        except Exception:
            continue

        gil = [list(r) for r in gi]
        gol = [list(r) for r in go]
        last = {'input': gi, 'output': go}
        if gil == gol:
            continue
        rec, ok_bounds, ntemp = _recon(gil, bgc)
        if ok_bounds and ntemp == len(objs) and rec == gol:
            return {'input': gi, 'output': go}
    return last


def derive_operations(I, O):
    """Rule: every two-coloured template object is replicated (CopyI) onto every place
    where one of its single-coloured parts occurs alone (Paste at that offset)."""
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    A = np.asarray(I, dtype=int)
    B = np.asarray(O, dtype=int)
    H, W = A.shape
    g = [[int(A[r, c]) for c in range(W)] for r in range(H)]
    bgc = Counter(A.flatten().tolist()).most_common(1)[0][0]

    # ---- connected components of non-background cells (8-connected) ------------------
    seen = [[False] * W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] != bgc and not seen[r][c]:
                st = [(r, c)]
                seen[r][c] = True
                comp = []
                while st:
                    x, y = st.pop()
                    comp.append((x, y))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and not seen[nx][ny] and g[nx][ny] != bgc:
                                seen[nx][ny] = True
                                st.append((nx, ny))
                comps.append(comp)

    # ---- templates = components carrying exactly two colours -------------------------
    templates = []
    for comp in comps:
        colors = sorted({g[r][c] for r, c in comp})
        if len(colors) != 2:
            continue
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        objn = {(r - r0, c - c0): g[r][c] for r, c in comp}
        oh = max(r for r, _ in objn) + 1
        ow = max(c for _, c in objn) + 1
        places = set()
        for col in colors:                      # each single-coloured half of the template
            part = [(r, c) for (r, c), v in objn.items() if v == col]
            pr = min(r for r, _ in part)
            pc = min(c for _, c in part)
            pn = [(r - pr, c - pc) for r, c in part]
            ph = max(r for r, _ in pn) + 1
            pw = max(c for _, c in pn) + 1
            for i in range(H - ph + 1):         # every occurrence of that half in I
                for j in range(W - pw + 1):
                    good = True
                    for dr, dc in pn:
                        if g[i + dr][j + dc] != col:
                            good = False
                            break
                    if good:
                        places.add((i - pr, j - pc))
        templates.append((r0, c0, oh, ow, objn, sorted(places)))
    templates.sort(key=lambda t: (t[0], t[1]))

    ops, sels = [], []
    cur = [row[:] for row in g]

    for (r0, c0, oh, ow, objn, places) in templates:
        clip = [[g[r0 + i][c0 + j] for j in range(ow)] for i in range(oh)]

        # which placements still change the template's footprint at that spot?
        todo = []
        for (tr, tc) in places:
            if tr < 0 or tc < 0 or tr + oh > H or tc + ow > W:
                continue
            if any(cur[tr + dr][tc + dc] != v for (dr, dc), v in objn.items()):
                todo.append((tr, tc))
        if not todo:
            continue

        # CopyI of the template's bounding rectangle: the whole rectangle is intended,
        # it is what the clipboard must hold (object cells + their in-box surroundings).
        ops.append(28)
        sels.append([r0, c0, oh - 1, ow - 1])

        for (tr, tc) in todo:
            foot = {(tr + dr, tc + dc) for (dr, dc) in objn}
            prev = {}
            for i in range(oh):
                for j in range(ow):
                    v = clip[i][j]
                    if v == 0:                       # 0 is transparent for Paste
                        continue
                    rr, cc = tr + i, tc + j
                    if 0 <= rr < H and 0 <= cc < W:
                        prev[(rr, cc)] = cur[rr][cc]
                        cur[rr][cc] = v
            ops.append(30)
            sels.append([tr, tc, 0, 0])              # Paste uses the top-left corner only

            # the stamp is a rectangle: anything it covered that is NOT part of the
            # replicated object keeps the colour it had before the Paste
            repair = {}
            for (rr, cc), old in prev.items():
                if (rr, cc) not in foot and cur[rr][cc] != old:
                    repair.setdefault(old, []).append((rr, cc))
            for col in sorted(repair):
                cells = sorted(repair[col])
                ops.append(int(col))
                sels.append(sel_of(cells))
                for (rr, cc) in cells:
                    cur[rr][cc] = col

    # safety net (normally empty): anything the replication did not settle
    diff = {}
    for r in range(H):
        for c in range(W):
            if cur[r][c] != int(B[r, c]):
                diff.setdefault(int(B[r, c]), []).append((r, c))
    for col in sorted(diff):
        cells = sorted(diff[col])
        ops.append(int(col))
        sels.append(sel_of(cells))
        for (rr, cc) in cells:
            cur[rr][cc] = col

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
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
                                f"for task 72322fa7"
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
                    f"Failed to build a complete episode for task 72322fa7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72322fa7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
