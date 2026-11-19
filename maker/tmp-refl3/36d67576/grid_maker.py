"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
    """Episode-level colour roles.

    bgc  : canvas background
    mainc: the body colour of the motif, kept on every incomplete copy
    markerc: the single marker cell, also kept on every incomplete copy
    The colours that have to be *completed* are drawn per instance from what is left,
    which is what the rule is about (they are never shown on an incomplete copy), so
    only the three roles above have to stay constant across the episode.
    """
    cols = list(range(10))
    bgc, mainc, markerc = random.sample(cols, 3)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, mainc=None, markerc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or mainc is None or markerc is None:
        bgc, mainc, markerc = sample(cols, 3)
    remcols = difference(cols, (bgc, mainc, markerc))

    hlo = min(10, max_h)
    wlo = min(10, max_w)

    while True:
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        ncols = unifint(diff_lb, diff_ub, (1, len(remcols)))
        ccols = sample(remcols, ncols)
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        ncells = len(obj)
        if ncells < 4:
            continue
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(ccols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5 + 10
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            oh, ow = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        if succ == 0:
            continue
        break

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """One motif on the grid is complete; every other copy of it shows only the cells
    that survived the stencil (body + marker), in some mirrored / rotated orientation.

    So the route is: grab the complete motif once (CopyI on its own cells), and for each
    incomplete copy — erase the fragment, stamp the motif, then actually perform the
    reflection / quarter turn that orients it, on the region it applies to.
    """
    import numpy as np
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []
    SUBMIT = [0, 0, ho - 1, wo - 1]

    changed = [(r, c) for r in range(h) for c in range(w) if I[r, c] != O[r, c]]
    if not changed or (ho, wo) != (h, w):
        return [34], [SUBMIT]

    # every completed cell was background before, so the background colour is simply
    # the (single) value the changed cells held in the input
    ivals = {int(I[r, c]) for r, c in changed}
    if len(ivals) == 1:
        bgc = ivals.pop()
    else:
        from collections import Counter
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    def comps(G):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if G[r, c] != bgc and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cur = []
                    while stack:
                        rr, cc = stack.pop()
                        cur.append((rr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                        and G[nr, nc] != bgc:
                                    seen[nr, nc] = True
                                    stack.append((nr, nc))
                    out.append(cur)
        return out

    def patch_of(G, cells):
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        c1 = max(c for _, c in cells)
        pat = np.full((r1 - r0 + 1, c1 - c0 + 1), bgc, dtype=int)
        for r, c in cells:
            pat[r - r0, c - c0] = int(G[r, c])
        return r0, c0, pat

    # the complete motif is the biggest connected clump of the input
    icomps = comps(I)
    if len(icomps) < 2:
        return [34], [SUBMIT]
    tpl_cells = max(icomps, key=len)
    tr, tc, P = patch_of(I, tpl_cells)
    th, tw = P.shape

    # 24 = Rotate90 (CCW), 25 = Rotate270 (CW), 26 = FlipH (fliplr), 27 = FlipV (flipud)
    # each entry: patch transform, rotation op (or None), flip ops applied afterwards.
    # ordered so the cheapest exact match wins — a flip that would not change the
    # picture is never emitted, because the transform it is part of matched earlier.
    TRANSFORMS = [
        (lambda A: A,                        None, []),        # identity
        (lambda A: np.fliplr(A),             None, [26]),      # vmirror
        (lambda A: np.flipud(A),             None, [27]),      # hmirror
        (lambda A: np.rot90(A, 2),           None, [26, 27]),  # rot180
        (lambda A: np.rot90(A, 1),           24,   []),        # rot90 ccw
        (lambda A: np.rot90(A, 3),           25,   []),        # rot90 cw
        (lambda A: np.flipud(np.rot90(A, 1)), 24,  [27]),      # dmirror (transpose)
        (lambda A: np.fliplr(np.rot90(A, 1)), 24,  [26]),      # cmirror (anti-transpose)
    ]

    def square_for(rot, r0, c0):
        """A square selection is what makes ARCLE's rotation land where it should, so
        stamp the motif inside a square that turns onto the copy's box.  Returns
        (square_r, square_c, side, paste_r, paste_c) or None when no square fits."""
        s = max(th, tw)
        best = None
        for a in range(s - th + 1):
            for b in range(s - tw + 1):
                if rot == 24:                     # (i,j) -> (s-1-j, i)
                    pr, pc = r0 - (s - b - tw), c0 - a
                else:                             # (i,j) -> (j, s-1-i)
                    pr, pc = r0 - b, c0 - (s - a - th)
                if pr < 0 or pc < 0 or pr + s > h or pc + s > w:
                    continue
                # everything in the square other than the copy's own box gets carried
                # around by the turn, so it has to be plain background on both sides
                ok = True
                for rr in range(pr, pr + s):
                    for cc in range(pc, pc + s):
                        if r0 <= rr < r0 + tw and c0 <= cc < c0 + th:
                            continue
                        if I[rr, cc] != bgc or O[rr, cc] != bgc:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                cost = abs(pr - r0) + abs(pc - c0)
                cand = (cost, pr, pc, s, pr + a, pc + b)
                if best is None or cand < best:
                    best = cand
        if best is None:
            return None
        return best[1], best[2], best[3], best[4], best[5]

    # every incomplete copy: a clump of the output that gained cells
    jobs = []
    for comp in comps(O):
        if not any(I[r, c] != O[r, c] for r, c in comp):
            continue                                  # the complete motif itself
        r0, c0, Q = patch_of(O, comp)
        plan = None
        for tf, rot, flips in TRANSFORMS:
            TP = tf(P)
            if TP.shape == Q.shape and np.array_equal(TP, Q):
                plan = (rot, flips)
                break
        if plan is None:
            jobs.append((r0, c0, comp, None))
            continue
        rot, flips = plan
        dh, dw = Q.shape
        sq = None
        if rot is not None:
            sq = square_for(rot, r0, c0)
            if sq is None:
                jobs.append((r0, c0, comp, None))     # no room to turn: paint it
                continue
        # a turn or a mirror rewrites the copy's whole box, so nothing foreign may
        # be sitting inside it
        if rot is not None or flips:
            dirty = any(O[rr, cc] != Q[rr - r0, cc - c0]
                        for rr in range(r0, r0 + dh) for cc in range(c0, c0 + dw))
            if dirty:
                jobs.append((r0, c0, comp, None))
                continue
        jobs.append((r0, c0, comp, (rot, flips, dh, dw, sq)))

    if not jobs:
        return [34], [SUBMIT]

    stamped = [j for j in jobs if j[3] is not None]
    if stamped:
        # grab the complete motif — its own cells, not its bounding rectangle
        ops.append(28)
        sels.append(sel_of(tpl_cells))

    # work outward from the complete motif
    jobs.sort(key=lambda j: ((j[0] - tr) ** 2 + (j[1] - tc) ** 2, j[0], j[1]))

    tpl_set = set(tpl_cells)
    zero_cells = [(i, j) for i in range(th) for j in range(tw)
                  if P[i, j] == 0 and (tr + i, tc + j) in tpl_set]

    for r0, c0, comp, plan in jobs:
        if plan is None:
            # no clean way to turn this one in place: complete it colour by colour
            bycol = {}
            for r, c in comp:
                if I[r, c] != O[r, c]:
                    bycol.setdefault(int(O[r, c]), []).append((r, c))
            for col in sorted(bycol):
                ops.append(col)
                sels.append(sel_of(bycol[col]))
            continue

        rot, flips, dh, dw, sq = plan
        # 1. wipe the fragment, so the square holds nothing but what we stamp
        frag = [(r, c) for r, c in comp if I[r, c] != bgc]
        if frag:
            ops.append(bgc)
            sels.append(sel_of(frag))
        # 2. stamp the motif
        if rot is None:
            pr, pc, side, par, pac = r0, c0, None, r0, c0
        else:
            pr, pc, side, par, pac = sq
        ops.append(30)
        sels.append([par, pac, 0, 0])
        # 3. colour 0 is "nothing" to Paste, so any 0-coloured motif cell is drawn by hand
        if bgc != 0 and zero_cells:
            ops.append(0)
            sels.append(sel_of([(par + i, pac + j) for i, j in zero_cells]))
        # 4. the quarter turn, on the square it needs (whole square intended)
        if rot is not None:
            ops.append(rot)
            sels.append([pr, pc, side - 1, side - 1])
        # 5. the mirror(s), on the copy's own box (whole box intended)
        for f in flips:
            ops.append(f)
            sels.append([r0, c0, dh - 1, dw - 1])

    ops.append(34)
    sels.append(SUBMIT)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
