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
import random
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, mainc, markerc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, mainc, markerc)]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc, "ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC generator, colors injected, h/w bounds parametrised,
#    plus a guard that makes every occurrence unambiguously readable)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w,
             bgc=None, mainc=None, markerc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or mainc is None or markerc is None:
        bgc, mainc, markerc = sample(cols, 3)
    remcols = difference(cols, (bgc, mainc, markerc))
    if not ccols:
        ncols = unifint(diff_lb, diff_ub, (1, len(remcols)))
        ccols = sample(remcols, ncols)
    ccols = tuple(ccols)
    symms = (identity, dmirror, cmirror, hmirror, vmirror)
    hlo, wlo = min(10, max_h), min(10, max_w)
    while True:
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        if oh > h or ow > w:
            continue
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(ccols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj

        # unambiguity guard: two symmetries that leave the *partial* (mainc +
        # markerc) pattern identical must also leave the completed object
        # identical -- otherwise an occurrence could not be read uniquely.
        seenmap = {}
        amb = False
        for f1 in symms:
            for f2 in symms:
                o = frozenset(f1(f2(obj)))
                p = frozenset(e for e in o if e[0] in (mainc, markerc))
                iv = invert(ulcorner(p))
                kk = frozenset(shift(p, iv))
                vv = frozenset(shift(o, iv))
                if seenmap.get(kk, vv) != vv:
                    amb = True
                    break
                seenmap[kk] = vv
            if amb:
                break
        if amb:
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
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice(symms)
            mf2 = choice(symms)
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            noh, now = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - noh and ij[1] <= w - now)
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


# ----------------------------------------------------------------------------
# 3. operations
#
# Rule: one complete stencil (the biggest object) is present; everywhere else
# only its "main + marker" skeleton appears, possibly mirrored / turned.  Each
# skeleton gets completed into a full copy of the stencil in that orientation.
# Route: CopyI the stencil once, Paste it onto every skeleton, then Flip /
# Rotate the stamped copy into the skeleton's orientation.
# ----------------------------------------------------------------------------
def _tf(k, a):
    if k == 0:
        return a
    if k == 1:
        return np.fliplr(a)
    if k == 2:
        return np.flipud(a)
    if k == 3:
        return np.fliplr(np.flipud(a))
    r = np.rot90(a, 3)          # rot90 CW  == ARCLE op25
    if k == 4:
        return r
    if k == 5:
        return np.fliplr(r)
    if k == 6:
        return np.flipud(r)
    return np.fliplr(np.flipud(r))


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    SUBMIT_SEL = [0, 0, H - 1, W - 1]

    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # ---- 8-connected non-background components -----------------------------
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if I[r0, c0] != bgc and not seen[r0, c0]:
                stack = [(r0, c0)]
                seen[r0, c0] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < H and 0 <= nb < W and not seen[na, nb] \
                                    and I[na, nb] != bgc:
                                seen[na, nb] = True
                                stack.append((na, nb))
                comps.append(cells)

    if len(comps) < 2:
        return [34], [SUBMIT_SEL]

    template = max(comps, key=len)
    skel_cols = set()
    for comp in comps:
        if comp is template:
            continue
        for (r, c) in comp:
            skel_cols.add(int(I[r, c]))
    if not skel_cols:
        return [34], [SUBMIT_SEL]

    tr0 = min(r for r, c in template)
    tr1 = max(r for r, c in template)
    tc0 = min(c for r, c in template)
    tc1 = max(c for r, c in template)
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    A = I[tr0:tr0 + th, tc0:tc0 + tw].copy()
    Mask = np.zeros((th, tw), dtype=bool)
    for (r, c) in template:
        Mask[r - tr0, c - tc0] = True
    holes = ~Mask
    paste_ok = True if not holes.any() else bool(np.all(A[holes] == bgc))

    # ---- every placement of the skeleton, in every orientation -------------
    placements = {}
    for k in range(8):
        Bv = _tf(k, A)
        Bm = _tf(k, Mask)
        Bh, Bw = Bv.shape
        pcells = [(r, c) for r in range(Bh) for c in range(Bw)
                  if Bm[r, c] and int(Bv[r, c]) in skel_cols]
        if not pcells:
            continue
        pr0 = min(r for r, c in pcells)
        pc0 = min(c for r, c in pcells)
        ph = max(r for r, c in pcells) - pr0 + 1
        pw = max(c for r, c in pcells) - pc0 + 1
        pat = [(r - pr0, c - pc0, int(Bv[r, c])) for r, c in pcells]
        for i in range(H - ph + 1):
            for j in range(W - pw + 1):
                hit = True
                for dr, dc, v in pat:
                    if I[i + dr, j + dc] != v:
                        hit = False
                        break
                if hit:
                    Ro, Co = i - pr0, j - pc0
                    key = (Ro, Co, Bv.tobytes(), Bm.tobytes(), Bh, Bw)
                    if key not in placements or k < placements[key][0]:
                        placements[key] = (k, Bv, Bm, Ro, Co)

    ordered = sorted(placements.values(), key=lambda t: (t[3], t[4], t[0]))

    G = I.copy()
    copied = False

    for (k, Bv, Bm, Ro, Co) in ordered:
        Bh, Bw = Bv.shape
        tgt = [(Ro + r, Co + c, int(Bv[r, c]))
               for r in range(Bh) for c in range(Bw) if Bm[r, c]]
        tgt_in = [(r, c, v) for r, c, v in tgt if 0 <= r < H and 0 <= c < W]
        if not tgt_in:
            continue
        if all(G[r, c] == v for r, c, v in tgt_in):
            continue                      # already complete (e.g. the stencil)

        transposing = k >= 4
        inside = (Ro >= 0 and Co >= 0 and Ro + Bh <= H and Co + Bw <= W)
        use_paste = inside and paste_ok
        occ = {(r, c) for r, c, v in tgt}
        if use_paste:                     # nothing foreign may sit in the frame
            for r in range(Ro, Ro + Bh):
                for c in range(Co, Co + Bw):
                    if (r, c) not in occ and G[r, c] != bgc:
                        use_paste = False
                        break
                if not use_paste:
                    break

        Sr = Sc = s = None
        pR, pC = Ro, Co
        if use_paste and transposing:
            # square frame S so that one CW turn carries the stamped stencil
            # exactly onto the skeleton's frame
            s = max(th, tw)
            found = None
            for Sr_ in range(max(0, Ro + Bh - s), min(Ro, H - s) + 1):
                for Sc_ in range(max(0, Co + Bw - s), min(Co, W - s) + 1):
                    a = Ro - Sr_
                    b = Co - Sc_
                    pr = s - th - b
                    pc = a
                    good = True
                    for r in range(Sr_, Sr_ + s):
                        for c in range(Sc_, Sc_ + s):
                            if Ro <= r < Ro + Bh and Co <= c < Co + Bw:
                                continue
                            if G[r, c] != bgc:
                                good = False
                                break
                        if not good:
                            break
                    if good:
                        found = (Sr_, Sc_, Sr_ + pr, Sc_ + pc)
                        break
                if found:
                    break
            if found is None:
                use_paste = False
            else:
                Sr, Sc, pR, pC = found

        if use_paste:
            prect = [(pR + a, pC + b) for a in range(th) for b in range(tw)]
            pset = set(prect)
            # (a) free the skeleton frame outside the stamping spot
            if transposing:
                clr = sorted((r, c) for r in range(Ro, Ro + Bh)
                             for c in range(Co, Co + Bw)
                             if (r, c) not in pset and G[r, c] != bgc)
                if clr:
                    ops.append(bgc)
                    sels.append(sel_of([(int(r), int(c)) for r, c in clr]))
                    for r, c in clr:
                        G[r, c] = bgc
            # (b) clear the stamping spot when the stencil carries colour 0
            #     (Paste never writes 0 cells, so they must be cleared first)
            need0 = any(A[a, b] == 0 and G[pR + a, pC + b] != 0
                        for a in range(th) for b in range(tw))
            if need0:
                ops.append(0)
                sels.append(sel_of([(int(r), int(c)) for r, c in sorted(prect)]))
                G[pR:pR + th, pC:pC + tw] = 0
            # (c) the stencil itself, copied once from the input
            if not copied:
                # full rectangle: the complete stencil frame, background included
                ops.append(28)
                sels.append([tr0, tc0, th - 1, tw - 1])
                copied = True
            # (d) stamp it
            ops.append(30)
            sels.append([int(pR), int(pC), 0, 0])
            for a in range(th):
                for b in range(tw):
                    if A[a, b] != 0:
                        G[pR + a, pC + b] = A[a, b]
            # (e) turn / mirror the stamped copy into the skeleton's orientation
            if transposing:
                # full square region, background included
                ops.append(25)
                sels.append([int(Sr), int(Sc), s - 1, s - 1])
                blk = G[Sr:Sr + s, Sc:Sc + s].copy()
                G[Sr:Sr + s, Sc:Sc + s] = np.rot90(blk, 3)
            if k in (1, 3, 5, 7):
                # full rectangle: the whole stamped copy
                ops.append(26)
                sels.append([int(Ro), int(Co), Bh - 1, Bw - 1])
                G[Ro:Ro + Bh, Co:Co + Bw] = np.fliplr(G[Ro:Ro + Bh, Co:Co + Bw])
            if k in (2, 3, 6, 7):
                ops.append(27)
                sels.append([int(Ro), int(Co), Bh - 1, Bw - 1])
                G[Ro:Ro + Bh, Co:Co + Bw] = np.flipud(G[Ro:Ro + Bh, Co:Co + Bw])
        else:
            # no room for a clean stamp here: draw the missing part of this
            # copy directly, one colour at a time
            bycol = {}
            for r, c, v in tgt_in:
                if G[r, c] != v:
                    bycol.setdefault(v, []).append((int(r), int(c)))
            for v in sorted(bycol):
                ops.append(int(v))
                sels.append(sel_of(sorted(bycol[v])))
                for r, c in bycol[v]:
                    G[r, c] = v

    ops.append(34)
    sels.append(SUBMIT_SEL)
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
