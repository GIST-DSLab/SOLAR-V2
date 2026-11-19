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
    """Episode-level colour roles: background, main body colour, marker colour.

    The completion colours are drawn (per instance) from whatever is left over, so the
    only thing that has to be pinned down is the role triple.  When bgc != 0 we keep 0
    out of the foreground pool entirely: ARCLE's Copy/Paste treat 0 as 'nothing', so a
    0-coloured object cell would silently fail to be stamped.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    if bgc == 0:
        pool = [c for c in cols if c != 0]
    else:
        pool = [c for c in cols if c != 0 and c != bgc]
    mainc, markerc = random.sample(pool, 2)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, mainc=None, markerc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    fgpool = [c for c in cols if c != bgc and (bgc == 0 or c != 0)]
    if mainc is None or markerc is None:
        mainc, markerc = sample(fgpool, 2)
    remcols = [c for c in fgpool if c != mainc and c != markerc]

    hlo = min(10, max_h)
    wlo = min(10, max_w)

    while True:
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))

        # the motif lives in a square n x n bounding box and spans it fully, so every
        # dihedral image of it occupies a congruent square box
        n = unifint(diff_lb, diff_ub, (3, 5))
        bounds = asindices(canvas(-1, (n, n)))
        obj = None
        for _ in range(60):
            ncells = unifint(diff_lb, diff_ub, (max(4, n), n * n))
            cand = {choice(totuple(bounds))}
            good = True
            for k in range(ncells - 1):
                frontier = totuple((bounds - cand) & mapply(neighbors, cand))
                if len(frontier) == 0:
                    good = False
                    break
                cand.add(choice(frontier))
            if not good:
                continue
            cand = normalize(cand)
            if shape(cand) == (n, n):
                obj = cand
                break
        if obj is None:
            continue

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
        tocompcobj = {(choice(remcols), ij) for ij in tocompc}
        fullobj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)

        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(fullobj) != fullobj:
                isfakesymm = True
                break
        if isfakesymm:
            continue

        gi = canvas(bgc, (h, w))
        loci = randint(0, h - n)
        locj = randint(0, w - n)
        plcd = shift(fullobj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(fullobj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5 + 10
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(fullobj))
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
    """Rule: one complete motif is the template; every other copy on the grid shows only
    its main/marker cells, in some mirrored/rotated orientation.  Each incomplete copy is
    re-stamped from the template and then mirrored/rotated into its orientation, so the
    reflection is actually performed on the region it applies to."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def comps(G, bg):
        seen = np.zeros((h, w), dtype=bool)
        res = []
        for r in range(h):
            for c in range(w):
                if G[r, c] != bg and not seen[r, c]:
                    st = [(r, c)]
                    seen[r, c] = True
                    cur = []
                    while st:
                        rr, cc = st.pop()
                        cur.append((rr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w and (not seen[nr, nc]) and G[nr, nc] != bg:
                                    seen[nr, nc] = True
                                    st.append((nr, nc))
                    res.append(cur)
        return res

    # dihedral images of an s x s box, ordered by number of ARCLE ops needed
    # 26 = FlipH (fliplr), 27 = FlipV (flipud), 24 = Rotate90 (CCW), 25 = Rotate270 (CW)
    TRANSFORMS = [
        ([],       lambda r, c, s: (r, c)),                  # identity
        ([26],     lambda r, c, s: (r, s - 1 - c)),          # vmirror
        ([27],     lambda r, c, s: (s - 1 - r, c)),          # hmirror
        ([24],     lambda r, c, s: (s - 1 - c, r)),          # rot90 CCW
        ([25],     lambda r, c, s: (c, s - 1 - r)),          # rot90 CW
        ([26, 27], lambda r, c, s: (s - 1 - r, s - 1 - c)),  # rot180
        ([24, 27], lambda r, c, s: (c, r)),                  # dmirror (transpose)
        ([25, 27], lambda r, c, s: (s - 1 - c, s - 1 - r)),  # cmirror (anti-transpose)
    ]

    counts = Counter(I.flatten().tolist())
    freq_order = [c for c, _ in counts.most_common()]
    bg = freq_order[0]
    plan = None

    if (ho, wo) == (h, w):
        for cand in freq_order[:4]:
            icomps = comps(I, cand)
            if len(icomps) < 2:
                continue
            tmpl = max(icomps, key=len)
            sr = min(r for r, _ in tmpl)
            sc = min(c for _, c in tmpl)
            er = max(r for r, _ in tmpl)
            ec = max(c for _, c in tmpl)
            if er - sr != ec - sc:
                continue
            s = er - sr + 1
            pat = {(r - sr, c - sc): int(I[r, c]) for r, c in tmpl}
            jobs = []
            ok = True
            nchanged = 0
            for comp in comps(O, cand):
                if not any(I[r, c] != O[r, c] for r, c in comp):
                    continue          # the intact template itself
                nchanged += 1
                r0 = min(r for r, _ in comp)
                c0 = min(c for _, c in comp)
                r1 = max(r for r, _ in comp)
                c1 = max(c for _, c in comp)
                if (r1 - r0, c1 - c0) != (s - 1, s - 1):
                    ok = False
                    break
                cpat = {(r - r0, c - c0): int(O[r, c]) for r, c in comp}
                found = None
                for oplist, f in TRANSFORMS:
                    if {f(r, c, s): v for (r, c), v in pat.items()} == cpat:
                        found = oplist
                        break
                if found is None:
                    ok = False
                    break
                jobs.append((r0, c0, found))
            if ok and nchanged > 0:
                plan = (cand, sr, sc, s, jobs)
                break

    if plan is not None:
        bg, sr, sc, s, jobs = plan
        # work outward from the template
        jobs.sort(key=lambda j: ((j[0] - sr) ** 2 + (j[1] - sc) ** 2, j[0], j[1]))

        # grab the one complete motif; bbox selection is intended in full (background included)
        ops.append(28)
        sels.append([sr, sc, s - 1, s - 1])

        for (dr, dc, oplist) in jobs:
            if bg == 0:
                # the stamp is transparent when the background is 0, so it cannot cover the
                # incomplete copy: clear that copy away first, then stamp on the clean square
                hole = [(r, c) for r in range(dr, dr + s) for c in range(dc, dc + s)
                        if int(I[r, c]) != bg]
                if hole:
                    ops.append(bg)
                    sels.append(sel_of(hole))
            ops.append(30)
            sels.append(sel_of([(dr, dc)]))
            for o in oplist:
                # whole square region is meant here: the stamped motif plus its background
                ops.append(o)
                sels.append([dr, dc, s - 1, s - 1])
    else:
        # safety net: complete each copy cell-wise, one copy (and one colour) at a time
        for comp in comps(O, bg):
            diff = [(r, c) for r, c in comp if I[r, c] != O[r, c]]
            if not diff:
                continue
            bycol = {}
            for r, c in diff:
                bycol.setdefault(int(O[r, c]), []).append((r, c))
            for col in sorted(bycol):
                ops.append(col)
                sels.append(sel_of(bycol[col]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
