"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    lim = max(2, min(6, (min(max_h, max_w) - 2) // 2))
    oh = unifint(diff_lb, diff_ub, (2, lim))
    ow = unifint(diff_lb, diff_ub, (2, lim))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nb, ij)) < 4 for nb in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    # the object must not itself be an exact upscale (else the answer is ambiguous)
    for k in range(2, min(oh, ow) + 1):
        if oh % k == 0 and ow % k == 0 and upscale(downscale(go, k), k) == go:
            return generate(diff_lb, diff_ub, max_h, max_w, bgc=bgc, noisec=noisec, objc=objc)
    fac = unifint(diff_lb, diff_ub, (2, max(2, min((max_h - 2) // oh, (max_w - 2) // ow))))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh + 2, max(oh + 2, max_h)))
    w = unifint(diff_lb, diff_ub, (ow + 2, max(ow + 2, max_w)))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = dominant colour of the border ring (the big object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour = the colour whose cells never stand alone (solid upscaled blocks)
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    noisec = rest[0] if rest else None
    if noisec is None and examples:
        for ei, eo in examples:
            ei = np.asarray(ei)
            eb = ei[0, :].tolist() + ei[-1, :].tolist() + ei[:, 0].tolist() + ei[:, -1].tolist()
            ebg = Counter(eb).most_common(1)[0][0]
            vals = [int(v) for v in np.unique(ei) if int(v) not in (ebg, objc)]
            if vals:
                noisec = vals[0]
                break
    if noisec is None:
        noisec = objc

    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    # scale factor n = the largest block size that the object is an exact upscale of
    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        small = M.reshape(H // k, k, W // k, k).any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    ph, pw = H // fac, W // fac

    # ---- a small ARCLE simulator, used only to drop steps that change nothing ----
    def simulate(oo, ss):
        g = np.zeros((30, 30), dtype=int); g[:hi, :wi] = I
        dims = [hi, wi]
        cl = np.zeros((30, 30), dtype=int); cd = [0, 0]
        for o, s in zip(oo, ss):
            m = np.zeros((30, 30), dtype=bool)
            if isinstance(s, dict):
                for r, c in s["cells"]:
                    m[r, c] = True
            else:
                r, c, h, w = s
                m[r:r + h + 1, c:c + w + 1] = True
            if not m.any():
                continue
            rr, cc = np.where(m)
            x0, x1, y0, y1 = rr.min(), rr.max(), cc.min(), cc.max()
            if o <= 9:
                g[m] = o
            elif 10 <= o <= 19:
                if m.sum() == 1 and x0 < dims[0] and y0 < dims[1]:
                    oldc, newc = g[x0, y0], o - 10
                    reg = {(x0, y0)}; q = [(x0, y0)]
                    while q:
                        r, c = q.pop()
                        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                            if 0 <= nr < dims[0] and 0 <= nc < dims[1] and g[nr, nc] == oldc \
                                    and (nr, nc) not in reg:
                                reg.add((nr, nc)); q.append((nr, nc))
                    for r, c in reg:
                        g[r, c] = newc
            elif o == 29:
                if x1 < dims[0] and y1 < dims[1]:
                    cl = np.zeros((30, 30), dtype=int)
                    cl[:x1 - x0 + 1, :y1 - y0 + 1] = np.where(m[x0:x1 + 1, y0:y1 + 1],
                                                             g[x0:x1 + 1, y0:y1 + 1], 0)
                    cd = [x1 - x0 + 1, y1 - y0 + 1]
            elif o == 30:
                h, w = min(cd[0], 30 - x0), min(cd[1], 30 - y0)
                patch = cl[:h, :w]
                tgt = g[x0:x0 + h, y0:y0 + w]
                tgt[patch != 0] = patch[patch != 0]
            elif o == 33:
                h, w = x1 - x0 + 1, y1 - y0 + 1
                patch = np.where(m[x0:x1 + 1, y0:y1 + 1], g[x0:x1 + 1, y0:y1 + 1], 0)
                g = np.zeros((30, 30), dtype=int); g[:h, :w] = patch
                dims = [h, w]
        return g[:dims[0], :dims[1]].copy()

    def pieces_fill(oo, ss):
        # the object takes the noise colour, one connected piece at a time
        oo, ss = list(oo), list(ss)
        S = simulate(oo, ss)
        if noisec != objc:
            cs = {(r, c) for r in range(ph) for c in range(pw) if S[r, c] == objc}
            while cs:
                st = min(cs); q = [st]; cs.discard(st)
                while q:
                    r, c = q.pop()
                    for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if nb in cs:
                            cs.discard(nb); q.append(nb)
                oo.append(10 + int(noisec)); ss.append([st[0], st[1], 0, 0])
        return oo, ss

    def build(early_wipe, late_wipe, early_recolor=False, absolute=False):
        oy, ox = (r0, c0) if absolute else (0, 0)
        ops, sels = [], []
        G = I[r0:r1 + 1, c0:c1 + 1].copy()       # python-side mirror of the object box

        # 1. inside the object's box, noise is just background: wipe it there
        if early_wipe:
            nb_ = [(r, c) for r in range(H) for c in range(W) if G[r, c] == noisec and noisec != objc]
            if not nb_ and not early_recolor:
                return None
            if nb_:
                ops.append(int(bgc)); sels.append(sel_of([(r0 + r, c0 + c) for r, c in nb_]))
            for r, c in nb_:
                G[r, c] = bgc
        if early_recolor:
            # ... and the object takes the noise colour right away
            oc_ = [(r, c) for r in range(H) for c in range(W) if G[r, c] == objc]
            ops.append(int(noisec)); sels.append(sel_of([(r0 + r, c0 + c) for r, c in oc_]))
            for r, c in oc_:
                G[r, c] = noisec

        # 2. crop the canvas to the object's bounding box (full rectangle = the box)
        if not absolute:
            ops.append(33); sels.append([r0, c0, H - 1, W - 1])
        n_fixed = len(ops)

        # 3. keep every n-th row, then every n-th column: copy line k*n onto line k
        clip = [None]

        def compact(axis, count, length):
            for k in range(1, count):
                s = k * fac
                if axis == 0:
                    src = G[s, :length].copy(); cur = G[k, :length]
                    cell = lambda t: (k, t)
                    srect = [oy + s, ox, 0, length - 1]          # the whole source row
                else:
                    src = G[:length, s].copy(); cur = G[:length, k]
                    cell = lambda t: (t, k)
                    srect = [oy, ox + s, length - 1, 0]          # the whole source column
                clr = [cell(t) for t in range(length) if src[t] == 0 and cur[t] != 0]
                need_paste = any(src[t] != 0 and cur[t] != src[t] for t in range(length))
                if clr:                                    # 0s do not paste: write them directly
                    ops.append(0); sels.append(sel_of([(r + oy, c + ox) for r, c in clr]))
                    for (r, c) in clr:
                        G[r, c] = 0
                if need_paste:
                    key = (axis, tuple(src.tolist()))
                    if clip[0] != key:
                        ops.append(29); sels.append(srect)
                        clip[0] = key
                    ops.append(30); sels.append([oy + cell(0)[0], ox + cell(0)[1], 0, 0])
                    for t in range(length):
                        if src[t] != 0:
                            r, c = cell(t)
                            G[r, c] = src[t]

        compact(0, ph, W)
        compact(1, pw, ph)
        # cut the canvas down to the compacted ph x pw block
        ops.append(33); sels.append([oy, ox, ph - 1, pw - 1])

        # drop compaction steps whose removal leaves the small grid unchanged
        keep = [k < n_fixed or k == len(ops) - 1 for k in range(len(ops))]

        def norm(g):                 # with a later wipe, noise there counts as background
            g = g.copy()
            if late_wipe and noisec != objc:
                g[g == noisec] = bgc
            return g

        target = simulate(ops, sels)
        ntarget = norm(target)
        changed = True
        while changed:
            changed = False
            for k in range(len(ops)):
                if keep[k]:
                    continue
                oo = ops[:k] + ops[k + 1:]; ss = sels[:k] + sels[k + 1:]
                if np.array_equal(norm(simulate(oo, ss)), ntarget):
                    ops, sels, keep = oo, ss, keep[:k] + keep[k + 1:]
                    changed = True
                    break

        # noise that reached the small grid -> background
        if late_wipe:
            target = simulate(ops, sels)
            nzs = [(r, c) for r in range(ph) for c in range(pw) if target[r, c] == noisec and noisec != objc]
            if nzs:
                ops.append(int(bgc)); sels.append(sel_of(nzs))
        return pieces_fill(ops, sels)

    def lean(oo, ss):
        fin = simulate(oo, ss)
        for k in range(len(oo)):
            if np.array_equal(simulate(oo[:k] + oo[k + 1:], ss[:k] + ss[k + 1:]), fin):
                return False
        return True

    P = M[::fac, ::fac]
    expect = np.where(P, noisec, bgc)

    def prune(oo, ss):
        # drop any step whose removal leaves the finished grid unchanged
        # (the crops are the frame of the reading and are never dropped)
        fin = simulate(oo, ss)
        changed = True
        while changed:
            changed = False
            for k in range(len(oo)):
                if oo[k] == 33:
                    continue
                o2 = oo[:k] + oo[k + 1:]; s2 = ss[:k] + ss[k + 1:]
                if np.array_equal(simulate(o2, s2), fin):
                    oo, ss = o2, s2
                    changed = True
                    break
        return oo, ss

    if P.all():
        # a solid object: every n-th line looks the same, so its top-left
        # ph x pw corner already is the answer's shape
        oo = [33, 10 + int(noisec)]
        ss = [[r0, c0, ph - 1, pw - 1], [0, 0, 0, 0]]
        if noisec == objc:
            oo, ss = oo[:1], ss[:1]
        return oo + [34], ss + [[0, 0, ph - 1, pw - 1]]

    best = None
    variants = [(ew, lw, er, ab) for ab in (False, True)
                for ew, lw, er in ((True, False, False), (False, True, False),
                                   (False, False, False), (True, False, True))]
    for ew, lw, er, ab in variants:
        if er and (noisec == objc or noisec == 0):
            continue
        cand = build(ew, lw, er, ab)
        if cand is None or not np.array_equal(simulate(*cand), expect):
            continue
        cand = prune(*cand)
        if best is None:
            best = cand
        if lean(*cand):
            best = cand
            break
    ops, sels = best

    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
