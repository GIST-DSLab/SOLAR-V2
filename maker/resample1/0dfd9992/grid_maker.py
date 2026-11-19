"""
ARC Task: 0dfd9992 (RE-ARC) — LLM-generated grid_maker
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
    bgc, noisec = random.sample(range(10), 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- noise colour: the convention read from the demonstrations ---
    noisec = None
    if examples:
        exs = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in examples]
        cand = None
        for a, b in exs:
            s = set(np.unique(a).tolist()) - set(np.unique(b).tolist())
            if s:
                cand = s if cand is None else (cand & s if cand & s else cand)
        if cand:
            inI = [c for c in cand if (I == c).any()]
            if inI:
                noisec = min(inI, key=lambda c: (I == c).sum())
    if noisec is None:
        # fallback: the colour forming the biggest solid component (noise rectangles)
        pal = np.unique(I).tolist()
        def maxcomp(col):
            m = (I == col); seen = np.zeros_like(m); best = 0
            for r in range(hi):
                for c in range(wi):
                    if m[r, c] and not seen[r, c]:
                        st = [(r, c)]; seen[r, c] = 1; n = 0
                        while st:
                            y, x = st.pop(); n += 1
                            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                                yy, xx = y+dy, x+dx
                                if 0 <= yy < hi and 0 <= xx < wi and m[yy, xx] and not seen[yy, xx]:
                                    seen[yy, xx] = 1; st.append((yy, xx))
                    best = max(best, n) if m[r, c] else best
            return best
        noisec = max(pal, key=lambda c: (maxcomp(c), -(I == c).sum()))

    known = I != noisec

    def period(axis):
        n = I.shape[axis]
        for p in range(1, n):
            ok = True
            if axis == 0:
                a, b = I[:-p], I[p:]; ka, kb = known[:-p], known[p:]
            else:
                a, b = I[:, :-p], I[:, p:]; ka, kb = known[:, :-p], known[:, p:]
            both = ka & kb
            if (a[both] != b[both]).any():
                ok = False
            if ok:
                return p
        return n
    ph, pw = period(0), period(1)

    # mirror axis (s = r + r') consistent with all known data, most overlap wins
    def axis(axis_):
        n = I.shape[axis_]
        A = I if axis_ == 0 else I.T
        K = known if axis_ == 0 else known.T
        best = None
        for s in range(1, 2 * n - 2, 2):  # axes between cells (tile boundaries)
            ov = 0; ok = True
            for r in range(n):
                r2 = s - r
                if r2 <= r or r2 >= n:
                    continue
                both = K[r] & K[r2]
                if (A[r][both] != A[r2][both]).any():
                    ok = False; break
                ov += int(both.sum())
            if ok and ov > 0 and (best is None or ov > best[0]):
                best = (ov, s)
        return None if best is None else best[1]
    sr_ax, sc_ax = axis(0), axis(1)

    def orbit1(x, p, s, n):
        out = set()
        for base in ([x] + ([s - x] if s is not None else [])):
            for k in range(-(n // p) - 2, n // p + 3):
                y = base + k * p
                if 0 <= y < n:
                    out.add(y)
        return out

    def value_at(r, c):
        best = None
        for rr in orbit1(r, ph, sr_ax, hi):
            for cc in orbit1(c, pw, sc_ax, wi):
                if known[rr, cc]:
                    d = abs(rr - r) + abs(cc - c)
                    if best is None or d < best[0]:
                        best = (d, int(I[rr, cc]))
        return None if best is None else best[1]

    # noise components (8-connected)
    seen = np.zeros_like(known)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if not known[r, c] and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = 1; cells = []
                while st:
                    y, x = st.pop(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y+dy, x+dx
                            if 0 <= yy < hi and 0 <= xx < wi and not known[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = 1; st.append((yy, xx))
                comps.append(cells)

    def starts(lo, hi_, p, s, n, flip):
        # possible source start positions of a [lo, hi_] span
        size = hi_ - lo + 1
        res = []
        if not flip:
            bases = [lo]
        else:
            if s is None:
                return res
            bases = [s - hi_]
        for b in bases:
            for k in range(-(n // p) - 2, n // p + 3):
                st0 = b + k * p
                if 0 <= st0 and st0 + size <= n and not (st0 == lo and not flip):
                    res.append(st0)
                elif 0 <= st0 and st0 + size <= n and flip:
                    res.append(st0)
        return res

    G = I.copy()
    ops, sels = [], []
    for cells in comps:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        best = None
        for fr in (False, True):
            for fc in (False, True):
                rstarts = starts(r0, r1, ph, sr_ax, hi, fr) if fr else [r0 + k * ph for k in range(-(hi // ph) - 2, hi // ph + 3) if 0 <= r0 + k * ph and r0 + k * ph + bh <= hi]
                cstarts = starts(c0, c1, pw, sc_ax, wi, fc) if fc else [c0 + k * pw for k in range(-(wi // pw) - 2, wi // pw + 3) if 0 <= c0 + k * pw and c0 + k * pw + bw <= wi]
                for a in rstarts:
                    for b in cstarts:
                        if (a, b) == (r0, c0) and not fr and not fc:
                            continue
                        if not known[a:a+bh, b:b+bw].all():
                            continue
                        key = (int(fr) + int(fc), abs(a - r0) + abs(b - c0))
                        if best is None or key < best[0]:
                            best = (key, a, b, fr, fc)
        if best is not None:
            _, a, b, fr, fc = best
            srcb = I[a:a+bh, b:b+bw]
            newG = G.copy()
            reg = newG[r0:r0+bh, c0:c0+bw]
            reg[srcb != 0] = srcb[srcb != 0]
            if (newG != G).any():
                # copy the clean symmetric partner of the damaged rectangle (full rect intended)
                ops.append(28); sels.append([a, b, bh - 1, bw - 1])
                ops.append(30); sels.append([r0, c0, 0, 0])
                G = newG
            rect = [r0, c0, bh - 1, bw - 1]  # whole damaged rectangle, mirrored in place
            for flag, op, sl in ((fr, 27, (slice(None, None, -1), slice(None))),
                                 (fc, 26, (slice(None), slice(None, None, -1)))):
                if flag:
                    sub = G[r0:r0+bh, c0:c0+bw]
                    nsub = sub[sl]
                    if (nsub != sub).any():
                        G[r0:r0+bh, c0:c0+bw] = nsub.copy()
                        ops.append(op); sels.append(rect)
            fix_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        else:
            fix_cells = cells
        # remaining cells (Paste-transparent zeros, leftover noise): paint by symmetry value
        bycol = {}
        for r, c in fix_cells:
            v = int(I[r, c]) if known[r, c] else value_at(r, c)
            if v is not None and G[r, c] != v:
                bycol.setdefault(v, []).append((r, c))
        for v in sorted(bycol):
            ops.append(v); sels.append(sel_of(bycol[v]))
            for r, c in bycol[v]:
                G[r, c] = v

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
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
                                f"for task 0dfd9992"
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
                    f"Failed to build a complete episode for task 0dfd9992 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0dfd9992-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
