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
from collections import deque
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """
    The generator samples bgc + noisec (disjoint) and a palette `ccols` of
    pattern colours (noisec excluded).  The rule is "repair the wallpaper where
    the damage colour covers it", so the DAMAGE colour must be identical across
    the whole episode — otherwise a solver cannot tell which colour is damage.
    bgc and the pattern palette are fixed too, so every instance of the episode
    shares one colour scheme.  No discrete structural variants exist (the rule
    is the same for every block size / rotation), so no instance_plan is needed.
    """
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, noisec=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)

    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    if ccols is None:
        numc = unifint(diff_lb, diff_ub, (2, 9))
        ccols = sample(remove(noisec, cols), numc)
    ccols = [c for c in ccols if c != noisec]
    if len(ccols) < 2:
        ccols = sample(remove(noisec, cols), 2)

    hlo, wlo = min(10, max_h), min(10, max_w)
    for _attempt in range(30):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))

        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(choice(ccols), ij) for ij in pinds})

        go = canvas(bgc, (h, w))
        locs = set()
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci, locj = hp * a, wp * b
                locs.add((loci, locj))
                mf1 = identity if a % 2 == 0 else hmirror
                mf2 = identity if b % 2 == 0 else vmirror
                mf = compose(mf1, mf2)
                go = paint(go, shift(mf(pobj), (loci, locj)))

        numpatches = unifint(diff_lb, diff_ub, (1, max(1, int((h * w) ** 0.5 // 2))))
        gi = tuple(e for e in go)
        places = apply(lbind(shift, pinds), locs)
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = randint(2, min(6, h))
            pw = randint(2, min(6, w))
            loci = randint(0, h - ph)
            locj = randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            candset = apply(normalize, apply(rbind(toobject, gi2), places))
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 \
                    and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 \
                    and (pobj in candset or hmirror(pobj) in candset
                         or vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
                succ += 1
                gi = gi2
        if gi != go:                       # need at least one damaged patch
            break

    # rot90/rot270 swap the dimensions -> only allow them when that still fits
    if h <= max_w and w <= max_h:
        rotf = choice((identity, rot90, rot180, rot270))
    else:
        rotf = choice((identity, rot180))
    return {'input': rotf(gi), 'output': rotf(go)}


# ------------------------------------------------------------- operations ---
def _components(mask, h, w):
    """4-connected components of a boolean mask."""
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    a, b = q.popleft(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True; q.append((na, nb))
                comps.append(cells)
    return comps


def _compat(lines, noise):
    """pairwise "these two lines agree wherever neither is damaged" matrix"""
    n = len(lines)
    mat = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            m = (a != noise) & (b != noise)
            ok = bool(np.all(a[m] == b[m]))
            mat[i][j] = mat[j][i] = ok
    return mat


def _fold(n, s, o):
    """index -> block index for a mirrored tiling of block size s, phase o"""
    f = []
    for r in range(n):
        x = (r - o) % (2 * s)
        f.append(x if x < s else 2 * s - 1 - x)
    return f


def _options(n, mat, levels=3):
    """mirrored-tilings (s, o) compatible with `mat`; smallest block first"""
    out, lv = [], 0
    for s in range(1, max(1, n // 2) + 1):
        found = []
        for o in range(2 * s):
            f = _fold(n, s, o)
            groups = {}
            for r, x in enumerate(f):
                groups.setdefault(x, []).append(r)
            ok = True
            for idxs in groups.values():
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        if not mat[idxs[i]][idxs[j]]:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if ok:
                found.append((s, o, f))
        if found:
            out.extend(found); lv += 1
            if lv >= levels:
                break
    return out


def _hypothesis(I, noise, hi, wi):
    """
    Hypothesis: `noise` is the damage colour.  The wallpaper is a block of
    sr x sc mirrored in both directions (exactly how the generator builds it),
    so rows fold to sr classes and columns to sc classes.  Accept only if all
    UNDAMAGED cells of a class carry one single colour and every damaged cell
    lands in a class that has an undamaged witness.
    Returns (block_size, row_fold, col_fold, class_value) or None.
    """
    ropts = _options(hi, _compat([I[r] for r in range(hi)], noise))
    copts = _options(wi, _compat([I[:, c] for c in range(wi)], noise))
    if not ropts or not copts:
        return None
    combos = sorted([(a, b) for a in ropts for b in copts], key=lambda t: t[0][0] * t[1][0])
    for (sr, orr, fr), (sc, oc, fc) in combos:
        val, ok = {}, True
        for r in range(hi):
            for c in range(wi):
                v = int(I[r, c])
                if v == noise:
                    continue
                k = (fr[r], fc[c])
                if k in val:
                    if val[k] != v:
                        ok = False; break
                else:
                    val[k] = v
            if not ok: break
        if not ok:
            continue
        if any((fr[r], fc[c]) not in val
               for r in range(hi) for c in range(wi) if I[r, c] == noise):
            continue
        return sr * sc, fr, fc, val
    return None


def derive_operations(I, O):
    """
    Everything is measured from I: which colour is the damage, the mirrored
    wallpaper period, and the colour each damaged cell must take.  O is never
    consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ---- 1. which colour is the damage?  Test every colour as a hypothesis and
    #         keep the cheapest description (MDL): the wallpaper block plus the
    #         damage blobs.  A real pattern colour is either huge or scattered
    #         into many components, so it always loses.
    best = None
    for col in sorted(set(I.flatten().tolist())):
        res = _hypothesis(I, col, hi, wi)
        if res is None:
            continue
        score, fr, fc, val = res
        ncell = int((I == col).sum())
        ncomp = len(_components(I == col, hi, wi))
        key = (score + ncell + 4 * ncomp, ncell, col)
        if best is None or key < best[0]:
            best = (key, col, fr, fc, val)
    if best is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    _, noise, fr, fc, val = best

    # ---- 2. repair the wallpaper one damaged blob at a time; inside a blob
    #         paint one colour at a time (each colour = the wallpaper value of
    #         the block cell those cells fold onto).
    patches = _components(I == noise, hi, wi)
    patches.sort(key=lambda cells: min(cells))
    for cells in patches:
        by_color = {}
        for (r, c) in sorted(cells):
            t = val.get((fr[r], fc[c]))
            if t is None or t == noise:
                continue
            by_color.setdefault(t, []).append((r, c))
        for t in sorted(by_color, key=lambda k: by_color[k][0]):
            ops.append(t)                       # Color<t>
            sels.append(sel_of(by_color[t]))    # exactly the damaged cells

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
