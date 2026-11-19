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
from collections import deque
import numpy as np


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None) -> dict:
    from dsl import (interval, asindices, canvas, remove,
                     paint, shift, hmirror, vmirror, identity, compose, apply,
                     lbind, rbind, backdrop, fill, toobject, normalize, sfilter,
                     dmirror, rot90, rot180, rot270)
    from utils import unifint
    from random import sample, choice, randint
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
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
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
           len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and \
           (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset
                or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    """Reading: the whole grid is ONE doubly periodic wallpaper with translation
    period (Pr, Pc); patches of a single intruder colour hide parts of it.
    Every cell equals every other cell in its (r mod Pr, c mod Pc) class, so:
      1. punch each intruder patch out (Color0 -> holes),
      2. slide transparent copies of the grid onto itself by lattice vectors
         (k*Pr, m*Pc) -- CopyO of the overlap rectangle + Paste at the shifted
         origin.  Holes are 0, so they neither travel nor overwrite; every
         surviving class representative flows into the holes of its class.
    The lattice of translations carries the pattern; a mirror fold (CopyO +
    Flip + Paste across a fold line half a period away) is used only for a
    residue class whose every lattice copy is hidden.
    The intruder colour is read from the demonstrations (the colour present in
    their inputs but never in their outputs); without them, it is the colour
    whose removal leaves the tightest periodic pattern."""
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def periods_of(ok, axis):
        n = I.shape[axis]
        for p in range(1, n):
            if axis == 0:
                a, b, ka, kb = I[p:], I[:n - p], ok[p:], ok[:n - p]
            else:
                a, b, ka, kb = I[:, p:], I[:, :n - p], ok[:, p:], ok[:, :n - p]
            if not ((a != b) & ka & kb).any():
                return p
        return n

    # ── intruder colour ────────────────────────────────────────────────────
    noise = None
    if examples:
        in_cols, out_cols = set(), set()
        for ei, eo in examples:
            in_cols |= set(np.asarray(ei).flatten().tolist())
            out_cols |= set(np.asarray(eo).flatten().tolist())
        cand = sorted(in_cols - out_cols)
        if len(cand) == 1:
            noise = cand[0]
    if noise is None:
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            pr, pc = periods_of(~m, 0), periods_of(~m, 1)
            if pr >= hi or pc >= wi:
                continue
            key = (pr * pc, int(m.sum()), col)
            if best is None or key < best:
                best = key
        noise = best[2] if best is not None else int(I[0, 0])

    mask = (I == noise)
    known = ~mask
    Pr = periods_of(known, 0)
    Pc = periods_of(known, 1)

    ops, sels = [], []
    g = I.copy()

    # ── 1. punch out each intruder patch (one connected patch at a time) ───
    if noise != 0 and mask.any():
        seen = np.zeros_like(mask)
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)]); seen[r, c] = True; cells = []
                    while q:
                        y, x = q.popleft(); cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; q.append((ny, nx))
                    ops.append(0); sels.append(sel_of(cells))
                    for y, x in cells:
                        g[y, x] = 0

    # ── 2. slide the grid onto itself by lattice vectors ───────────────────
    vecs = []
    for k in range(-(hi // Pr), hi // Pr + 1):
        for m in range(-(wi // Pc), wi // Pc + 1):
            dr, dc = k * Pr, m * Pc
            if (dr, dc) == (0, 0) or abs(dr) >= hi or abs(dc) >= wi:
                continue
            vecs.append((abs(k) + abs(m), abs(k), abs(m), dr, dc))
    vecs.sort()

    def slide(dr, dc):
        ch, cw = hi - abs(dr), wi - abs(dc)
        r0, c0 = max(0, -dr), max(0, -dc)
        tr, tc = max(0, dr), max(0, dc)
        src = g[r0:r0 + ch, c0:c0 + cw]
        sk = known[r0:r0 + ch, c0:c0 + cw]
        tk = known[tr:tr + ch, tc:tc + cw]
        learn = (~tk) & sk
        if not learn.any():
            return False
        fillm = learn & (src != 0)
        known[tr:tr + ch, tc:tc + cw] |= learn
        if fillm.any():
            tgt = g[tr:tr + ch, tc:tc + cw]
            tgt[fillm] = src[fillm]
            # CopyO of the whole overlap rectangle, pasted one period over
            ops.append(29); sels.append([r0, c0, ch - 1, cw - 1])
            ops.append(30); sels.append([tr, tc, 0, 0])
        return True

    # ── 3. a residue class the lattice cannot reach (every copy of it hidden)
    #       is still recoverable from the pattern's mirror lines, spaced half
    #       a period apart: fold the grid across such a line ─────────────────
    base = I != noise

    def fold_lines(P, vertical):
        n = hi if vertical else wi
        best_cls = None
        for phase in range(P):
            lines, support, good = [], 0, True
            for A in range(phase, 2 * n - 2, P):
                a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
                if a1 - a0 < 1:
                    continue
                if vertical:
                    gv, kv = I[a0:a1 + 1, :], base[a0:a1 + 1, :]
                else:
                    gv, kv = I[:, a0:a1 + 1].T, base[:, a0:a1 + 1].T
                both = kv & kv[::-1]
                if (both & (gv != gv[::-1])).any():
                    good = False
                    break
                support += int(both.sum())
                lines.append(A)
            if good and support and (best_cls is None or support > best_cls[0]):
                best_cls = (support, lines)
        return best_cls[1] if best_cls else []

    def fold(A, vertical):
        n = hi if vertical else wi
        a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
        if a1 - a0 < 1:
            return False
        if vertical:
            gv, kv = g[a0:a1 + 1, :], known[a0:a1 + 1, :]
        else:
            gv, kv = g[:, a0:a1 + 1].T, known[:, a0:a1 + 1].T
        mg, mk = gv[::-1].copy(), kv[::-1].copy()
        learn = (~kv) & mk
        if not learn.any():
            return False
        fillm = learn & (mg != 0)
        kv |= learn
        if fillm.any():
            gv[fillm] = mg[fillm]
            # copy the band, mirror it in place, paste the copy back: the two
            # halves merge (holes are 0 and never overwrite)
            if vertical:
                sel = [a0, 0, a1 - a0, wi - 1]
                ops.extend([29, 27, 30]); sels.extend([sel, sel, [a0, 0, 0, 0]])
            else:
                sel = [0, a0, hi - 1, a1 - a0]
                ops.extend([29, 26, 30]); sels.extend([sel, sel, [0, a0, 0, 0]])
        return True

    fold_rows = fold_lines(Pr, True) if Pr < hi else []
    fold_cols = fold_lines(Pc, False) if Pc < wi else []

    for _ in range(10):
        if known.all():
            break
        progress = False
        for _, _, _, dr, dc in vecs:
            if known.all():
                break
            progress |= slide(dr, dc)
        if known.all():
            break
        for A in fold_rows:
            if fold(A, True):
                progress = True
                break
        else:
            for B in fold_cols:
                if fold(B, False):
                    progress = True
                    break
        if not progress:
            break

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
