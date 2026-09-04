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
from random import randint
try:
    from dsl import *
    from utils import *
except ImportError:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec) -> dict:
    cols = interval(0, 10, 1)
    lim = max(10, min(max_h, max_w))          # rot90 at the end may swap h/w
    h = unifint(diff_lb, diff_ub, (10, lim))
    w = unifint(diff_lb, diff_ub, (10, lim))
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
    maxtr = 10 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if (len(sfilter(gi2, lambda r: noisec not in r)) >= 2
                and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2
                and (pobj in candset or hmirror(pobj) in candset
                     or vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset)):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """The grid is one wallpaper pattern: periodic, and mirror-symmetric about
    evenly spaced fold lines.  Rectangular patches of a single intruding colour
    hide parts of it.  So: punch the patches out (Color0 -> holes), then let the
    surviving pattern flow into the holes -- slide a transparent copy of the grid
    onto itself by one period (CopyO + Paste; the holes are 0, so they neither
    travel nor overwrite), and fold the grid onto itself across each mirror line
    (CopyO + Flip + Paste re-merges the two halves).  Holes whose true colour is
    0 simply stay 0.  Every quantity is measured from I."""
    import numpy as np
    from collections import deque

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(mask):
        seen = np.zeros((hi, wi), dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)]); seen[r, c] = True; n = 0
                    while q:
                        y, x = q.popleft(); n += 1
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; q.append((ny, nx))
                    out.append(n)
        return out

    def periods(ok, axis):
        # every step at which all pairs of trusted cells that far apart agree
        n = I.shape[axis]
        out = []
        for p in range(1, n):
            if axis == 0:
                a, b, ka, kb = I[p:], I[:n - p], ok[p:], ok[:n - p]
            else:
                a, b, ka, kb = I[:, p:], I[:, :n - p], ok[:, p:], ok[:, :n - p]
            if not ((a != b) & ka & kb).any():
                out.append(p)
        return out

    def period(ok, axis):
        p = periods(ok, axis)
        return p[0] if p else I.shape[axis]

    # ── which colour is the intruder: the one whose removal leaves the
    #    tightest doubly periodic pattern behind ────────────────────────────
    best = None
    for col in sorted(set(I.flatten().tolist())):
        m = (I == col)
        # an intruder never covers everything: it leaves clean rows and columns
        if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
            continue
        pr, pc = period(~m, 0), period(~m, 1)
        if pr >= hi or pc >= wi:
            continue
        key = (pr * pc, len(components(m)), int(m.sum()), col)
        if best is None or key < best:
            best = key
    noise = best[3] if best is not None else int(I[0, 0])

    mask = (I == noise)

    ops, sels = [], []
    g = I.copy()
    known = ~mask

    # ── punch out every damaged patch (maximal rectangles of intruder) ──────
    left = mask.copy()
    for r in range(hi):
        for c in range(wi):
            if not left[r, c]:
                continue
            w = 0
            while c + w < wi and left[r, c + w]:
                w += 1
            h = 1
            while r + h < hi and left[r + h, c:c + w].all():
                h += 1
            left[r:r + h, c:c + w] = False
            g[r:r + h, c:c + w] = 0
            if noise != 0:
                ops.append(0); sels.append([r, c, h - 1, w - 1])

    # ── slide the grid onto itself by one period ───────────────────────────
    def slide(dr, dc):
        ch, cw = hi - abs(dr), wi - abs(dc)
        if ch <= 0 or cw <= 0:
            return False
        r0, c0 = max(0, -dr), max(0, -dc)
        tr, tc = max(0, dr), max(0, dc)
        src, sk = g[r0:r0 + ch, c0:c0 + cw], known[r0:r0 + ch, c0:c0 + cw]
        tk = known[tr:tr + ch, tc:tc + cw]
        fill = (~tk) & sk & (src != 0)
        zero = (~tk) & sk & (src == 0)          # partner says this hole is a 0: already right
        if not (fill.any() or zero.any()):
            return False
        if fill.any():
            g[tr:tr + ch, tc:tc + cw][fill] = src[fill]
            ops.append(29); sels.append([r0, c0, ch - 1, cw - 1])
            ops.append(30); sels.append([tr, tc, 0, 0])
        known[tr:tr + ch, tc:tc + cw] |= (fill | zero)
        return True

    # ── fold the grid across a mirror line (rows r, A-r swap) ──────────────
    def fold(A, vertical):
        n = hi if vertical else wi
        a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
        if a1 - a0 < 1:
            return False
        if vertical:
            gv, kv = g[a0:a1 + 1, :], known[a0:a1 + 1, :]
        else:
            gv, kv = g[:, a0:a1 + 1].T, known[:, a0:a1 + 1].T
        mg, mk = gv[::-1], kv[::-1]
        if ((kv & mk) & (gv != mg)).any():
            return False                       # not a real mirror line here
        fill = (~kv) & mk & (mg != 0)
        zero = (~kv) & mk & (mg == 0)
        if not fill.any():
            if zero.any():
                kv |= zero
                return True
            return False
        gv[fill] = mg[fill]
        kv |= (fill | zero)
        # Flip alone would only swap the holes across the line; the Paste of the
        # pre-flip copy puts the untouched half back, so the two halves merge.
        if vertical:
            sel = [a0, 0, a1 - a0, wi - 1]
            ops.extend([29, 27, 30]); sels.extend([sel, sel, [a0, 0, 0, 0]])
        else:
            sel = [0, a0, hi - 1, a1 - a0]
            ops.extend([29, 26, 30]); sels.extend([sel, sel, [0, a0, 0, 0]])
        return True

    # ── the pattern repeats every period AND folds across mirror lines spaced
    #    half a period apart; take the tightest period the untouched cells
    #    support that really does own such mirror lines ─────────────────────
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

    def measure(vertical):
        axis = 0 if vertical else 1
        cands = periods(base, axis) or [I.shape[axis]]
        for p in cands:
            lines = fold_lines(p, vertical)
            if lines:
                return p, lines
        return cands[0], []

    base = ~mask
    Pr, fold_rows = measure(True)
    Pc, fold_cols = measure(False)

    for _ in range(10):
        if known.all():
            break
        moved = False
        for dr, dc in ((Pr, 0), (-Pr, 0), (0, Pc), (0, -Pc)):
            if slide(dr, dc):
                moved = True
        if known.all():
            break
        for A in fold_rows:
            if fold(A, True):
                moved = True
        for B in fold_cols:
            if fold(B, False):
                moved = True
        if not moved:
            break

    # ── drop any op the final grid does not depend on ──────────────────────
    def run(o_list, s_list):
        gg = I.copy()
        clip = None
        for op, (r, c, h, w) in zip(o_list, s_list):
            if op <= 9:
                gg[r:r + h + 1, c:c + w + 1] = op
            elif op == 29:
                clip = gg[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30 and clip is not None:
                p = clip[:hi - r, :wi - c]
                tgt = gg[r:r + p.shape[0], c:c + p.shape[1]]
                np.copyto(tgt, p, where=(p > 0))
            elif op in (26, 27):
                reg = gg[r:r + h + 1, c:c + w + 1]
                f = np.fliplr(reg) if op == 26 else np.flipud(reg)
                reg[:] = np.where(f > 0, f, 0)
        return gg

    target = run(ops, sels)
    k = 0
    while k < len(ops):
        if (run(ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]) == target).all():
            ops.pop(k); sels.pop(k)
            k = 0
        else:
            k += 1

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
