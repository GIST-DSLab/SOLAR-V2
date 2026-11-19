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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
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
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading: the wallpaper repeats on a lattice of translations (Sr rows,
    Sc cols).  Every cell equals every other cell of its residue class
    (i mod Sr, j mod Sc), so a hidden cell is recovered from any visible
    copy of itself.  Punch the occluding colour out (Color0 -> holes), then
    lay translated copies of the grid onto itself, one lattice vector at a
    time (CopyO the overlap rectangle, Paste it shifted by (k*Sr, m*Sc)):
    holes are 0 and carry nothing, visible cells land on equal cells, so
    only holes change.  Only if some residue class is hidden in EVERY copy
    do we fall back to the mirror folds of the wallpaper.  O is never read."""
    import numpy as np
    from collections import deque
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def periods(ok, axis):
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

    def ncomp(mask):
        seen = np.zeros_like(mask)
        n = 0
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    n += 1
                    q = deque([(r, c)]); seen[r, c] = True
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < hi and 0 <= xx < wi and mask[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = True; q.append((yy, xx))
        return n

    # ── the occluding colour ──────────────────────────────────────────────
    noise = None
    if examples:
        cand = None
        for a, b in examples:
            a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
            s = set(np.unique(a).tolist()) - set(np.unique(b).tolist())
            if s:
                cand = s if cand is None else (cand & s or cand)
        if cand:
            inI = [c for c in cand if (I == c).any()]
            if len(inI) == 1:
                noise = inI[0]
    if noise is None:
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            pr = (periods(~m, 0) or [hi])[0]
            pc = (periods(~m, 1) or [wi])[0]
            if pr >= hi or pc >= wi:
                continue
            key = (pr * pc, ncomp(m), int(m.sum()), col)
            if best is None or key < best:
                best = key
        noise = best[3] if best is not None else None
    ops, sels = [], []
    if noise is None or not (I == noise).any():
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    mask = (I == noise)
    g = I.copy()
    g[mask] = 0
    known = ~mask
    base = known.copy()

    # all cells of the occluding colour -> holes
    if noise != 0:
        ops.append(0); sels.append(sel_of([(int(r), int(c)) for r, c in zip(*np.where(mask))]))

    # ── the translation lattice ───────────────────────────────────────────
    Sr = (periods(base, 0) or [hi])[0]
    Sc = (periods(base, 1) or [wi])[0]

    def shift_paste(dr, dc):
        ch, cw = hi - abs(dr), wi - abs(dc)
        if ch <= 0 or cw <= 0:
            return False
        r0, c0 = max(0, -dr), max(0, -dc)
        tr, tc = max(0, dr), max(0, dc)
        src = g[r0:r0 + ch, c0:c0 + cw]
        sk = known[r0:r0 + ch, c0:c0 + cw]
        tk = known[tr:tr + ch, tc:tc + cw]
        gain = (~tk) & sk
        if not gain.any():
            return False
        fill = gain & (src != 0)
        if fill.any():
            g[tr:tr + ch, tc:tc + cw][fill] = src[fill]
            ops.append(29); sels.append([r0, c0, ch - 1, cw - 1])
            ops.append(30); sels.append([tr, tc, 0, 0])
        known[tr:tr + ch, tc:tc + cw] |= gain
        return True

    vecs = []
    for k in range(-(hi // Sr), hi // Sr + 1):
        for m in range(-(wi // Sc), wi // Sc + 1):
            if (k, m) != (0, 0) and abs(k * Sr) < hi and abs(m * Sc) < wi:
                vecs.append((abs(k) + abs(m), abs(k), k, m))
    vecs.sort()
    for _ in range(4):
        if known.all():
            break
        moved = False
        for _, _, k, m in vecs:
            if known.all():
                break
            if shift_paste(k * Sr, m * Sc):
                moved = True
        if not moved:
            break

    # ── fallback: classes hidden in every copy -> fold across mirror lines ─
    def fold_lines(vertical):
        n = hi if vertical else wi
        out = []
        for A in range(1, 2 * n - 2):
            a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
            if a1 - a0 < 1:
                continue
            if vertical:
                gv, kv = I[a0:a1 + 1, :], base[a0:a1 + 1, :]
            else:
                gv, kv = I[:, a0:a1 + 1].T, base[:, a0:a1 + 1].T
            both = kv & kv[::-1]
            if both.sum() >= 2 * (wi if vertical else hi) and not (both & (gv != gv[::-1])).any():
                out.append(A)
        return out

    def fold(A, vertical):
        n = hi if vertical else wi
        a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
        if vertical:
            gv, kv = g[a0:a1 + 1, :], known[a0:a1 + 1, :]
        else:
            gv, kv = g[:, a0:a1 + 1].T, known[:, a0:a1 + 1].T
        mg, mk = gv[::-1], kv[::-1]
        gain = (~kv) & mk
        if not gain.any():
            return False
        fill = gain & (mg != 0)
        if fill.any():
            gv[fill] = mg[fill]
            if vertical:
                sel = [a0, 0, a1 - a0, wi - 1]
                ops.extend([29, 27, 30]); sels.extend([sel, sel, [a0, 0, 0, 0]])
            else:
                sel = [0, a0, hi - 1, a1 - a0]
                ops.extend([29, 26, 30]); sels.extend([sel, sel, [0, a0, 0, 0]])
        kv |= gain
        return True

    if not known.all():
        frs, fcs = fold_lines(True), fold_lines(False)
        for _ in range(6):
            if known.all():
                break
            moved = False
            for A in frs:
                moved |= fold(A, True)
            for B in fcs:
                moved |= fold(B, False)
            for _, _, k, m in vecs:
                moved |= shift_paste(k * Sr, m * Sc)
            if not moved:
                break

    # ── drop any op the finished grid does not depend on (own result, not O)
    def cells_of(s):
        if isinstance(s, dict):
            return [tuple(x) for x in s["cells"]]
        r, c, h, w = s
        return [(y, x) for y in range(r, r + h + 1) for x in range(c, c + w + 1)]

    def run(o_list, s_list):
        gg = I.copy()
        clip = None
        for op, s in zip(o_list, s_list):
            if op <= 9:
                for y, x in cells_of(s):
                    gg[y, x] = op
            elif op == 29:
                r, c, h, w = s
                clip = gg[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30 and clip is not None:
                r, c = s[0], s[1]
                p = clip[:hi - r, :wi - c]
                tgt = gg[r:r + p.shape[0], c:c + p.shape[1]]
                np.copyto(tgt, p, where=(p > 0))
            elif op in (26, 27):
                r, c, h, w = s
                reg = gg[r:r + h + 1, c:c + w + 1]
                reg[:] = np.fliplr(reg) if op == 26 else np.flipud(reg)
        return gg

    target = run(ops, sels)
    changed = True
    while changed:
        changed = False
        # whole copy-paste groups first (CopyO [+Flip] + Paste)
        k = 0
        while k < len(ops):
            if ops[k] == 29:
                e = k + 1
                while e < len(ops) and ops[e] != 30:
                    e += 1
                no, ns = ops[:k] + ops[e + 1:], sels[:k] + sels[e + 1:]
                if (run(no, ns) == target).all():
                    ops, sels = no, ns
                    changed = True
                    continue
            k += 1
        k = 0
        while k < len(ops):
            no, ns = ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]
            if (run(no, ns) == target).all():
                ops, sels = no, ns
                changed = True
                continue
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
