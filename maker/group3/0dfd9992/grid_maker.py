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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + diff_lb * (hi - lo)
        b = lo + diff_ub * (hi - lo)
        return random.randint(int(round(a)), int(round(b)))
    cols = list(range(10))
    if bgc is None or noisec is None:
        bgc, noisec = random.sample(cols, 2)
    H = max(10, min(30, max_h)); W = max(10, min(30, max_w))
    h = unifint(10, H)
    w = unifint(10, W)
    hp = unifint(2, h // 2 - 1)
    wp = unifint(2, w // 2 - 1)
    remcols = [c for c in cols if c != noisec]
    numc = unifint(2, 9)
    ccols = random.sample(remcols, numc)
    pobj = np.array([[random.choice(ccols) for _ in range(wp)] for _ in range(hp)])
    go = np.full((h, w), bgc, dtype=int)
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            t = pobj
            if a % 2 == 1:
                t = t[::-1, :]
            if b % 2 == 1:
                t = t[:, ::-1]
            r0, c0 = hp * a, wp * b
            hh, ww = min(hp, h - r0), min(wp, w - c0)
            if hh > 0 and ww > 0:
                go[r0:r0 + hh, c0:c0 + ww] = t[:hh, :ww]
    numpatches = unifint(1, int((h * w) ** 0.5 // 2))
    variants = {pobj.tobytes(), pobj[::-1].tobytes(), pobj[:, ::-1].tobytes(), pobj[::-1, ::-1].tobytes()}
    places = [(hp * a, wp * b) for a in range(h // hp + 1) for b in range(w // wp + 1)]
    gi = go.copy()
    succ = tr = 0
    while succ < numpatches and tr < 5 * numpatches:
        tr += 1
        ph, pw = random.randint(2, 6), random.randint(2, 6)
        li, lj = random.randint(0, h - ph), random.randint(0, w - pw)
        gi2 = gi.copy(); gi2[li:li + ph, lj:lj + pw] = noisec
        ok_rows = int((~(gi2 == noisec).any(axis=1)).sum())
        ok_cols = int((~(gi2 == noisec).any(axis=0)).sum())
        found = False
        for (pi, pj) in places:
            if pi + hp <= h and pj + wp <= w:
                if gi2[pi:pi + hp, pj:pj + wp].tobytes() in variants:
                    found = True; break
        if ok_rows >= 2 and ok_cols >= 2 and found:
            succ += 1; gi = gi2
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k); go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """Same rule as before (the grid is a mirrored wallpaper; rectangular
    patches of one intruding colour hide parts of it), different unit of work:
    instead of punching out every patch and sliding/folding the WHOLE grid
    onto itself, repair one damaged patch at a time.  For each patch pick a
    symmetric twin of its bounding box elsewhere in the grid -- reached by one
    period step and/or a fold across a mirror line -- that is currently free of
    the intruder, copy that twin, paste it over the patch and, if the twin is a
    mirror image, flip the pasted block in place.  Everything is measured from I."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def comp_list(mask):
        seen = np.zeros(mask.shape, dtype=bool)
        out = []
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
                    out.append(cells)
        return out

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

    # ── the intruding colour ──────────────────────────────────────────────
    best = None
    for col in sorted(set(I.flatten().tolist())):
        m = (I == col)
        if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
            continue
        pr = (periods(~m, 0) or [hi])[0]
        pc = (periods(~m, 1) or [wi])[0]
        if pr >= hi or pc >= wi:
            continue
        key = (pr * pc, len(comp_list(m)), int(m.sum()), col)
        if best is None or key < best:
            best = key
    noise = best[3] if best is not None else int(I[0, 0])
    base = I != noise

    # ── translation period and mirror-line phase on each axis ──────────────
    def fold_phase(P, vertical):
        n = hi if vertical else wi
        best_cls = None
        for phase in range(P):
            support, good = 0, True
            for A in range(phase, 2 * n - 2, P):
                a0, a1 = max(0, A - (n - 1)), min(A, n - 1)
                if a1 - a0 < 1:
                    continue
                gv = I[a0:a1 + 1, :] if vertical else I[:, a0:a1 + 1].T
                kv = base[a0:a1 + 1, :] if vertical else base[:, a0:a1 + 1].T
                both = kv & kv[::-1]
                if (both & (gv != gv[::-1])).any():
                    good = False; break
                support += int(both.sum())
            if good and support and (best_cls is None or support > best_cls[0]):
                best_cls = (support, phase)
        return None if best_cls is None else best_cls[1]

    def measure(vertical):
        axis = 0 if vertical else 1
        cands = periods(base, axis) or [I.shape[axis]]
        for p in cands:
            ph = fold_phase(p, vertical)
            if ph is not None:
                return p, ph
        return cands[0], None

    Pr, phr = measure(True)
    Pc, phc = measure(False)

    def axis_maps(lo, hi_, n, P, phase):
        """images of the interval [lo, hi_] under the axis symmetries:
        (start, reflected?, distance)"""
        out = []
        for k in range(-(2 * n) // P - 2, (2 * n) // P + 3):
            t = k * P
            if 0 <= lo + t and hi_ + t <= n - 1:
                out.append((lo + t, False, abs(t)))
            if phase is not None:
                A = phase + k * P
                s, e = A - hi_, A - lo
                if 0 <= s and e <= n - 1:
                    out.append((s, True, abs(s - lo)))
        return out

    g = I.copy()
    ops, sels = [], []

    def emit(op, sel):
        ops.append(op); sels.append(sel)

    def repair_rect(r0, c0, rh, cw):
        """fill rows r0..r0+rh-1, cols c0..c0+cw-1 from a clean symmetric twin"""
        rows = axis_maps(r0, r0 + rh - 1, hi, Pr, phr)
        colsm = axis_maps(c0, c0 + cw - 1, wi, Pc, phc)
        cands = []
        for (sr, fr, dr) in rows:
            for (sc, fc, dc) in colsm:
                if sr == r0 and sc == c0 and not fr and not fc:
                    continue
                src = g[sr:sr + rh, sc:sc + cw]
                # twin must be intruder-free now and must not overlap the box
                if (src == noise).any():
                    continue
                if not (sr + rh <= r0 or r0 + rh <= sr or sc + cw <= c0 or c0 + cw <= sc):
                    continue
                cands.append((int(fr) + int(fc), dr + dc, sr, sc, fr, fc))
        if not cands:
            return False
        cands.sort()
        _, _, sr, sc, fr, fc = cands[0]
        box = [r0, c0, rh - 1, cw - 1]
        src = g[sr:sr + rh, sc:sc + cw].copy()
        tgt = g[r0:r0 + rh, c0:c0 + cw]
        if fr or fc:
            # clear the whole box so the flip after the paste mirrors only the twin
            if (tgt != 0).any():
                emit(0, box)  # full rectangle intended: clear it before pasting into it
                tgt[:] = 0
        else:
            # a transparent paste cannot write the twin's 0s: punch the
            # intruder cells of this box out first if any twin cell there is 0
            nz = (tgt == noise)
            if noise != 0 and (nz & (src == 0)).any():
                cells = [(r0 + a, c0 + b) for a, b in zip(*np.nonzero(nz))]
                emit(0, sel_of(cells))
                tgt[nz] = 0
        emit(29, [sr, sc, rh - 1, cw - 1])     # CopyO the clean twin (full rectangle)
        before = tgt.copy()
        tgt[:] = np.where(src > 0, src, tgt)
        if (tgt != before).any():
            emit(30, [r0, c0, 0, 0])           # Paste over the patch
        else:
            ops.pop(); sels.pop()
        if fr:
            f = tgt[::-1].copy()
            if (f != tgt).any():
                emit(27, box); tgt[:] = f      # FlipV the pasted block (full rectangle)
        if fc:
            f = tgt[:, ::-1].copy()
            if (f != tgt).any():
                emit(26, box); tgt[:] = f      # FlipH the pasted block (full rectangle)
        return True

    def max_rects(mask):
        left = mask.copy(); out = []
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
                out.append((r, c, h, w))
        return out

    for _ in range(20):
        mask = (g == noise)
        if not mask.any():
            break
        progress = False
        for cells in comp_list(mask):
            if not (g[tuple(np.array(cells).T)] == noise).any():
                continue
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            r0, c0 = min(rs), min(cs)
            if repair_rect(r0, c0, max(rs) - r0 + 1, max(cs) - c0 + 1):
                progress = True
                continue
            # no clean twin for the whole patch: take it a rectangle at a time
            cm = np.zeros_like(mask)
            for r, c in cells:
                cm[r, c] = g[r, c] == noise
            for (r, c, h, w) in max_rects(cm):
                if repair_rect(r, c, h, w):
                    progress = True
                else:
                    for rr in range(r, r + h):
                        if repair_rect(rr, c, 1, w):
                            progress = True
                        else:
                            for cc in range(c, c + w):
                                if g[rr, cc] == noise and repair_rect(rr, cc, 1, 1):
                                    progress = True
        if not progress:
            break

    emit(34, [0, 0, hi - 1, wi - 1])
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
