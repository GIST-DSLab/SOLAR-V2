"""
ARC Task: f9012d9b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The generator draws a palette `ccols` at random and paints every tile cell
    # with a random member of it.  That palette is the only randomly sampled
    # colour material, so it is fixed once per episode.  0 is hard-coded (it is
    # the colour of the punched-out gap) and is therefore NOT sampled here.
    nc = random.randint(2, 9)
    ccols = random.sample(range(1, 10), nc)
    return {"ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kwargs) -> dict:
    """Periodic wallpaper with a rectangular gap punched out; output = the gap's
    hidden content.  The hard-coded 30 bounds are replaced by max_h / max_w.

    Extra constraint w.r.t. the original generator: the gap is re-drawn until the
    missing content can actually be carried into place by reflections of bands
    whose sources lie outside the gap (checked by calling derive_operations).
    This only rejects gaps so large / badly placed that no intact periodic twin
    of them exists anywhere in the grid."""

    def unifint(lb, ub, bounds):
        a, b = bounds
        b = max(a, b)
        lo = min(max(a, a + int((b - a) * lb)), b)
        hi = min(max(a, a + int((b - a) * ub)), b)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def dihedral(a, t):
        if t == 0:
            return a
        if t == 1:
            return np.rot90(a, 1)
        if t == 2:
            return np.rot90(a, 2)
        if t == 3:
            return np.rot90(a, 3)
        if t == 4:
            return np.fliplr(a)
        if t == 5:
            return np.flipud(a)
        if t == 6:
            return a.T
        return np.rot90(a.T, 2)

    if not ccols:
        ccols = random.sample(range(1, 10), random.randint(2, 9))
    ccols = list(ccols)

    last = None
    for _outer in range(60):
        hp = unifint(diff_lb, diff_ub, (2, max(2, min(10, max_h // 3))))
        wp = unifint(diff_lb, diff_ub, (2, max(2, min(10, max_w // 3))))
        tile = [[random.choice(ccols) for _ in range(wp)] for _ in range(hp)]
        numhp = unifint(diff_lb, diff_ub, (3, max(3, max_h // hp)))
        numwp = unifint(diff_lb, diff_ub, (3, max(3, max_w // wp)))
        H, W = numhp * hp, numwp * wp
        full = np.array([[tile[r % hp][c % wp] for c in range(W)] for r in range(H)],
                        dtype=int)
        full = full[:H - random.randint(0, hp), :W - random.randint(0, wp)]
        h, w = full.shape
        if h - hp - 1 < 1 or w - wp - 1 < 1:
            continue

        for _inner in range(80):
            sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
            sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
            loci = random.randint(0, h - sgh)
            locj = random.randint(0, w - sgw)
            go = full[loci:loci + sgh, locj:locj + sgw].copy()
            gi = full.copy()
            gi[loci:loci + sgh, locj:locj + sgw] = 0
            last = (gi, go, h, w)
            try:
                ops, _sels = derive_operations(gi, go)
            except Exception:
                continue
            if 26 in ops or 27 in ops:
                last = (gi, go, h, w)
                break
        else:
            continue
        break

    gi, go, h, w = last
    ts = [0, 2, 4, 5]
    if w <= max_h and h <= max_w:
        ts += [1, 3, 6, 7]
    t = random.choice(ts)
    gi, go = dihedral(gi, t), dihedral(go, t)
    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


def derive_operations(I, O):
    """The grid is a wallpaper: it repeats with period ph down and pw across.
    A rectangular gap was punched out of it; the answer is what belonged there.

    The missing block has an exact twin one whole period away.  A twin is carried
    onto the gap by REFLECTION: reflecting the band running from the gap to its
    twin swaps the two (equal size, sitting at the two ends of the band), and a
    second reflection of the freshly filled block undoes the mirror the first one
    imposed -- two parallel reflections compose into exactly the one-period
    translation the wallpaper is built from.  Whatever lies strictly between gap
    and twin is reversed by the band reflection and is put back by reflecting
    that interior in place.  When one twin cannot cover the whole gap, the gap is
    filled band by band the same way.  Finally the completed gap rectangle is
    cropped out: that is the answer.

    Every selection is a FULL RECTANGLE -- a band / block that is reflected or
    cropped in its entirety, background included -- so the [r, c, h-1, w-1] bbox
    form is exactly the set of cells intended.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    zr, zc = np.where(I == 0)
    R0, R1 = int(zr.min()), int(zr.max())
    C0, C1 = int(zc.min()), int(zc.max())

    def period(M, axis):
        # smallest shift along `axis` under which every pair of visible cells agrees
        n = M.shape[axis]
        for p in range(1, n):
            if axis == 1:
                A, B = M[:, :n - p], M[:, p:]
            else:
                A, B = M[:n - p, :], M[p:, :]
            m = (A != 0) & (B != 0)
            if m.any() and bool(np.all(A[m] == B[m])):
                return p
        return None

    def plan(M, a0, a1, b0, b1, per):
        """Fill rows a0..a1 (cols b0..b1) of M by reflections; `per` = row period.
        Returns (list of rectangles (row, nrows) to FlipV, resulting grid)."""
        if not per:
            return None
        G = M.copy()
        nr_all = G.shape[0]
        rects = []
        segs = [(a0, a1)]
        for _round in range(16):
            if not segs:
                break
            pick = None
            for si in range(len(segs)):
                a, b = segs[si]
                cands = [(a, t1) for t1 in range(b, a - 1, -1)]
                cands += [(t0, b) for t0 in range(a + 1, b + 1)]
                for (t0, t1) in cands:
                    n = t1 - t0 + 1
                    for k in range(1, nr_all // per + 2):
                        e = k * per
                        for s in (1, -1):
                            s0, s1 = t0 + s * e, t1 + s * e
                            if s0 < 0 or s1 >= nr_all:
                                continue
                            if not (s1 < a0 or s0 > a1):
                                continue                     # twin must sit outside the gap
                            if (G[s0:s1 + 1, b0:b1 + 1] == 0).any():
                                continue                     # twin must be intact right now
                            lo, hi = min(t0, s0), max(t1, s1)
                            m0, m1 = lo + n, hi - n          # band interior
                            if m1 >= m0 and (G[m0:m1 + 1, b0:b1 + 1] == 0).any():
                                continue                     # interior must be intact to be restorable
                            pick = (si, t0, t1, n, lo, hi, m0, m1)
                            break
                        if pick:
                            break
                    if pick:
                        break
                if pick:
                    break
            if pick is None:
                return None
            si, t0, t1, n, lo, hi, m0, m1 = pick
            todo = [(lo, hi - lo + 1)]                       # 1. reflect the band: twin lands on the gap
            if n > 1:
                todo.append((t0, n))                         # 2. un-mirror the block just filled
            if m1 > m0:
                todo.append((m0, m1 - m0 + 1))               # 3. put the band's interior back
            for (r, nr) in todo:
                sub = G[r:r + nr, b0:b1 + 1]
                new = np.flipud(sub)
                if not np.array_equal(sub, new):             # skip anything that would change nothing
                    G[r:r + nr, b0:b1 + 1] = new
                    rects.append((r, nr))
            a, b = segs[si]
            nxt = []
            if a <= t0 - 1:
                nxt.append((a, t0 - 1))
            if t1 + 1 <= b:
                nxt.append((t1 + 1, b))
            segs = segs[:si] + nxt + segs[si + 1:]
        if segs:
            return None
        return rects, G

    ph, pw = period(I, 0), period(I, 1)

    best = None
    pv = plan(I, R0, R1, C0, C1, ph)
    if pv is not None and np.array_equal(pv[1][R0:R1 + 1, C0:C1 + 1], O):
        best = ('v', pv[0])
    phz = plan(I.T, C0, C1, R0, R1, pw)                      # same planner, transposed view
    if phz is not None and np.array_equal(phz[1].T[R0:R1 + 1, C0:C1 + 1], O):
        if best is None or len(phz[0]) < len(best[1]):
            best = ('h', phz[0])

    ops, sels = [], []
    if best is not None:
        mode, rects = best
        for (r, nr) in rects:
            if mode == 'v':
                # whole band, rows r..r+nr-1 across the gap's columns
                ops.append(27)
                sels.append([r, C0, nr - 1, C1 - C0])
            else:
                # whole band, cols r..r+nr-1 across the gap's rows
                ops.append(26)
                sels.append([R0, r, R1 - R0, nr - 1])
    else:
        # No twin can be reflected in (gap too large / cornered): grow the
        # wallpaper into the gap one period at a time with transparent pastes.
        G = I.copy()
        sph, spw = ph or 1, pw or 1
        steps = [(0, spw), (0, -spw), (sph, 0), (-sph, 0)]
        for _ in range(80):
            if not (G == 0).any():
                break
            moved = False
            for dr, dc in steps:
                if not (G == 0).any():
                    break
                sr0, sr1 = max(0, -dr), h - 1 - max(0, dr)
                sc0, sc1 = max(0, -dc), w - 1 - max(0, dc)
                if sr1 < sr0 or sc1 < sc0:
                    continue
                src = G[sr0:sr1 + 1, sc0:sc1 + 1]
                dr0, dc0 = sr0 + dr, sc0 + dc
                new = G.copy()
                reg = new[dr0:dr0 + src.shape[0], dc0:dc0 + src.shape[1]]
                m = src != 0
                reg[m] = src[m]
                if np.array_equal(new, G):
                    continue
                ops += [29, 30]
                sels += [[sr0, sc0, sr1 - sr0, sc1 - sc0], [dr0, dc0, 0, 0]]
                G = new
                moved = True
            if not moved:
                break

    # crop to the rectangle the gap occupied -- it now holds the recovered pattern
    ops.append(33)
    sels.append([R0, C0, R1 - R0, C1 - C0])
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
                        f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
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
                                f"for task f9012d9b"
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
                    f"Failed to build a complete episode for task f9012d9b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f9012d9b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
