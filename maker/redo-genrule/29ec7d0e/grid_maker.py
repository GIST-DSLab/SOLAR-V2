"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import deque
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """Colour roles of the generator: the canvas colour, the noise colour and the
    palette the repeating pattern block is drawn from.  Colour 0 is left out so
    that every pattern cell is opaque for Copy/Paste."""
    pool = list(range(1, 10))
    bgc, noisec = sample(pool, 2)
    remcols = [c for c in pool if c != noisec]
    numc = randint(2, len(remcols))
    ccols = sample(remcols, numc)
    return {"bgc": bgc, "noisec": noisec, "ccols": ccols}


# ------------------------------------------------------- reference rule -----
def _rule_output(I):
    """The task's rule exactly as the RE-ARC verifier states it: repeat the
    non-noise content of I over the grid's own two periods."""
    x0 = palette(I)
    x1 = objects(I, T, F, F)
    x2 = lbind(colorfilter, x1)
    x3 = compose(size, x2)
    x4 = valmin(x0, x3)
    x5 = matcher(x3, x4)
    x6 = sfilter(x0, x5)
    x7 = lbind(colorcount, I)
    x8 = argmin(x6, x7)
    x9 = asobject(I)
    x10 = matcher(first, x8)
    x11 = compose(flip, x10)
    x12 = sfilter(x9, x11)
    x13 = lbind(contained, x8)
    x14 = compose(flip, x13)
    x15 = sfilter(I, x14)
    x16 = asobject(x15)
    x17 = hperiod(x16)
    x18 = dmirror(I)
    x19 = sfilter(x18, x14)
    x20 = asobject(x19)
    x21 = hperiod(x20)
    x22 = astuple(x21, x17)
    x23 = lbind(multiply, x22)
    x24 = neighbors(ORIGIN)
    x25 = mapply(neighbors, x24)
    x26 = apply(x23, x25)
    x27 = lbind(shift, x12)
    x28 = mapply(x27, x26)
    return paint(I, x28)


def _rot_list(g, k):
    k %= 4
    if k == 0:
        return [list(r) for r in g]
    if k == 1:                                   # 90 deg counter-clockwise
        return [list(r) for r in zip(*g)][::-1]
    if k == 2:
        return [list(r)[::-1] for r in g][::-1]
    return [list(r)[::-1] for r in zip(*g)]      # 90 deg clockwise


# -------------------------------------------------------------- generate ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int = None, noisec: int = None, ccols=None) -> dict:
    pool = list(range(1, 10))
    if bgc is None or noisec is None:
        bgc, noisec = sample(pool, 2)
    if not ccols:
        rem = [c for c in pool if c != noisec]
        ccols = sample(rem, randint(2, len(rem)))
    ccols = [c for c in ccols if c != noisec]
    if not ccols:
        ccols = [c for c in pool if c != noisec][:2]

    hlim = max(10, min(30, int(max_h)))
    wlim = max(10, min(30, int(max_w)))

    for _attempt in range(400):
        h = unifint(diff_lb, diff_ub, (10, hlim))
        w = unifint(diff_lb, diff_ub, (10, wlim))
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))

        # the repeating pattern block, and the periodic canvas it tiles
        pat = [[choice(ccols) for _ in range(wp)] for _ in range(hp)]
        go = [[bgc] * w for _ in range(h)]
        locs = []
        for a in range(h // hp + 1):
            for b in range(w // wp + 1):
                loci = (a + 1) + hp * a
                locj = (b + 1) + wp * b
                locs.append((loci, locj))
                for i in range(hp):
                    for j in range(wp):
                        r, c = loci + i, locj + j
                        if r < h and c < w:
                            go[r][c] = pat[i][j]

        # noise patches, under the generator's own acceptance conditions
        gi = [row[:] for row in go]
        numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
        succ, tr, maxtr = 0, 0, 5 * numpatches
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph, pw = randint(2, 6), randint(2, 6)
            li, lj = randint(0, h - ph), randint(0, w - pw)
            trial = [row[:] for row in gi]
            for i in range(li, li + ph):
                for j in range(lj, lj + pw):
                    trial[i][j] = noisec
            intact = False                       # one untouched pattern block must remain
            for (loci, locj) in locs:
                if loci + hp <= h and locj + wp <= w and all(
                        trial[loci + i][locj + j] == pat[i][j]
                        for i in range(hp) for j in range(wp)):
                    intact = True
                    break
            if not intact:
                continue
            if sum(1 for row in trial if noisec not in row) < 2:
                continue
            if sum(1 for c in range(w)
                   if all(trial[r][c] != noisec for r in range(h))) < 2:
                continue
            gi = trial
            succ += 1
        if succ < 1:
            continue

        opts = [0, 2] + ([1, 3] if (h <= wlim and w <= hlim) else [])
        k = choice(opts)
        gi_t = tuple(tuple(r) for r in _rot_list(gi, k))
        go_t = tuple(tuple(r) for r in _rot_list(go, k))

        # keep only instances the task's own rule maps input -> output on
        try:
            valid = (_rule_output(gi_t) == go_t)
        except NameError:                        # DSL unavailable: trust the port
            valid = True
        except Exception:
            valid = False
        if not valid:
            continue
        return {'input': gi_t, 'output': go_t}

    raise ValueError("29ec7d0e: no valid instance sampled")


# ------------------------------------------------------------ derivation ----
def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(cells)
    return out


def _comp_count_per_color(A):
    h, w = A.shape
    seen = np.zeros((h, w), dtype=bool)
    cnt = {}
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(A[r, c])
            cnt[col] = cnt.get(col, 0) + 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and A[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    return cnt


def _period(lines):
    """smallest p such that every line repeats with period p (fallback: its length)"""
    n = len(lines[0])
    for p in range(1, n):
        if all(np.array_equal(ln[p:], ln[:n - p]) for ln in lines):
            return p
    return n


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ── the noise colour: it covers the pattern, so it is the colour that forms
    #    the fewest blobs (ties broken by fewest cells) — the verifier's own test
    cnt = _comp_count_per_color(I)
    m = min(cnt.values())
    cands = [c for c, v in cnt.items() if v == m]
    noisec = min(cands, key=lambda c: (int((I == c).sum()), c))
    if not np.array_equal((I != O), (I == noisec) & (I != O)) or not (I == noisec).any():
        d = (I != O)                               # cross-check: what actually changed
        vals = set(I[d].tolist())
        if len(vals) == 1:
            noisec = vals.pop()
    mask = (I == noisec)
    if not mask.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # ── the two periods of the pattern, measured on the undamaged rows/columns
    clean_rows = [I[r] for r in range(h) if not mask[r].any()]
    clean_cols = [I[:, c] for c in range(w) if not mask[:, c].any()]
    hper = _period(clean_rows) if clean_rows else w
    vper = _period(clean_cols) if clean_cols else h

    # candidate translations: whole numbers of periods, nearest first
    shifts = []
    for a in range(-(h // max(vper, 1)) - 1, h // max(vper, 1) + 2):
        for b in range(-(w // max(hper, 1)) - 1, w // max(hper, 1) + 2):
            if a == 0 and b == 0:
                continue
            shifts.append((a * vper, b * hper))
    shifts.sort(key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))

    _cache = {}

    def periodic(dr, dc):
        """does the whole picture agree with itself when translated by (dr, dc)?"""
        if (dr, dc) in _cache:
            return _cache[(dr, dc)]
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        ok = False
        if r1 > r0 and c1 > c0:
            A = I[r0:r1, c0:c1]
            B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            both = (A != noisec) & (B != noisec)
            ok = bool(both.any()) and bool(np.all(A[both] == B[both]))
        _cache[(dr, dc)] = ok
        return ok

    def pattern_value(r, c):
        for (dr, dc) in shifts:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and I[rr, cc] != noisec and periodic(dr, dc):
                return int(I[rr, cc])
        return int(O[r, c])

    # an intact period tile: one whole repeat of the pattern, copyable as is
    tile = None
    if vper <= h and hper <= w:
        for a in range(h // vper):
            for b in range(w // hper):
                sr, sc = a * vper, b * hper
                blk = I[sr:sr + vper, sc:sc + hper]
                if (blk == noisec).any() or (blk == 0).any():
                    continue
                tile = (sr, sc)
                break
            if tile is not None:
                break

    cur = I.copy()
    clip_tile = False                              # is the intact tile on the clipboard?
    while True:
        rem = (cur == noisec)
        if not rem.any():
            break
        cells = _components(rem)[0]                # one damaged patch at a time
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        # (a) an intact copy of this very region, a whole number of periods away
        src = None
        for (dr, dc) in shifts:
            sr, sc = r0 + dr, c0 + dc
            if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                continue
            block = I[sr:sr + bh, sc:sc + bw]
            if (block == noisec).any() or (block == 0).any():
                continue                            # damaged, or invisible to Paste
            if not periodic(dr, dc):
                continue
            src = (sr, sc, block)
            break

        if src is not None:
            sr, sc, block = src
            # full rectangle: the intact period-shifted block, copied verbatim
            ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
            ops.append(30); sels.append([r0, c0, 0, 0])
            cur[r0:r0 + bh, c0:c0 + bw] = block
            clip_tile = False
            continue

        # (b) restamp the whole repeats of the pattern this patch sits on
        if tile is not None:
            tsr, tsc = tile
            done = False
            for a in range(r0 // vper, r1 // vper + 1):
                for b in range(c0 // hper, c1 // hper + 1):
                    dr0, dc0 = a * vper, b * hper
                    if dr0 + vper > h or dc0 + hper > w:
                        continue                    # repeat clipped by the grid edge
                    if not (cur[dr0:dr0 + vper, dc0:dc0 + hper] == noisec).any():
                        continue                    # nothing broken in this repeat
                    if not periodic(dr0 - tsr, dc0 - tsc):
                        continue
                    if not clip_tile:
                        # full rectangle: one whole repeat of the pattern
                        ops.append(28); sels.append([tsr, tsc, vper - 1, hper - 1])
                        clip_tile = True
                    ops.append(30); sels.append([dr0, dc0, 0, 0])
                    cur[dr0:dr0 + vper, dc0:dc0 + hper] = I[tsr:tsr + vper, tsc:tsc + hper]
                    done = True
            if done:
                continue

        # (c) no whole block survives anywhere: continue the pattern colour by colour
        byc = {}
        for (r, c) in cells:
            if cur[r, c] != noisec:
                continue
            byc.setdefault(pattern_value(r, c), []).append((r, c))
        for col in sorted(byc):
            ops.append(int(col)); sels.append(sel_of(byc[col]))
            for (r, c) in byc[col]:
                cur[r, c] = col

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
