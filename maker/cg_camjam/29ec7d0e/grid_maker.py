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
import random
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    pool = [c for c in cols if c != noisec]
    random.shuffle(pool)
    return {"bgc": bgc, "noisec": noisec, "ccols_pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, ccols_pool=None) -> dict:
    from random import randint, choice
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    remcols = [c for c in range(10) if c != noisec]
    if ccols_pool is None:
        ccols_pool = list(remcols)
        random.shuffle(ccols_pool)
    ccols_pool = [c for c in ccols_pool if c != noisec]
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = ccols_pool[:numc]
    pat = [[choice(ccols) for _ in range(wp)] for _ in range(hp)]
    go = [[bgc] * w for _ in range(h)]
    places = []
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
            places.append((loci, locj))
            for i in range(hp):
                for j in range(wp):
                    if 0 <= loci + i < h and 0 <= locj + j < w:
                        go[loci + i][locj + j] = pat[i][j]
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = [row[:] for row in go]

    def intact_somewhere(g):
        for (li, lj) in places:
            if li + hp > h or lj + wp > w:
                continue
            if all(g[li + i][lj + j] == pat[i][j] for i in range(hp) for j in range(wp)):
                return True
        return False

    succ, tr, maxtr = 0, 0, 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph, pw = randint(2, 6), randint(2, 6)
        li, lj = randint(0, h - ph), randint(0, w - pw)
        g2 = [row[:] for row in gi]
        for i in range(li, li + ph):
            for j in range(lj, lj + pw):
                g2[i][j] = noisec
        if intact_somewhere(g2):
            rows_ok = sum(1 for r in g2 if noisec not in r)
            cols_ok = sum(1 for j in range(w) if all(g2[i][j] != noisec for i in range(h)))
            if rows_ok >= 2 and cols_ok >= 2:
                succ += 1
                gi = g2
    k = choice((0, 1, 2, 3))
    gi = np.rot90(np.array(gi), -k).tolist()
    go = np.rot90(np.array(go), -k).tolist()
    return {"input": gi, "output": go}


def _period_tile(I, nc, allow_gaps=False):
    """Smallest (vp, hp) such that every non-noise cell agrees with its
    (r % vp, c % hp) class and every class has at least one witness.
    With allow_gaps, an unwitnessed class may be completed from a tile row or
    tile column that is a single solid colour (the background separator lines)."""
    h, w = I.shape
    known = I != nc
    cands = sorted(((a, b) for a in range(1, h) for b in range(1, w)),
                   key=lambda t: (t[0] * t[1], t[0], t[1]))
    for vp, hp in cands:
        R = {}
        ok = True
        for r in range(h):
            for c in range(w):
                if not known[r, c]:
                    continue
                k = (r % vp, c % hp)
                v = int(I[r, c])
                if R.setdefault(k, v) != v:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if len(R) == vp * hp:
            return vp, hp, R
        if not allow_gaps:
            continue
        good = True
        for a in range(vp):
            for b in range(hp):
                if (a, b) in R:
                    continue
                colv = {R[(x, b)] for x in range(vp) if (x, b) in R}
                rowv = {R[(a, y)] for y in range(hp) if (a, y) in R}
                if len(colv) == 1 and len(rowv) != 1:
                    R[(a, b)] = next(iter(colv))
                elif len(rowv) == 1 and len(colv) != 1:
                    R[(a, b)] = next(iter(rowv))
                elif len(rowv) == 1 and rowv == colv:
                    R[(a, b)] = next(iter(rowv))
                else:
                    good = False
        if good and len(R) == vp * hp and vp <= h // 2 and hp <= w // 2:
            return vp, hp, R
    return None


def _infer(I):
    h, w = I.shape
    pal = sorted(set(I.flatten().tolist()))

    def ncomp(col):
        m = I == col
        seen = np.zeros_like(m)
        n = 0
        for r in range(h):
            for c in range(w):
                if m[r, c] and not seen[r, c]:
                    n += 1
                    st = [(r, c)]
                    seen[r, c] = True
                    while st:
                        y, x = st.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = True
                                st.append((yy, xx))
        return n

    # noise = the colour forming the fewest, then smallest, blobs; confirm it by
    # checking that masking it out leaves a clean periodic tiling
    order = sorted(pal, key=lambda col: (ncomp(col), int((I == col).sum())))
    for gaps in (False, True):
        for nc in order:
            res = _period_tile(I, nc, gaps)
            if res is not None and (I == nc).any():
                vp, hp, R = res
                if vp < h and hp < w:
                    return nc, vp, hp, R
    return None


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]

    info = _infer(I)
    if info is None:
        return [34], [full]
    nc, vp, hp, R = info
    noise = I == nc
    if not noise.any():
        return [34], [full]
    tile = lambda r, c: R[(r % vp, c % hp)]

    # Reading: the picture IS one period tile repeated over the whole canvas.
    # Pick a reference tile window (vp x hp, any phase) - an undamaged one if it exists.
    best = None
    for r0 in range(0, h - vp + 1):
        for c0 in range(0, w - hp + 1):
            k = int(noise[r0:r0 + vp, c0:c0 + hp].sum())
            if best is None or k < best[0]:
                best = (k, r0, c0)
    _, r0, c0 = best
    G = I.copy()

    copy_op = 28
    if best[0] > 0:
        # no clean tile anywhere: mend the reference tile first, colour by colour,
        # from the tile measured on the undamaged cells, then copy from the canvas
        groups = {}
        for r in range(r0, r0 + vp):
            for c in range(c0, c0 + hp):
                if noise[r, c]:
                    groups.setdefault(tile(r, c), []).append((r, c))
        for col in sorted(groups):
            ops.append(col); sels.append(sel_of(groups[col]))
            for (r, c) in groups[col]:
                G[r, c] = col
        copy_op = 29

    # lattice of tile slots sharing the reference phase
    rs = r0 % vp
    cs = c0 % hp
    row_starts = list(range(rs - vp if rs else 0, h, vp))
    col_starts = list(range(cs - hp if cs else 0, w, hp))

    clip = None
    for tr in row_starts:
        for tc in col_starts:
            if tr == r0 and tc == c0:
                continue
            a0, a1 = max(tr, 0), min(tr + vp, h) - 1
            b0, b1 = max(tc, 0), min(tc + hp, w) - 1
            dmg = [(r, c) for r in range(a0, a1 + 1) for c in range(b0, b1 + 1)
                   if G[r, c] == nc]
            if not dmg:
                continue                      # an intact slot needs nothing
            if not any(tile(r, c) != 0 for r, c in dmg):
                continue                      # Paste can't write 0s; Color0 below
            sr, sc = r0 + (a0 - tr), c0 + (b0 - tc)
            want = (sr, sc, a1 - a0, b1 - b0)
            if clip != want:
                # full rectangle: the reference tile (or the part of it matching
                # this edge-clipped slot) copied verbatim
                ops.append(copy_op); sels.append([sr, sc, a1 - a0, b1 - b0])
                clip = want
            ops.append(30); sels.append([a0, b0, 0, 0])
            for r in range(a0, a1 + 1):
                for c in range(b0, b1 + 1):
                    v = tile(r, c)
                    if v != 0:
                        G[r, c] = v

    # noise cells whose tile value is 0 (transparent to Paste)
    left = [(r, c) for r in range(h) for c in range(w) if G[r, c] == nc]
    if left:
        ops.append(0); sels.append(sel_of(left))

    ops.append(34); sels.append(full)
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
