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
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


# -------------------------------------------------------------- generate ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, noisec: int) -> dict:
    cols = interval(0, 10, 1)
    lim = max(10, min(max_h, max_w))
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
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
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
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------ derivation ----
def _row_period_ok(A, nc, p):
    h, w = A.shape
    for r in range(h - p):
        for c in range(w):
            a = A[r, c]
            b = A[r + p, c]
            if a != nc and b != nc and a != b:
                return False
    return True


def _col_period_ok(A, nc, p):
    return _row_period_ok(A.T, nc, p)


def _infer(A):
    """Measure (noise colour, vertical period, horizontal period, clean grid) from A alone."""
    h, w = A.shape
    best = None
    for nc in sorted(set(A.flatten().tolist())):
        vps = [p for p in range(1, h + 1) if _row_period_ok(A, nc, p)]
        hps = [p for p in range(1, w + 1) if _col_period_ok(A, nc, p)]
        for vp, hp in sorted([(a, b) for a in vps for b in hps], key=lambda t: t[0] * t[1]):
            tile = {}
            bad = False
            for r in range(h):
                for c in range(w):
                    v = int(A[r, c])
                    if v == nc:
                        continue
                    k = (r % vp, c % hp)
                    if k in tile and tile[k] != v:
                        bad = True
                        break
                    tile[k] = v
                if bad:
                    break
            if bad or len(tile) != vp * hp:
                continue
            if nc in tile.values():          # noise colour never occurs in the clean pattern
                continue
            R = np.array([[tile[(r % vp, c % hp)] for c in range(w)] for r in range(h)], dtype=int)
            cand = (vp * hp, nc, vp, hp, R)
            if best is None or cand[0] < best[0]:
                best = cand
            break
    if best is None:
        return None
    _, nc, vp, hp, R = best
    return nc, vp, hp, R


def _blobs(mask):
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


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    info = _infer(I)
    if info is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    nc, vp, hp, R = info

    G = I.copy()
    if np.array_equal(G, R):                      # nothing damaged -> nothing to do
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    mask = (I == nc)
    for cells in _blobs(mask):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        # find an intact copy of this exact region one whole pattern-period away
        cand = []
        for i in range(-(h // vp) - 1, h // vp + 2):
            for j in range(-(w // hp) - 1, w // hp + 2):
                if i == 0 and j == 0:
                    continue
                sr, sc = r0 + i * vp, c0 + j * hp
                if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                    continue
                if mask[sr:sr + bh, sc:sc + bw].any():
                    continue
                cand.append((abs(i) + abs(j), abs(i), sr, sc))
        cand.sort()

        if cand:
            _, _, sr, sc = cand[0]
            # full rectangle: the whole period-shifted source block is copied verbatim
            ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
            ops.append(30); sels.append([r0, c0, 0, 0])
            src = I[sr:sr + bh, sc:sc + bw]
            for dr in range(bh):
                for dc in range(bw):
                    v = int(src[dr, dc])
                    if v != 0:                     # Paste is transparent for 0
                        G[r0 + dr, c0 + dc] = v

        # cells of this patch the paste could not carry (0-valued pattern cells,
        # or no intact period-shifted source existed): paint from the measured tile
        rest = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if G[r, c] != R[r, c]:
                    rest.setdefault(int(R[r, c]), []).append((r, c))
        for col in sorted(rest):
            ops.append(col); sels.append(sel_of(rest[col]))
            for (r, c) in rest[col]:
                G[r, c] = col

    # safety net: any cell still off the reconstructed pattern
    rem = {}
    for r in range(h):
        for c in range(w):
            if G[r, c] != R[r, c]:
                rem.setdefault(int(R[r, c]), []).append((r, c))
    for col in sorted(rem):
        ops.append(col); sels.append(sel_of(rem[col]))

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
