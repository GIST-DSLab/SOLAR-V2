"""
ARC Task: 484b58aa (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


# ----------------------------------------------------------------- helpers ---
# The task: a sheared periodic wallpaper is damaged by rectangular patches of one
# "noise" colour.  The solver identifies that colour (it forms by far the fewest
# 4-connected components), measures the wallpaper's vertical/horizontal periods
# from the undamaged rows/columns, and restores every damaged cell from the
# copies of the pattern living +-1/+-2 periods away.

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _count_objects(g):
    """objects(I, T, F, F): univalued, 4-connected, background included."""
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    cnt = {}
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(g[r, c])
            cnt[col] = cnt.get(col, 0) + 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and g[nx, ny] == col:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
    return cnt


def _noise_color(g):
    """colour with the fewest connected components (ties broken by fewest cells)."""
    cnt = _count_objects(g)
    m = min(cnt.values())
    cands = sorted([c for c, n in cnt.items() if n == m])
    return min(cands, key=lambda c: int((g == c).sum())), cands


def _hperiod(rows):
    """smallest p with rows[:, p:] == rows[:, :-p], else the full width."""
    n, m = rows.shape
    for p in range(1, m):
        if np.array_equal(rows[:, p:], rows[:, :m - p]):
            return p
    return m


def _periods(g, noise):
    """(vertical, horizontal) period, measured on the undamaged rows / columns."""
    clean_rows = [r for r in range(g.shape[0]) if not (g[r] == noise).any()]
    clean_cols = [c for c in range(g.shape[1]) if not (g[:, c] == noise).any()]
    if not clean_rows or not clean_cols:
        return None, None
    return _hperiod(g[:, clean_cols].T), _hperiod(g[clean_rows, :])


def _reconstruct(g, noise, vp, hp):
    """paint g with all its non-noise cells shifted by (a*vp, b*hp), a,b in -2..2."""
    h, w = g.shape
    out = g.copy()
    for a in range(-2, 3):
        for b in range(-2, 3):
            dr, dc = a * vp, b * hp
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            if r0 >= r1 or c0 >= c1:
                continue
            src = g[r0:r1, c0:c1]
            msk = src != noise
            out[r0 + dr:r1 + dr, c0 + dc:c1 + dc][msk] = src[msk]
    return out


# ------------------------------------------------------------ sample_colors --
def sample_colors(num_examples=None) -> dict:
    # noisec = the damage colour (a fixed role for the whole episode);
    # ccols_pool = the fixed palette ordering the wallpaper draws its colours from.
    cols = list(range(10))
    noisec = random.choice(cols)
    pool = [c for c in cols if c != noisec]
    random.shuffle(pool)
    return {"noisec": noisec, "ccols_pool": pool}


# ----------------------------------------------------------------- generate --
def _attempt(diff_lb, diff_ub, max_h, max_w, noisec, ccols_pool, few):
    k = random.choice((0, 1, 2, 3))                     # final rotation (rotf)
    if k % 2 == 1:
        hub, wub = min(30, max_w), min(30, max_h)
    else:
        hub, wub = min(30, max_h), min(30, max_w)
    if hub < 6 or wub < 6:
        return None
    h = _unifint(diff_lb, diff_ub, (min(10, hub), hub))
    w = _unifint(diff_lb, diff_ub, (min(10, wub), wub))
    hp = _unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    wp = _unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
    numc = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = list(ccols_pool[:numc])
    ofs = random.randint(1, hp - 1) if hp > 1 else 0

    # sheared tiling of an hp x wp patch: go[i][j] = patt[(i + ofs*(j//wp)) % hp][j % wp]
    patt = np.array([[random.choice(ccols) for _ in range(wp)] for _ in range(hp)], dtype=int)
    ii = (np.arange(h)[:, None] + ofs * (np.arange(w)[None, :] // wp)) % hp
    jj = np.broadcast_to(np.arange(w)[None, :] % wp, (h, w))
    go = patt[ii, jj]

    places = []                                          # lattice-aligned tile origins
    for b in range(w // wp):
        c0 = wp * b
        for r0 in range((-ofs * b) % hp, h - hp + 1, hp):
            places.append((r0, c0))
    if not places:
        return None

    gi = go.copy()
    numpatches = 1 if few else _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    succ, tr, maxtr = 0, 0, 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph, pw = random.randint(2, 6), random.randint(2, 6)
        if ph > h or pw > w:
            continue
        r = random.randint(0, h - ph)
        c = random.randint(0, w - pw)
        g2 = gi.copy()
        g2[r:r + ph, c:c + pw] = noisec
        if not any(not (g2[pr:pr + hp, pc:pc + wp] == noisec).any() for pr, pc in places):
            continue                                     # one whole tile must stay intact
        if sum(1 for x in range(h) if not (g2[x] == noisec).any()) < 2:
            continue                                     # >= 2 undamaged rows
        if sum(1 for y in range(w) if not (g2[:, y] == noisec).any()) < 2:
            continue                                     # >= 2 undamaged columns
        gi = g2
        succ += 1
    if succ == 0:
        return None

    gi, go = np.rot90(gi, k), np.rot90(go, k)
    H, W = gi.shape
    if H > max_h or W > max_w:
        return None

    # the instance must be one the reference solver reproduces exactly
    nc, cands = _noise_color(gi)
    if nc != noisec or len(cands) != 1:
        return None
    vp, hpd = _periods(gi, noisec)
    if vp is None:
        return None
    if vp < H and not np.array_equal(go[vp:, :], go[:H - vp, :]):
        return None
    if hpd < W and not np.array_equal(go[:, hpd:], go[:, :W - hpd]):
        return None
    if not np.array_equal(_reconstruct(gi, noisec, vp, hpd), go):
        return None
    return {"input": gi.tolist(), "output": go.tolist()}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, ccols_pool=None, **kwargs) -> dict:
    if noisec is None:
        noisec = random.choice(range(10))
    if ccols_pool is None:
        ccols_pool = [c for c in range(10) if c != noisec]
        random.shuffle(ccols_pool)
    for t in range(400):
        res = _attempt(diff_lb, diff_ub, max_h, max_w, noisec, ccols_pool, t > 250)
        if res is not None:
            return res
    raise RuntimeError("484b58aa: could not build a valid instance")


# ----------------------------------------------------------- derive_operations
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    noise, _ = _noise_color(I)                 # the damage colour: fewest components
    vp, hp = _periods(I, noise)                # wallpaper periods from undamaged rows/cols
    if vp is None:
        vp, hp = h, w

    # what the wallpaper says each damaged cell should be: read it off a copy of the
    # pattern one or two periods away (all visible in I)
    target = {}
    offs = [(a * vp, b * hp) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]
    for r in range(h):
        for c in range(w):
            if I[r, c] != noise:
                continue
            val = None
            for dr, dc in offs:
                sr, sc = r - dr, c - dc
                if 0 <= sr < h and 0 <= sc < w and I[sr, sc] != noise:
                    val = int(I[sr, sc])
                    break
            if val is None or val != int(O[r, c]):
                val = int(O[r, c])             # safety net; the read-off already agrees
            if val != noise:
                target[(r, c)] = val

    # work patch by patch: each connected blob of noise is one damaged region
    seen, regions = set(), []
    for r in range(h):
        for c in range(w):
            if I[r, c] != noise or (r, c) in seen:
                continue
            stack, comp = [(r, c)], []
            seen.add((r, c))
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and I[nx, ny] == noise:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            comp.sort()
            regions.append(comp)
    regions.sort(key=lambda comp: comp[0])

    # restore one patch completely before moving to the next: one Color op per
    # pattern colour inside that patch, selecting exactly those cells
    for comp in regions:
        by_col = {}
        for cell in comp:
            if cell in target:
                by_col.setdefault(target[cell], []).append(cell)
        for col in sorted(by_col, key=lambda k: by_col[k][0]):
            ops.append(int(col))
            sels.append(sel_of(by_col[col]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])          # bbox = whole grid, Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
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
                                f"for task 484b58aa"
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
                    f"Failed to build a complete episode for task 484b58aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"484b58aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
