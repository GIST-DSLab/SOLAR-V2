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

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """bgc and the noise colour are the two roles the rule depends on.

    The pattern colours (`ccols`) are irrelevant to the rule (it is purely
    structural: restore the periodic tiling), so they stay free.  The noise
    colour must be stable across the episode or a solver cannot know which
    colour is the damage.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    noisec = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "noisec": noisec}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, noisec: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
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
    maxtr = 5 * numpatches + 40
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
    if succ == 0:
        raise ValueError('no noise patch could be placed')
    rotchoices = [identity, rot180]
    if w <= max_h and h <= max_w:
        rotchoices = rotchoices + [rot90, rot270]
    rotf = choice(tuple(rotchoices))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------ trajectory ----
def derive_operations(I, O):
    """Rule: the grid is one pattern tile repeated with period (P, Q); some
    rectangles have been blotted out with a single 'noise' colour.

    Route (exactly the rule, nothing else):
      1. erase the noise  -> the damaged cells become holes (0 = transparent),
      2. copy the surviving pattern and paste it back onto itself shifted by a
         whole period.  Paste is transparent for 0, so every shifted copy only
         drops pattern into the holes; the intact cells are untouched.
         Repeat with whichever period offset still recovers something.
      Cells whose true colour is 0 need nothing: the erase already left them 0.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    diff = (I != O)
    if not diff.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # the damage is one colour that never occurs in the clean tiling
    noisec = int(I[diff][0])

    # --- the two periods of the tiling -------------------------------------
    P = None
    for p in range(1, h):
        if np.array_equal(O[p:, :], O[:h - p, :]):
            P = p
            break
    Q = None
    for q in range(1, w):
        if np.array_equal(O[:, q:], O[:, :w - q]):
            Q = q
            break

    W = I.copy()

    # --- 1. erase the noise (one op, the whole blotted region) -------------
    if noisec != 0:
        noise_cells = [(int(r), int(c)) for r, c in np.argwhere(I == noisec)]
        ops.append(0)
        sels.append(sel_of(noise_cells))          # exact mask of the damage
        W[I == noisec] = 0

    # --- 2. shift the surviving pattern by whole periods and paste ---------
    ivals = [0] + ([-1, 1, -2, 2] if P else [])
    jvals = [0] + ([-1, 1, -2, 2] if Q else [])
    cand = []
    for i in ivals:
        for j in jvals:
            if i == 0 and j == 0:
                continue
            dr = i * (P if P else 0)
            dc = j * (Q if Q else 0)
            if abs(dr) >= h or abs(dc) >= w:
                continue
            cand.append((dr, dc))

    def plan(g, dr, dc):
        sr, sc = max(0, -dr), max(0, -dc)
        tr, tc = max(0, dr), max(0, dc)
        hh, ww = h - abs(dr), w - abs(dc)
        src = g[sr:sr + hh, sc:sc + ww].copy()
        dst = g[tr:tr + hh, tc:tc + ww]
        m = (src != 0) & (dst != src)             # cells the Paste will change
        return sr, sc, tr, tc, hh, ww, src, m

    while (W != O).any():
        best, bestcnt = None, 0
        for (dr, dc) in cand:
            cnt = int(plan(W, dr, dc)[7].sum())
            if cnt > bestcnt:
                bestcnt, best = cnt, (dr, dc)
        if best is None:
            break
        dr, dc = best
        sr, sc, tr, tc, hh, ww, src, m = plan(W, dr, dc)
        # full rectangle: the surviving pattern minus the strip that falls off
        ops.append(29); sels.append([sr, sc, hh - 1, ww - 1])
        ops.append(30); sels.append([tr, tc, 0, 0])
        W[tr:tr + hh, tc:tc + ww][m] = src[m]

    # --- safety net (should not trigger): whatever the shifts could not reach
    rem = (W != O)
    if rem.any():
        for col in sorted({int(v) for v in O[rem].tolist()}):
            cells = [(int(r), int(c)) for r, c in np.argwhere(rem & (O == col))]
            ops.append(int(col)); sels.append(sel_of(cells))
            for r, c in cells:
                W[r, c] = col

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
