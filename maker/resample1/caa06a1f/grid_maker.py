"""
ARC Task: caa06a1f (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of

ROTS = ["identity", "rot90", "rot180", "rot270"]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    tric = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "tric": tric, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, tric, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    swap = rot in ("rot90", "rot270")
    mh, mw = (max_w, max_h) if swap else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (10, min(30, mh)))
    w = _unifint(diff_lb, diff_ub, (10, min(30, mw)))
    vp = _unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    hp = _unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    remcols = [c for c in range(10) if c not in (bgc, tric)]
    numc = _unifint(diff_lb, diff_ub, (2, min(8, len(remcols), max(2, hp * vp))))
    ccols = random.sample(remcols, numc)
    tile = [[random.choice(ccols) for _ in range(hp)] for _ in range(vp)]
    gi = np.array([[tile[r % vp][c % hp] for c in range(w)] for r in range(h)])
    go = np.array([[tile[r % vp][(c - 1) % hp] for c in range(w)] for r in range(h)])
    ioffs = _unifint(diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = _unifint(diff_lb, diff_ub, (1, w - 2 * hp))
    gi[:ioffs, :] = tric
    gi[:, :joffs] = tric
    k = ROTS.index(rot)
    # dsl rot90 = clockwise = np.rot90(k=-1)
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _min_period(block, axis):
    n = block.shape[axis]
    for p in range(1, n):
        if axis == 0 and np.array_equal(block[p:], block[:-p]):
            return p
        if axis == 1 and np.array_equal(block[:, p:], block[:, :-p]):
            return p
    return n


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # band colour = dominant colour on the grid border (the cut-away strip)
    border = list(I[0]) + list(I[-1]) + list(I[:, 0]) + list(I[:, -1])
    tric = Counter(int(v) for v in border).most_common(1)[0][0]

    # intact periodic region = bbox of non-band cells (a corner rectangle)
    rr, cc = np.where(I != tric)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    block = I[r0:r1 + 1, c0:c1 + 1]
    vp = _min_period(block, 0)
    hp = _min_period(block, 1)

    # shift direction: sum of unit vectors of corners that are pattern
    dr = dc = 0
    if I[0, 0] != tric: dc -= 1
    if I[0, W - 1] != tric: dr -= 1
    if I[H - 1, W - 1] != tric: dc += 1
    if I[H - 1, 0] != tric: dr += 1
    move_op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]

    G = I.copy()

    def emit(op, sel, newG):
        nonlocal G
        if np.array_equal(newG, G):
            return
        G = newG
        ops.append(op); sels.append(sel)

    # 1. Translate the intact pattern one cell (ARCLE clips the edge, the
    #    vacated line reads 0). The whole rectangle IS the object.
    src = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    newG = G.copy()
    newG[r0:r1 + 1, c0:c1 + 1] = 0
    for r, c in src:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and I[r, c] != 0:
            newG[nr, nc] = I[r, c]
    emit(move_op, sel_of(src), newG)

    pr0 = max(0, r0 + dr); pr1 = min(H - 1, r1 + dr)
    pc0 = max(0, c0 + dc); pc1 = min(W - 1, c1 + dc)

    # 2. If the pattern uses colour 0 (transparent for Paste), lay a 0 base
    #    over the band so pasted tiles show their 0 cells there.
    has_zero = bool((block == 0).any())
    if has_zero and tric != 0:
        band = [(r, c) for r in range(H) for c in range(W)
                if not (pr0 <= r <= pr1 and pc0 <= c <= pc1)]
        newG = G.copy()
        for r, c in band:
            newG[r, c] = 0
        emit(0, sel_of(band), newG)

    # 3. Copy a period-aligned window of the shifted pattern and paste it
    #    repeatedly to extend the periodic pattern over the whole grid.
    def window(lo, hi, p):
        s = lo + ((-lo) % p)  # s % p == 0
        if s > hi:
            s = lo
        L = ((hi - s + 1) // p) * p
        if L == 0:
            L = hi - s + 1
        return s, L

    sr, Lr = window(pr0, pr1, vp)
    sc, Lc = window(pc0, pc1, hp)
    clip = G[sr:sr + Lr, sc:sc + Lc].copy()
    ops.append(29); sels.append([int(sr), int(sc), int(Lr - 1), int(Lc - 1)])
    copy_idx = len(ops) - 1
    origins = [(tr, tc) for tr in range(sr % vp, H, Lr) for tc in range(sc % hp, W, Lc)]
    for tr, tc in origins:
        newG = G.copy()
        hh = min(Lr, H - tr); ww = min(Lc, W - tc)
        sub = clip[:hh, :ww]
        m = sub != 0
        newG[tr:tr + hh, tc:tc + ww][m] = sub[m]
        emit(30, [int(tr), int(tc), 0, 0], newG)
    if len(ops) == copy_idx + 1:
        ops.pop(); sels.pop()

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task caa06a1f"
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
                                f"for task caa06a1f"
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
                    f"Failed to build a complete episode for task caa06a1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"caa06a1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
