"""
ARC Task: fcb5c309 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, sqc = random.sample(range(10), 3)
    return {"bgc": bgc, "dotc": dotc, "sqc": sqc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round(diff_lb * (b - a)))
    hi = a + int(round(diff_ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _box(r0, c0, r1, c1):
    s = set()
    for c in range(c0, c1 + 1):
        s.add((r0, c)); s.add((r1, c))
    for r in range(r0, r1 + 1):
        s.add((r, c0)); s.add((r, c1))
    return s


def _rect(r0, c0, r1, c1):
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, sqc=None) -> dict:
    if bgc is None or dotc is None or sqc is None:
        bgc, dotc, sqc = random.sample(range(10), 3)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    numsq = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    go = None
    while tr < maxtr and succ < numsq:
        oh = random.randint(3, 7)
        ow = random.randint(3, 7)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if len(cands) == 0:
            break
        loci, locj = random.choice(cands)
        r1, c1 = loci + oh - 1, locj + ow - 1
        sq = _box(loci, locj, r1, c1)
        bd = _rect(loci, locj, r1, c1)
        if bd.issubset(inds):
            for r, c in sq:
                gi[r][c] = sqc
            ib = sorted(_rect(loci + 1, locj + 1, r1 - 1, c1 - 1))
            if numcells is None:
                numcells = _unifint(diff_lb, diff_ub, (1, len(ib)))
                cells = random.sample(ib, numcells)
                take = True
            else:
                nc = _unifint(diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = random.sample(ib, nc)
            for r, c in cells:
                gi[r][c] = dotc
            if take:
                go = [[(dotc if gi[r][c] == sqc else gi[r][c]) for c in range(locj, c1 + 1)]
                      for r in range(loci, r1 + 1)]
                take = False
            ob = _box(loci - 1, locj - 1, r1 + 1, c1 + 1)
            inds = (inds - bd) - ob
            succ += 1
        tr += 1
    if go is None:
        raise ValueError("no box placed")
    nnoise = _unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = random.sample(sorted(inds), min(nnoise, len(inds)))
    for r, c in noise:
        gi[r][c] = dotc
    return {"input": [list(row) for row in gi], "output": [list(row) for row in go]}


def _components(I, col):
    hi, wi = I.shape
    seen = set()
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != col or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in seen and I[na, nb] == col:
                        seen.add((na, nb))
                        stack.append((na, nb))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _is_frame(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return set(cells) == _box(r0, c0, r1, c1) and min(r1 - r0 + 1, c1 - c0 + 1) > 2


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    palette = sorted(cnt)

    # frame colour: the colour whose every component is a hollow rectangle (>=3x3)
    sqc = None
    for col in palette:
        comps = _components(I, col)
        if comps and all(_is_frame(cc) for cc in comps):
            sqc = col
            break
    if sqc is None:
        best_n = -1
        for col in palette:
            n = sum(1 for cc in _components(I, col) if _is_frame(cc))
            if n > best_n:
                best_n, sqc = n, col

    # dot colour: the rarest remaining colour
    others = [c for c in palette if c != sqc]
    dotc = min(others, key=lambda c: cnt[c]) if others else sqc

    # the frame enclosing the most dots
    frames = [cc for cc in _components(I, sqc) if _is_frame(cc)]

    def dots_inside(cells):
        r0, c0, r1, c1 = _bbox(cells)
        return int(np.sum(I[r0:r1 + 1, c0:c1 + 1] == dotc))

    chosen = max(frames, key=dots_inside)
    r0, c0, r1, c1 = _bbox(chosen)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []

    # 1. Cut the chosen box out first: crop to its full rectangle (frame + interior).
    ops.append(33); sels.append([r0, c0, bh - 1, bw - 1])

    # 2. Then recolour the frame, which is now the cropped canvas's border ring,
    #    to the dot colour.
    ring = sorted(_box(0, 0, bh - 1, bw - 1))
    ops.append(int(dotc)); sels.append(sel_of(ring))

    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fcb5c309"
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
                                f"for task fcb5c309"
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
                    f"Failed to build a complete episode for task fcb5c309 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fcb5c309-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
