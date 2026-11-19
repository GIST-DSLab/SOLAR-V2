"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    remcols = [c for c in cols if c != markercol]
    numbgc = random.randint(1, 8)
    bgcols = random.sample(remcols, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h, max_w, markercol=None, bgcols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        x = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * x))))

    cols = list(range(1, 10))
    if markercol is None:
        markercol = random.choice(cols)
    if bgcols is None:
        rem = [c for c in cols if c != markercol]
        bgcols = random.sample(rem, unifint(diff_lb, diff_ub, (1, 8)))
    bgcols = list(bgcols)
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = random.randint(2, h // 4)
    ow = random.randint(2, w // 4)
    gi = [[random.choice(bgcols) for _ in range(w)] for _ in range(h)]
    allinds = [(i, j) for i in range(h) for j in range(w)]
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    for (i, j) in random.sample(allinds, numbl):
        gi[i][j] = 0
    # break every pre-existing all-zero oh x ow block
    occs = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
            if all(gi[a][b] == 0 for a in range(i, i + oh) for b in range(j, j + ow))]
    tocover = set()
    for (i, j) in occs:
        tocover.add((i + random.randrange(oh), j + random.randrange(ow)))
    for (i, j) in tocover:
        gi[i][j] = random.choice(bgcols)
    go = [row[:] for row in gi]
    inds = set(allinds)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // ((oh + 2) * (ow + 2)))))
    tr = succ = 0
    placed = []
    maxtr = 5 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            break
        li, lj = random.choice(cands)
        bd = {(li + a, lj + b) for a in range(oh) for b in range(ow)}
        plcd = {(r, c) for r in range(li - 1, li + oh + 1) for c in range(lj - 1, lj + ow + 1)} - bd
        if plcd <= inds:
            succ += 1
            placed.append((li, lj))
            inds -= plcd
            for (a, b) in bd:
                gi[a][b] = 0
                go[a][b] = 0
            for (a, b) in plcd:
                if succ == 1:
                    gi[a][b] = markercol
                go[a][b] = markercol
            if succ > 1:
                lines = [[(li - 1, lj + k) for k in range(ow)],
                         [(li + oh, lj + k) for k in range(ow)],
                         [(li + k, lj - 1) for k in range(oh)],
                         [(li + k, lj + ow) for k in range(oh)]]
                for ln in lines:
                    a, b = random.choice(ln)
                    gi[a][b] = random.choice(bgcols)
    if succ < 2:
        raise ValueError("too few boxes placed")
    # the rule must be unambiguous: the only all-zero oh x ow holes are the
    # hidden boxes themselves (no stray holes overlapping or beside them)
    holes = {(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
             if all(gi[a][b] == 0 for a in range(i, i + oh) for b in range(j, j + ow))}
    if holes != set(placed):
        raise ValueError("ambiguous hole layout")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the marker frame is a STENCIL. Every all-zero hole of the
    # marker's interior size is a hidden box; stamp a copy of the marker frame
    # around each one (CopyI the marker once, Paste it at each hole's corner).
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    def ring(r0, c0, r1, c1):
        s = set()
        for c in range(c0, c1 + 1):
            s.add((r0, c)); s.add((r1, c))
        for r in range(r0, r1 + 1):
            s.add((r, c0)); s.add((r, c1))
        return s

    def zero_blocks(oh, ow):
        # all-zero oh x ow holes with room for a frame around them
        return [(r, c) for r in range(1, hi - oh) for c in range(1, wi - ow)
                if not I[r:r + oh, c:c + ow].any()]

    # the marker: a colour whose every cell lies on one hollow rectangle frame
    # (interior at least 2x2 and entirely 0)
    cands = []
    for col in range(1, 10):
        pts = list(zip(*np.nonzero(I == col)))
        if not pts:
            continue
        rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set((int(a), int(b)) for a, b in pts) != ring(r0, c0, r1, c1):
            continue
        if I[r0 + 1:r1, c0 + 1:c1].any():
            continue
        n = len(zero_blocks(r1 - r0 - 1, c1 - c0 - 1))
        cands.append((0 if n >= 2 else 1, col, r0, c0, r1, c1))

    if cands:
        cands.sort()
        _, mcol, r0, c0, r1, c1 = cands[0]
        oh, ow = r1 - r0 - 1, c1 - c0 - 1
        grid = I.copy()
        targets = [b for b in zero_blocks(oh, ow) if b != (r0 + 1, c0 + 1)]
        # nearest hidden boxes first
        targets.sort(key=lambda b: (abs(b[0] - r0 - 1) + abs(b[1] - c0 - 1), b))
        copied = False
        for (r, c) in targets:
            fr = ring(r - 1, c - 1, r + oh, c + ow)
            if all(grid[a, b] == mcol for a, b in fr):
                continue
            if not copied:
                # the marker frame's full bbox (frame + its zero interior; zeros
                # are transparent so the clipboard holds exactly the frame)
                ops.append(28); sels.append([r0, c0, r1 - r0, c1 - c0])
                copied = True
            # stamp the stencil: paste lands on the ring around this hole
            ops.append(30); sels.append([r - 1, c - 1, 0, 0])
            for a, b in fr:
                grid[a, b] = mcol

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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
