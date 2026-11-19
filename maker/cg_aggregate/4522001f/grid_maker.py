"""
ARC Task: 4522001f (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS] + [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, dotc=2, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    hmax = max(3, min(10, max_h // 3))
    wmax = max(3, min(10, max_w // 3))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = _unifint(diff_lb, diff_ub, (3, wmax))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((3 * h, 3 * w), bgc, dtype=int)

    def place(i, j):
        gi[i, j] = sqc; gi[i, j + 1] = sqc; gi[i + 1, j] = sqc; gi[i + 1, j + 1] = dotc
        go[i:i + 4, j:j + 4] = sqc
        go[i + 4:i + 8, j + 4:j + 8] = sqc

    def footprint(i, j):
        return {(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)}

    def nbrs(cells):
        out = set()
        for r, c in cells:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    loci = random.randint(0, min(h - 2, 3 * h - 8))
    locj = random.randint(0, min(w - 2, 3 * w - 8))
    place(loci, locj)
    fp = footprint(loci, locj)
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc} - nbrs(fp)
    noccs = _unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    succ, tr, maxtr = 0, 0, 10 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = [p for p in iinds if p[0] <= h - 2 and p[1] <= w - 2]
        if not cands:
            break
        i, j = random.choice(cands)
        fp = footprint(i, j)
        if fp <= iinds:
            succ += 1
            iinds = (iinds - fp) - nbrs(fp)
            place(i, j)
    # rot90 (CW) applied `rot` times, as rotf in the original
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared with the golf entry): there is ONE canonical picture -- every seed
    has its dot in the lower-right of its 2x2 block, the canvas is 3h x 3w anchored at
    the top-left, and each seed becomes a 4x4 square on its block plus a second 4x4
    square 4 cells further down-right. The other orientations are mirror images of that
    picture. (A rotation becomes a mirror here because the construction is symmetric
    about the diagonal.) So: mirror the input into canonical orientation (dot ->
    lower-right), grow the canvas, draw the canonical squares, then mirror the whole
    canvas back. The maker instead shifts the anchor and grows squares in the dot's
    direction; this route keeps one fixed drawing and moves the orientation into
    flips. Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                   # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None      # 1 cell per seed

    # orientation of the dot inside its block, voted over all seeds
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    drs, dcs = [], []
    for rd, cd in dots:
        drs.append(1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0)
        dcs.append(1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0)
    DR = Counter(drs).most_common(1)[0][0] if drs else 1
    DC = Counter(dcs).most_common(1)[0][0] if dcs else 1
    flip_lr = DC == 0   # dot on the left -> mirror left<->right
    flip_ud = DR == 0   # dot on the top  -> mirror up<->down

    G = I.copy()
    ops, sels = [], []

    def emit(op, sel, newG):
        nonlocal G
        if np.array_equal(newG, G):
            return           # would be a visible no-op
        G = newG
        ops.append(op); sels.append(sel)

    # 1. mirror the input into canonical orientation (whole input rectangle)
    full_in = [0, 0, hi - 1, wi - 1]
    if flip_lr:
        emit(26, full_in, G[:, ::-1].copy())
    if flip_ud:
        emit(27, full_in, G[::-1, :].copy())

    # 2. grow the canvas to 3h x 3w (the input stays at the top-left)
    NG = np.zeros((ho, wo), dtype=int)
    NG[:hi, :wi] = G
    G = NG
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    if bgc != 0:
        # the rows / columns the resize just added become background (full rectangles)
        g = G.copy(); g[hi:, :] = bgc
        emit(bgc, [hi, 0, ho - 1 - hi, wo - 1], g)
        g = G.copy(); g[:hi, wi:] = bgc
        emit(bgc, [0, wi, hi - 1, wo - 1 - wi], g)

    # 3. canonical drawing: each seed (dot now lower-right) grows two 4x4 squares
    blocks = [(r - 1, c - 1) for r in range(hi) for c in range(wi)
              if G[r, c] == dotc and r >= 1 and c >= 1]
    for i, j in blocks:
        for k in (0, 1):
            r0, c0 = i + 4 * k, j + 4 * k
            if r0 >= ho or c0 >= wo:
                continue
            r1, c1 = min(r0 + 4, ho), min(c0 + 4, wo)
            g = G.copy(); g[r0:r1, c0:c1] = sqc
            emit(sqc, [r0, c0, r1 - r0 - 1, c1 - c0 - 1], g)   # full 4x4 square rectangle

    # 4. mirror the whole canvas back to the input's orientation (whole canvas rectangle)
    full_out = [0, 0, ho - 1, wo - 1]
    if flip_lr:
        emit(26, full_out, G[:, ::-1].copy())
    if flip_ud:
        emit(27, full_out, G[::-1, :].copy())

    ops.append(34); sels.append(full_out)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4522001f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4522001f"
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
                                f"for task 4522001f"
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
                    f"Failed to build a complete episode for task 4522001f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4522001f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
