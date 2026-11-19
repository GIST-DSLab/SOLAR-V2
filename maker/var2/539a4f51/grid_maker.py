"""
ARC Task: 539a4f51 (RE-ARC) — LLM-generated grid_maker
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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from maker.sel_helpers import sel_of

# The four orientations the generator can hand the picture in: the nested-L
# corner lands in a different corner of the grid for each one.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # The empty area is hardcoded 0 in the generator, so it is not sampled here.
    # The band colours come from 1..9: 0 has to stay readable as "no picture
    # here", otherwise the nested-L block cannot be told from the blank canvas.
    # Every op this maker emits is a Color/Move/Resize that carries 0 correctly,
    # so a 0-coloured band would still be painted right -- it just could not be
    # *found*; the post-Move Color0 patch below covers the carrying half anyway.
    palette = random.sample(range(1, 10), 9)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None, rot=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(a, min(lo, b))
        hi = max(a, min(hi, b))
        return random.randint(lo, hi)

    if palette is None:
        palette = random.sample(range(1, 10), 9)
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    # the output is 2d x 2d, so the canvas limit halves d
    dmax = min(15, max_h // 2, max_w // 2)
    if dmax < 2:
        dmax = 2
    d = unifint(diff_lb, diff_ub, (2, dmax))

    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = list(palette[:numc])

    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [random.choice(ccols) for _ in range(numocc)]
    while len(set(arr)) == 1:
        arr = [random.choice(ccols) for _ in range(d)]
    p = len(arr)

    gi = [[(arr[max(r, c)] if max(r, c) < p else 0) for c in range(d)] for r in range(d)]
    N = 2 * d
    go = [[arr[max(r, c) % p] for c in range(N)] for r in range(N)]

    ai = np.rot90(np.array(gi, dtype=int), k=rot)
    ao = np.rot90(np.array(go, dtype=int), k=rot)
    return {"input": ai.tolist(), "output": ao.tolist()}


def derive_operations(I, O=None, examples=None):
    """Grow the canvas to twice the side and keep the nested-L pattern going.

    Everything below is measured from I alone: which corner the L's grow out of,
    how many bands the picture has, and the colour of each band.  O is never
    read -- handed a different answer grid this returns the same operations.

    The *route* is chosen from the picture's own numbers (its corner, its band
    count, its apex colour), so one picture is always solved the same way while
    different pictures get solved in a different order -- the way two people
    filling in the same pattern would not agree on where to start.
    """
    I = np.asarray(I, dtype=int)
    d = int(I.shape[0])
    D = 2 * d

    def ring(R, C, r, c):
        return max(abs(r - R), abs(c - C))

    # ── read the picture ──────────────────────────────────────────────────────
    # The bands 0..p-1 of a nested-L block fill a p x p square in one corner.
    nz = np.argwhere(I != 0)
    if len(nz) == 0:
        return [34], [[0, 0, D - 1, D - 1]]
    rmin, cmin = (int(v) for v in nz.min(0))
    rmax, cmax = (int(v) for v in nz.max(0))
    p = max(rmax - rmin + 1, cmax - cmin + 1)

    corner, arr = None, None
    for (R, C) in [(0, 0), (0, d - 1), (d - 1, 0), (d - 1, d - 1)]:
        bands, ok = [], True
        for j in range(p):
            vals = {int(I[r, c]) for r in range(d) for c in range(d)
                    if ring(R, C, r, c) == j}
            if len(vals) != 1 or 0 in vals:
                ok = False
                break
            bands.append(vals.pop())
        if not ok:
            continue
        if any(I[r, c] != 0 for r in range(d) for c in range(d)
               if ring(R, C, r, c) >= p):
            continue
        corner, arr = (R, C), bands
        break
    if corner is None:                      # defensive: fall back to the bbox
        corner = (0 if rmin == 0 else d - 1, 0 if cmin == 0 else d - 1)
        R, C = corner
        sr = 1 if R == 0 else -1
        sc = 1 if C == 0 else -1
        arr = [int(I[R + sr * j, C + sc * j]) for j in range(p)]
    Ri, Ci = corner

    # the block keeps its corner when the picture doubles
    Ro = 0 if Ri == 0 else D - 1
    Co = 0 if Ci == 0 else D - 1

    def shell(j):
        """the j-th L, counted outward from the output's corner"""
        cells = set()
        for k in range(j + 1):
            cells.add((Ro + (j if Ro == 0 else -j), Co + (k if Co == 0 else -k)))
            cells.add((Ro + (k if Ro == 0 else -k), Co + (j if Co == 0 else -j)))
        return sorted(cells)

    ops, sels = [], []

    # ── 1. grow the canvas: d x d -> 2d x 2d (a full rectangle, so a bbox) ─────
    ops.append(33)
    sels.append([0, 0, D - 1, D - 1])
    G = np.zeros((D, D), dtype=int)
    G[:d, :d] = I                            # ResizeGrid leaves I at the top-left

    def paint(col, cells):
        cells = [(int(r), int(c)) for r, c in cells]
        if all(G[r, c] == col for r, c in cells):
            return                            # nothing would change -- don't emit
        ops.append(int(col))
        sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = col

    # which route? derived from the picture, not from a coin flip
    corner_id = (0 if Ri == 0 else 2) + (0 if Ci == 0 else 1)
    route = (corner_id + p + int(arr[0])) % 4

    if route in (0, 1):
        # ── keep what is already drawn: slide the block to its corner of the
        #    bigger canvas, then carry the pattern on outward from it ──────────
        src = [(r, c) for r in range(d) for c in range(d) if ring(Ri, Ci, r, c) < p]
        dr, dc = Ro - Ri, Co - Ci
        steps = [(21 if dr > 0 else 20, 1 if dr > 0 else -1, 0)] * abs(dr) \
              + [(22 if dc > 0 else 23, 0, 1 if dc > 0 else -1)] * abs(dc)
        if steps:
            # one grab, then empty selections keep the same block gliding
            bg = G.copy()
            for r, c in src:
                bg[r, c] = 0
            obj = [(r, c, int(G[r, c])) for r, c in src if G[r, c] != 0]
            for i, (mop, sr_, sc_) in enumerate(steps):
                ops.append(mop)
                sels.append(sel_of(src) if i == 0 else sel_of([]))
            G[:, :] = bg
            for r, c, v in obj:
                rr, cc = r + dr, c + dc
                if 0 <= rr < D and 0 <= cc < D:
                    G[rr, cc] = v
            # a Move carries nothing where the picture itself is coloured 0
            # (ARCLE reads 0 as "nothing there"); put those cells down by hand,
            # and emit nothing when the block holds no 0.
            lost = [(r + dr, c + dc) for r, c in src
                    if int(I[r, c]) == 0 and G[r + dr, c + dc] != 0]
            if lost:
                paint(0, lost)
        order = range(p, D) if route == 0 else range(D - 1, p - 1, -1)
        for j in order:
            paint(arr[j % p], shell(j))
    elif route == 2:
        # ── lay the outermost colour down over the whole canvas, then draw the
        #    nested squares back in, biggest first ──────────────────────────────
        for s in range(D, 0, -1):
            block = [(r, c) for r in range(D) for c in range(D)
                     if ring(Ro, Co, r, c) < s]
            paint(arr[(s - 1) % p], block)
    else:
        # ── one colour at a time: every L of that colour in a single stroke,
        #    colours taken in the order they appear from the corner ────────────
        order = []
        for col in arr:
            if col not in order:
                order.append(col)
        for col in order:
            cells = []
            for j in range(D):
                if arr[j % p] == col:
                    cells.extend(shell(j))
            paint(col, cells)

    ops.append(34)
    sels.append([0, 0, D - 1, D - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
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
                                f"for task 539a4f51"
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
                    f"Failed to build a complete episode for task 539a4f51 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"539a4f51-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
