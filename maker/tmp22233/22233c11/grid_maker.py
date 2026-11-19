"""
ARC Task: 08ed6ac7 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    cols = remove(8, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    if ccols is None:
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))
    ccols = [c for c in ccols if c != bgc and c != 8] or [remcols[0]]
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        if len(inds) == 0:
            break
        tr += 1
        od = randint(1, 3)
        g = canvas(bgc, (4, 4))
        g = fill(g, 8, {(0, 3), (3, 0)})
        col = choice(ccols)
        g = fill(g, col, {(1, 1), (2, 2)})
        if choice((True, False)):
            g = hmirror(g)
        g = upscale(g, od)
        inobj = recolor(col, ofcolor(g, col))
        outobj = inobj | recolor(8, ofcolor(g, 8))
        loc = choice(totuple(inds))
        outobj = shift(outobj, loc)
        inobj = shift(inobj, loc)
        outobji = toindices(outobj)
        if toindices(inobj).issubset(inds) and (outobji & fullinds).issubset(inds):
            succ += 1
            inds = (inds - outobji) - mapply(neighbors, outobji)
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """One stencil, four orientations of the canvas.

    Every object is two od x od squares meeting at a corner inside a 4x4 frame
    of od-blocks.  Read in a fixed orientation the rule is a single stencil:
    an object whose squares run "\\" (frame blocks (1,1),(2,2)) gets an 8-block
    at its frame's top-right block (0,3).  Turning the canvas a quarter turn
    at a time brings every other marker into that same stencil position
    ("/" objects become "\\", and each frame corner in turn becomes the
    top-right one).  So: square up the canvas, and at each orientation paint
    the cells where the stencil fires, then turn the canvas back home and
    crop to the original size.  O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    bgc = int(I[0, 0])   # row 0 / col 0 are always pure canvas (frame margin)

    # objects: two squares touching at a corner -> one 8-connected component
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                objs.append(cells)

    sq = max(hi, wi)

    def rot_cell(r, c, n, k):          # k quarter turns CW of an n x n square
        for _ in range(k % 4):
            r, c = c, n - 1 - r
        return r, c

    # stencil firing set per orientation k (cells given in ORIGINAL coords)
    fire = {k: [] for k in range(4)}
    for cells in objs:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        od = max(1, (r1 - r0 + 1) // 2)
        R0, C0 = r0 - od, c0 - od
        filled = ({(1, 1), (2, 2)} if I[r0, c0] != bgc else {(1, 2), (2, 1)})
        for k in range(4):
            if {rot_cell(i, j, 4, k) for i, j in filled} != {(1, 1), (2, 2)}:
                continue
            for bi in range(4):
                for bj in range(4):
                    if rot_cell(bi, bj, 4, k) == (0, 3):
                        blk = [(R0 + bi * od + a, C0 + bj * od + b)
                               for a in range(od) for b in range(od)]
                        fire[k] += [(r, c) for r, c in blk
                                    if 0 <= r < hi and 0 <= c < wi]

    needed = [k for k in range(4) if fire[k]]
    ops, sels = [], []
    if not needed:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    def plans():
        far = [k for k in needed if k]
        if not far:
            yield []
            return
        # turn CW up to the farthest needed orientation, back by the shorter way
        for direction in (1, -1):
            steps = [(k if direction == 1 else (4 - k) % 4) for k in far]
            n = max(steps)
            for back in ((-direction,) * n, (direction,) * (4 - n)):
                yield [direction] * n + list(back)

    def simulate(turns):
        """returns (ops, sels) or None if some op would be a no-op / revisit"""
        P_ops, P_sels = [], []
        grid = np.zeros((30, 30), dtype=int)
        grid[:hi, :wi] = I
        dim = (hi, wi)
        states = [(dim, grid.tobytes())]
        k = 0

        def push(op, sel):
            P_ops.append(op); P_sels.append(sel)
            st = (dim, grid.tobytes())
            if st in states:
                return False
            states.append(st)
            return True

        def paint_here():
            cells = [rot_cell(r, c, sq, k) for r, c in fire[k]]
            for r, c in cells:
                grid[r, c] = 8
            return push(8, sel_of(sorted(set(cells))))

        if turns and sq != hi or turns and sq != wi:
            dim = (sq, sq)                       # square up the canvas
            if not push(33, [0, 0, sq - 1, sq - 1]):
                return None
        if fire[0] and not paint_here():
            return None
        done = {0}
        for t in turns:
            sub = grid[:sq, :sq].copy()
            grid[:sq, :sq] = np.rot90(sub, -1 if t == 1 else 1)
            k = (k + t) % 4
            if not push(25 if t == 1 else 24, [0, 0, sq - 1, sq - 1]):
                return None                      # whole square canvas turned
            if fire[k] and k not in done:
                done.add(k)
                if not paint_here():
                    return None
        if done != set(needed) or k != 0:
            return None
        if dim != (hi, wi):
            dim = (hi, wi)                       # crop back to the original canvas
            keep = grid[:hi, :wi].copy()
            grid[:] = 0
            grid[:hi, :wi] = keep
            if not push(33, [0, 0, hi - 1, wi - 1]):
                return None
        return P_ops, P_sels

    best = None
    for turns in plans():
        res = simulate(turns)
        if res is not None and (best is None or len(res[0]) < len(best[0])):
            best = res
    if best is None:
        # symmetric corner case where a turn would not change the canvas:
        # paint each orientation's firing set directly
        for k in needed:
            ops.append(8); sels.append(sel_of(sorted(set(fire[k]))))
    else:
        ops, sels = best
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 22233c11"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22233c11"
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
                                f"for task 22233c11"
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
                    f"Failed to build a complete episode for task 22233c11 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22233c11-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
