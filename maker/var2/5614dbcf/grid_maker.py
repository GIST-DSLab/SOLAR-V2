"""
ARC Task: 5614dbcf (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The only colours the generator samples at random are the "cell" colours
    # (colset).  0 is hard-coded as the empty cell and 5 is hard-coded as the
    # noise colour, so neither is fixed here.  The palette is fixed once per
    # episode so every instance of the episode is painted from the same set.
    opts = [c for c in range(1, 10) if c != 5]
    k = random.randint(4, 8)
    palette = random.sample(opts, k)
    return {"palette": palette}


def generate(diff_lb, diff_ub, max_h, max_w, palette=None) -> dict:
    if palette is None:
        palette = [c for c in range(1, 10) if c != 5]
    colopts = tuple(palette)

    dmax = min(10, max_h // 3, max_w // 3)
    if dmax < 2:
        dmax = 2
    dim_bounds = (2, dmax)
    col_card_bounds = (1, min(8, len(colopts)))
    noise_card_bounds = (0, 8)

    noisedindscands = totuple(asindices(canvas(0, (3, 3))))
    d = unifint(diff_lb, diff_ub, dim_bounds)
    cells_card_bounds = (1, d * d)
    go = canvas(0, (d, d))
    inds = totuple(asindices(go))
    numocc = unifint(diff_lb, diff_ub, cells_card_bounds)
    numcol = unifint(diff_lb, diff_ub, col_card_bounds)
    occs = sample(inds, numocc)
    colset = sample(colopts, numcol)
    gi = upscale(go, 3)
    for occ in inds:
        offset = multiply(3, occ)
        numnoise = unifint(diff_lb, diff_ub, noise_card_bounds)
        noise = sample(noisedindscands, numnoise)
        if occ in occs:
            col = choice(colset)
            go = fill(go, col, initset(occ))
            gi = fill(gi, col, shift(noisedindscands, offset))
        gi = fill(gi, 5, shift(noise, offset))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: I is a d x d picture blown up 3x; every 3x3 block carries one true
    colour (0 = empty) partly hidden under noise.  Write each block's true
    colour into one cell of a d x d window, then keep that window.

    The rule is fixed; the ROUTE through it is read off the picture itself, so
    the same instance always replays identically and different instances are
    solved differently:
        style      -- sweep / background-first / colour-by-colour / shape-by-shape
        anchor     -- which corner of the canvas the small picture is built in
        axis       -- rows or columns, for the sweeping style
        crop_first -- shrink the canvas first, or draw first and shrink after
    Every op is a paint the answer cannot be reached without, plus the one
    resize.  Cells whose colour is 0 are laid down with an explicit Color0.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    dh, dw = hi // 3, wi // 3

    # ---- which colour is the noise?  A task constant: the colour that turns up
    # in the demonstration inputs and never in their outputs.  5 otherwise.
    cands = []
    if examples:
        ins, outs = set(), set()
        for a, b in examples:
            ins |= set(np.asarray(a, dtype=int).flatten().tolist())
            outs |= set(np.asarray(b, dtype=int).flatten().tolist())
        only_in = sorted(ins - outs)
        if len(only_in) == 1:
            cands.append(int(only_in[0]))
    if 5 not in cands:
        cands.append(5)

    def read_blocks(n):
        out = []
        for i in range(dh):
            row = []
            for j in range(dw):
                vals = [v for v in I[3 * i:3 * i + 3, 3 * j:3 * j + 3].flatten().tolist() if v != n]
                if len(set(vals)) != 1:
                    return None
                row.append(int(vals[0]))
            out.append(row)
        return out

    T = None
    for n in cands:
        T = read_blocks(n)
        if T is not None:
            break
    if T is None:                                   # fallback: majority non-5
        T = []
        for i in range(dh):
            row = []
            for j in range(dw):
                vals = [v for v in I[3 * i:3 * i + 3, 3 * j:3 * j + 3].flatten().tolist() if v != 5]
                row.append(int(Counter(vals).most_common(1)[0][0]) if vals else 0)
            T.append(row)

    # ---- route choices, all read out of the picture T (hence out of I) ----
    flat = [T[i][j] for i in range(dh) for j in range(dw)]
    cnt = Counter(flat)
    t_sum = sum(flat)
    t_nz = sum(1 for v in flat if v != 0)
    t_ncol = len(cnt)

    style = (t_nz + t_ncol) % 4          # 0 sweep, 1 base+colours, 2 colours, 3 shapes
    anchor = t_sum % 4                   # 0 TL, 1 TR, 2 BL, 3 BR
    axis = (t_nz + dh) % 2               # 0 row sweep, 1 column sweep
    crop_first = (t_sum + t_ncol) % 2 == 1

    R = 0 if anchor in (0, 1) else hi - dh
    C = 0 if anchor in (0, 2) else wi - dw

    cur = [[int(I[R + i, C + j]) for j in range(dw)] for i in range(dh)]

    def pos(i, j):
        return (i, j) if crop_first else (R + i, C + j)

    first_seen = {}
    for i in range(dh):
        for j in range(dw):
            first_seen.setdefault(T[i][j], i * dw + j)

    groups = []   # (colour, [cells in picture coords]) in the order they are painted
    anything_to_do = any(cur[i][j] != T[i][j] for i in range(dh) for j in range(dw))

    if anything_to_do and style == 1:
        # lay a background down, then draw the other colours over it
        base = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if any(cur[i][j] != base for i in range(dh) for j in range(dw)):
            groups.append((base, [(i, j) for i in range(dh) for j in range(dw)]))
            for i in range(dh):
                for j in range(dw):
                    cur[i][j] = base
        for c in sorted([k for k in cnt if k != base], key=lambda k: (-cnt[k], first_seen[k])):
            cells = [(i, j) for i in range(dh) for j in range(dw)
                     if T[i][j] == c and cur[i][j] != c]
            if cells:
                groups.append((c, cells))

    elif anything_to_do and style == 2:
        # straight onto the grid, one colour at a time, in the order the
        # colours first appear in the picture
        for c in sorted(cnt, key=lambda k: first_seen[k]):
            cells = [(i, j) for i in range(dh) for j in range(dw)
                     if T[i][j] == c and cur[i][j] != c]
            if cells:
                groups.append((c, cells))

    elif anything_to_do and style == 3:
        # shape by shape: connected same-colour regions, biggest first
        seen = [[False] * dw for _ in range(dh)]
        comps = []
        for i in range(dh):
            for j in range(dw):
                if seen[i][j]:
                    continue
                c = T[i][j]
                seen[i][j] = True
                stack = [(i, j)]
                cells = []
                while stack:
                    r, q = stack.pop()
                    cells.append((r, q))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nq = r + dr, q + dc
                        if 0 <= nr < dh and 0 <= nq < dw and not seen[nr][nq] and T[nr][nq] == c:
                            seen[nr][nq] = True
                            stack.append((nr, nq))
                cells.sort()
                comps.append((c, cells))
        comps.sort(key=lambda kv: (-len(kv[1]), kv[1][0]))
        for c, cells in comps:
            todo = [(i, j) for (i, j) in cells if cur[i][j] != c]
            if todo:
                groups.append((c, todo))

    elif anything_to_do:
        # sweep, starting at the anchored corner and running away from it,
        # merging neighbouring cells of one colour into a single stroke
        rows = list(range(dh)) if R == 0 else list(range(dh - 1, -1, -1))
        cols = list(range(dw)) if C == 0 else list(range(dw - 1, -1, -1))
        lines = ([[(i, j) for j in cols] for i in rows] if axis == 0
                 else [[(i, j) for i in rows] for j in cols])
        for line in lines:
            run_c, run = None, []
            for (i, j) in line:
                c = T[i][j]
                if cur[i][j] == c:                  # already right, leave it alone
                    if run:
                        groups.append((run_c, run))
                        run, run_c = [], None
                    continue
                if run and c == run_c:
                    run.append((i, j))
                else:
                    if run:
                        groups.append((run_c, run))
                    run_c, run = c, [(i, j)]
            if run:
                groups.append((run_c, run))

    # ---- emit ----
    ops, sels = [], []
    # the resize selection is exactly the full d x d rectangle of the window
    crop_sel = [R, C, dh - 1, dw - 1]

    if crop_first:
        ops.append(33)
        sels.append(crop_sel)
    for c, cells in groups:
        ops.append(int(c))
        sels.append(sel_of([pos(int(i), int(j)) for (i, j) in cells]))
    if not crop_first:
        ops.append(33)
        sels.append(crop_sel)

    ops.append(34)
    sels.append([0, 0, dh - 1, dw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5614dbcf"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5614dbcf"
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
                                f"for task 5614dbcf"
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
                    f"Failed to build a complete episode for task 5614dbcf "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5614dbcf-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
