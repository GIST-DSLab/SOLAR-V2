"""
ARC Task: bda2d7a6 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

import numpy as np

from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """The rule (cyclic rotation of the nested band colours) is colour-agnostic,
    so only the episode palette is fixed here."""
    ncols = random.randint(2, 10)
    cols = random.sample(list(range(10)), ncols)
    return {"cols": cols}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, cols=None) -> dict:
    if cols is None:
        cols = random.sample(list(range(10)), random.randint(2, 10))
    cols = list(cols)

    hmax = max(2, min(14, int(max_h) // 2))
    wmax = max(2, min(14, int(max_w) // 2))

    while True:
        h = unifint(diff_lb, diff_ub, (2, hmax))
        w = unifint(diff_lb, diff_ub, (2, wmax))
        colord = [random.choice(cols) for _ in range(min(h, w))]
        # maximal runs of equal consecutive ring colours == connected components
        bands = []
        for ci in colord:
            if not bands or bands[-1] != ci:
                bands.append(ci)
        # >=2 bands  -> transformation is not the identity
        # bands[0] != bands[-1] -> avoid the verifier's "merge first & last" branch,
        #                          which would give the outermost band a different
        #                          rule and make a single episode ambiguous.
        if len(bands) >= 2 and bands[0] != bands[-1]:
            break

    shp = (h * 2, w * 2)
    gi = canvas(0, shp)
    for idx, ci in enumerate(colord):
        ulc = (idx, idx)
        lrc = (h * 2 - 1 - idx, w * 2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)

    objso = order(objects(gi, True, False, False), compose(maximum, shape))
    res = mpapply(recolor, apply(color, objso), (objso[-1],) + objso[:-1])
    go = paint(gi, res)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory
def derive_operations(I, O):
    """The grid is a stack of concentric bands; every band takes the colour of the
    band just outside it (outermost wraps to the innermost colour).  The whole
    figure is mirror-symmetric about both axes, so the recolouring is performed
    once inside the top-left quadrant and then REPLICATED:
        quadrant -> copy/paste + FlipH  -> whole top half
        top half -> copy/paste + FlipV  -> whole grid
    Destinations are cleared to 0 before each Paste, because Paste never writes
    the colour-0 cells of the copied region (0 is a legitimate band colour here).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    g = I.copy()
    ops, sels = [], []

    def emit_color(cells, val):
        cells = [(int(r), int(c)) for r, c in cells]
        if not cells:
            return
        if all(g[r, c] == val for r, c in cells):
            return                      # no visible effect -> do not emit
        ops.append(int(val))
        sels.append(sel_of(cells))
        for r, c in cells:
            g[r, c] = int(val)

    def rect_cells(r0, c0, hgt, wid):
        return [(r, c) for r in range(r0, r0 + hgt) for c in range(c0, c0 + wid)]

    def replicate(sr, sc, sh, sw, dr, dc, flip_op, axis):
        clip = g[sr:sr + sh, sc:sc + sw].copy()
        # 1. clear the destination so 0-coloured band cells come out as 0
        emit_color(rect_cells(dr, dc, sh, sw), 0)
        if not clip.any():
            return                       # nothing to replicate (all-zero source)
        # 2. copy the freshly recoloured source region from the WORKING grid
        pasted = g.copy()
        for i in range(sh):
            for j in range(sw):
                v = int(clip[i, j])
                if v != 0:
                    pasted[dr + i, dc + j] = v
        if not np.array_equal(pasted, g):
            ops.append(29)
            sels.append(sel_of(rect_cells(sr, sc, sh, sw)))
            ops.append(30)
            sels.append(sel_of([(dr, dc)]))
            g[:, :] = pasted
        # 3. mirror the freshly pasted block onto its symmetric orientation
        block = g[dr:dr + sh, dc:dc + sw]
        mirrored = np.fliplr(block) if axis == 'h' else np.flipud(block)
        if not np.array_equal(block, mirrored):
            g[dr:dr + sh, dc:dc + sw] = mirrored
            ops.append(flip_op)
            sels.append(sel_of(rect_cells(dr, dc, sh, sw)))

    # ---- connected components (the bands) of the input ------------------
    lab = -np.ones((H, W), dtype=int)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if lab[r0, c0] >= 0:
                continue
            col = int(I[r0, c0])
            k = len(comps)
            lab[r0, c0] = k
            q = deque([(r0, c0)])
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and lab[rr, cc] < 0 and int(I[rr, cc]) == col:
                        lab[rr, cc] = k
                        q.append((rr, cc))
            comps.append(cells)

    def maxdim(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    comps.sort(key=maxdim, reverse=True)   # outermost band first, then inward

    hh, ww = H // 2, W // 2
    if H % 2 == 0 and W % 2 == 0 and hh >= 1 and ww >= 1:
        # ---- recolour every band, but only inside the top-left quadrant ----
        for cells in comps:
            qc = [(r, c) for (r, c) in cells if r < hh and c < ww]
            if not qc:
                continue
            groups = {}
            for (r, c) in qc:
                groups.setdefault(int(O[r, c]), []).append((r, c))
            for val in sorted(groups, key=lambda v: -len(groups[v])):
                emit_color(groups[val], val)

        # ---- replicate the quadrant across the two mirror axes -------------
        replicate(0, 0, hh, ww, 0, ww, 26, 'h')     # top-left  -> top-right
        replicate(0, 0, hh, W, hh, 0, 27, 'v')      # top half  -> bottom half

    # ---- safety net (never fires for grids this generator produces) -------
    if not np.array_equal(g, O):
        rest = {}
        for r in range(H):
            for c in range(W):
                if g[r, c] != O[r, c]:
                    rest.setdefault(int(O[r, c]), []).append((r, c))
        for val in sorted(rest):
            emit_color(rest[val], val)

    ops.append(34)
    sels.append(sel_of(rect_cells(0, 0, H, W)))
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
                        f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bda2d7a6"
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
                                f"for task bda2d7a6"
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
                    f"Failed to build a complete episode for task bda2d7a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bda2d7a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
