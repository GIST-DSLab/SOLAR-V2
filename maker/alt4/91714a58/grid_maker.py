"""
ARC Task: 91714a58 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # targc must be non-zero: the block is preserved through the clipboard, and
    # ARCLE's Copy/Paste treat 0 as "nothing", so a 0-coloured block could not be stamped back.
    targc = random.choice([c for c in cols if c != bgc and c != 0])
    return {"bgc": bgc, "targc": targc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, targc) -> dict:
    def _best_block(grid, back):
        # largest solid axis-aligned rectangle (h>=2, w>=2) of a single non-background colour
        gh = len(grid)
        gw = len(grid[0])
        colors = set()
        for row in grid:
            for v in row:
                if v != back:
                    colors.add(v)
        best = None
        best_area = 0
        for col in colors:
            heights = [0] * gw
            for r in range(gh):
                for c in range(gw):
                    heights[c] = heights[c] + 1 if grid[r][c] == col else 0
                stack = []
                for c in range(gw + 1):
                    cur = heights[c] if c < gw else 0
                    start = c
                    while stack and stack[-1][1] >= cur:
                        st, ht = stack.pop()
                        wd = c - st
                        if ht >= 2 and wd >= 2:
                            area = ht * wd
                            if area > best_area:
                                best_area = area
                                best = (r - ht + 1, st, ht, wd, col)
                        start = st
                    stack.append((start, cur))
        return best

    cols = interval(0, 10, 1)
    lo_h = min(6, max_h)
    lo_w = min(6, max_w)
    remcols = remove(bgc, cols)

    result = None
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (lo_h, max_h))
        w = unifint(diff_lb, diff_ub, (lo_w, max_w))
        nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
        gi = canvas(bgc, (h, w))
        inds = totuple(asindices(gi))
        noise = sample(inds, nnoise)
        ih = randint(2, max(2, h // 2))
        iw = randint(2, max(2, w // 2))
        loci = randint(0, h - ih)
        locj = randint(0, w - iw)
        bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
        go = fill(gi, targc, bd)
        for ij in noise:
            col = choice(remcols)
            gi = fill(gi, col, {ij})
        gi = fill(gi, targc, bd)
        result = {'input': gi, 'output': go}
        # guard: the planted block must be the unique largest solid rectangle,
        # otherwise the instance would be genuinely ambiguous
        if _best_block(gi, bgc) == (loci, locj, ih, iw, targc):
            return result
    return result


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- background colour: shared by every instance of the episode ---
    cnt = Counter()
    if examples:
        for ex in examples:
            try:
                ei, eo = ex[0], ex[1]
            except Exception:
                continue
            cnt.update(np.asarray(eo, dtype=int).flatten().tolist())
    if not cnt:
        cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # --- the block: largest solid rectangle (>=2x2) of one non-background colour ---
    def best_block(grid, back):
        gh, gw = grid.shape
        colors = set(int(v) for v in np.unique(grid) if int(v) != back)
        best = None
        best_area = 0
        for col in colors:
            heights = [0] * gw
            for r in range(gh):
                for c in range(gw):
                    heights[c] = heights[c] + 1 if int(grid[r, c]) == col else 0
                stack = []
                for c in range(gw + 1):
                    cur = heights[c] if c < gw else 0
                    start = c
                    while stack and stack[-1][1] >= cur:
                        st, ht = stack.pop()
                        wd = c - st
                        if ht >= 2 and wd >= 2:
                            area = ht * wd
                            if area > best_area:
                                best_area = area
                                best = (r - ht + 1, st, ht, wd, col)
                        start = st
                    stack.append((start, cur))
        return best

    blk = best_block(I, bgc)
    ops, sels = [], []

    if blk is None:
        ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
        return ops, sels

    br, bc, bh, bw, bcol = blk
    block_cells = [(r, c) for r in range(br, br + bh) for c in range(bc, bc + bw)]

    if bcol != 0:
        # 1. take the block onto the clipboard (exactly the block's rectangle)
        ops.append(28); sels.append(sel_of(block_cells))
        # 2. wipe the drawing: every cell that is not already background
        drawn = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
        ops.append(int(bgc)); sels.append(sel_of(drawn))
        # 3. stamp the block back where it stood
        ops.append(30); sels.append(sel_of([(br, bc)]))
    else:
        # 0-coloured block cannot travel through the clipboard: clear around it instead
        keep = set(block_cells)
        drawn = [(r, c) for r in range(hi) for c in range(wi)
                 if int(I[r, c]) != bgc and (r, c) not in keep]
        if drawn:
            ops.append(int(bgc)); sels.append(sel_of(drawn))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 91714a58"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 91714a58"
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
                                f"for task 91714a58"
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
                    f"Failed to build a complete episode for task 91714a58 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"91714a58-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
