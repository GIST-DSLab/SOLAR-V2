"""
ARC Task: 6c434453 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 8)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, ncols)
    ccols = [c for c in ccols if c != bgc and c != 2]
    nobjs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 16)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if choice((True, False)):
            oh = choice((3, 5))
            ow = choice((3, 5))
            obji = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        else:
            oh = randint(1, 5)
            ow = randint(1, 5)
            bounds = asindices(canvas(-1, (oh, ow)))
            ncells = randint(1, oh * ow)
            obji = {choice(totuple(bounds))}
            for k in range(ncells - 1):
                obji.add(choice(totuple((bounds - obji) & mapply(dneighbors, obji))))
            obji = normalize(obji)
        oh, ow = shape(obji)
        flag = obji == box(obji) and set(shape(obji)).issubset({3, 5})
        if flag:
            objo = connect((0, ow // 2), (oh - 1, ow // 2)) | connect((oh // 2, 0), (oh // 2, ow - 1))
            tocover = backdrop(obji)
        else:
            objo = obji
            tocover = obji
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcdi = shift(obji, loc)
        if plcdi.issubset(inds):
            plcdo = shift(objo, loc)
            succ += 1
            tocoveri = shift(tocover, loc)
            inds = (inds - tocoveri) - mapply(dneighbors, tocoveri)
            col = choice(ccols)
            gi = fill(gi, col, plcdi)
            go = fill(go, 2 if flag else col, plcdo)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the generator paints the canvas with bgc first; it dominates the grid.
    vals, counts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(counts)])

    # Find single-colour 4-connected components that are hollow rectangular frames
    # of size 3 or 5 per side (the generator's "box" objects).
    visited = np.zeros_like(I, dtype=bool)
    frames = []
    for r in range(hi):
        for c in range(wi):
            if visited[r, c] or I[r, c] == bgc:
                continue
            color = I[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == color:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            if oh not in (3, 5) or ow not in (3, 5):
                continue
            ring = set()
            for y in range(r0, r1 + 1):
                ring.add((y, c0)); ring.add((y, c1))
            for x in range(c0, c1 + 1):
                ring.add((r0, x)); ring.add((r1, x))
            if set(comp) == ring:
                frames.append((r0, c0, oh, ow, sorted(comp)))

    # Reading: the frame object itself turns into a plus. Recolour the whole frame
    # to 2, clear the frame cells off its centre lines, then draw the centre lines
    # through the interior.
    ops, sels = [], []
    for r0, c0, oh, ow, comp in frames:
        mr, mc = r0 + oh // 2, c0 + ow // 2
        # 1. Recolour the whole frame object to 2 (it's one connected component)
        ops.append(12); sels.append([comp[0][0], comp[0][1], 0, 0])
        # 2. Frame cells that are not on the centre row/column go back to background
        off_axis = [(y, x) for (y, x) in comp if y != mr and x != mc]
        ops.append(bgc); sels.append(sel_of(off_axis))
        # 3. Draw the centre row and column across the interior of the frame
        inner = set()
        for y in range(r0 + 1, r0 + oh - 1):
            inner.add((y, mc))
        for x in range(c0 + 1, c0 + ow - 1):
            inner.add((mr, x))
        ops.append(2); sels.append(sel_of(sorted(inner)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6c434453"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6c434453"
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
                                f"for task 6c434453"
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
                    f"Failed to build a complete episode for task 6c434453 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6c434453-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
