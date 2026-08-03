"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Rule = "extend the repeating object downward to 10 rows".
    # It depends only on the object's pattern/period, not on which colors it uses,
    # so only the background needs to be fixed episode-wide.
    bgc = choice(interval(0, 10, 1))
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    hub = max(2, min(6, max_h))
    wub = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, hub))
        w = unifint(diff_lb, diff_ub, (8, wub))
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w // 2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (10, w))
        hoffs = randint(0, ow // 2 + 1)
        for k in range(10 // oh + 1):
            go = paint(go, shift(plcd, (k * oh, k * hoffs)))
        if len(palette(go[h:])) > 1:
            break
    gi = go[:h]
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Background = the color the generator paints the canvas with before placing objects.
    # The object stays strictly under half the width, so bgc is the strict majority in I.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- Measure the repetition of the object from I ---------------------------
    # I is the top rows of a strip in which one object is stamped repeatedly with a
    # constant step (dr, dc). I always contains at least two full periods, so the
    # step is measurable from I alone.
    def build(dr, dc):
        unit = [(r, c, int(I[r, c]))
                for r in range(min(dr, hi)) for c in range(wi) if I[r, c] != bgc]
        G = np.full((ho, wo), bgc, dtype=int)
        copies = []
        k = 0
        while k * dr < ho:
            cells = []
            for (ur, uc, v) in unit:
                r, c = ur + k * dr, uc + k * dc
                if 0 <= r < ho and 0 <= c < wo:
                    G[r, c] = v
                    cells.append((r, c, v))
            copies.append(cells)
            k += 1
        ubb = None
        if unit:
            ubb = (min(u[0] for u in unit), max(u[0] for u in unit),
                   min(u[1] for u in unit), max(u[1] for u in unit))
        return G, copies, unit, ubb

    cands = sorted((dr, abs(dc), dc) for dr in range(1, hi) for dc in range(-(wi - 1), wi))
    found = None
    for dr, _, dc in cands:
        ok = True
        for r in range(hi - dr):
            for c in range(wi):
                cc = c + dc
                if 0 <= cc < wi and I[r + dr, cc] != I[r, c]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        G, copies, unit, ubb = build(dr, dc)
        if unit and np.array_equal(G, O):
            found = (dr, dc, copies, unit, ubb)
            break

    ops, sels = [], []

    # 1. Grow the canvas from h rows to 10 rows (the input keeps rows 0..hi-1).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    # 2. The grown area arrives as zeros; make it background (skip when bgc is already 0).
    if bgc != 0 and ho > hi:
        ops.append(int(bgc))
        sels.append([hi, 0, ho - 1 - hi, wo - 1])

    def paint_copy_cells(cells):
        new = sorted([(r, c, v) for (r, c, v) in cells if r >= hi])
        i = 0
        while i < len(new):
            r, c, v = new[i]
            j = i + 1
            while (j < len(new) and new[j][0] == r
                   and new[j][1] == new[j - 1][1] + 1 and new[j][2] == v):
                j += 1
            ops.append(int(v))
            sels.append([r, c, 0, new[j - 1][1] - c])
            i = j

    if found is None:
        # Defensive fallback: stamp the below-input region region-by-region.
        rows = [[(r, c, int(O[r, c])) for c in range(wo) if O[r, c] != bgc]
                for r in range(hi, ho)]
        for cells in rows:
            paint_copy_cells(cells)
    else:
        dr, dc, copies, unit, ubb = found
        ur0, ur1, uc0, uc1 = ubb
        fg_cols = set(I.flatten().tolist()) - {bgc}
        zero_is_fg = 0 in fg_cols

        if not zero_is_fg:
            # 3. Each further stamp is the first stamp translated by k*(dr, dc):
            #    copy it out of the input once and paste it at each new position.
            last_rect = None
            for k in range(1, len(copies)):
                sr0 = max(ur0, hi - k * dr)
                sr1 = min(ur1, ho - 1 - k * dr)
                sc0 = max(uc0, -k * dc)
                sc1 = min(uc1, wo - 1 - k * dc)
                if sr0 > sr1 or sc0 > sc1:
                    continue
                if not any(sr0 <= ur <= sr1 and sc0 <= uc <= sc1 for (ur, uc, _) in unit):
                    continue
                rect = (sr0, sc0, sr1, sc1)
                if rect != last_rect:
                    ops.append(28)
                    sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
                    last_rect = rect
                ops.append(30)
                sels.append([sr0 + k * dr, sc0 + k * dc, 0, 0])
        else:
            # 0 is object content here, so Copy/Paste would drop it:
            # stamp each further copy explicitly, one whole copy at a time.
            for k in range(1, len(copies)):
                paint_copy_cells(copies[k])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
