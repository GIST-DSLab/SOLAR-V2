"""
ARC Task: e50d258f (RE-ARC) — LLM-generated grid_maker
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
    # The generator samples exactly two colors at the episode level: the canvas
    # background `bgc` and `padcol` (removed from the palette the blocks draw from).
    # Colour 2 is hardcoded (it is removed from the palette and used as the marker),
    # so it must NOT be re-sampled here.  The per-block colours are irrelevant to the
    # rule (only the COUNT of 2s matters), so they stay random inside generate().
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    padcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "padcol": padcol}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, padcol=None, **kwargs) -> dict:
    cols = remove(2, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(cols)
    remcols0 = remove(bgc, cols)
    if padcol is None:
        padcol = choice(remcols0)
    remcols0 = remove(padcol, remcols0) if padcol in remcols0 else remcols0

    hlo, hhi = min(10, max_h), max_h
    wlo, whi = min(10, max_w), max_w

    go = None
    gi = None
    while go is None:
        h = unifint(diff_lb, diff_ub, (hlo, hhi))
        w = unifint(diff_lb, diff_ub, (wlo, whi))
        gi = canvas(bgc, (h, w))
        num = unifint(diff_lb, diff_ub, (1, 10))
        indss = asindices(gi)
        remcols = remcols0
        maxtrials = 4 * num
        tr = 0
        succ = 0
        bound = None
        go = None
        while succ < num and tr <= maxtrials:
            if len(remcols) == 0 or len(indss) == 0:
                break
            oh = randint(3, 8)
            ow = randint(3, 8)
            subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
            if len(subs) == 0:
                tr += 1
                continue
            loci, locj = choice(subs)
            obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
            bd = backdrop(obj)
            if bd.issubset(indss):
                numcc = unifint(diff_lb, diff_ub, (1, min(7, len(remcols))))
                ccols = sample(remcols, numcc)
                if succ == 0:
                    numred = unifint(diff_lb, diff_ub, (1, oh * ow))
                    bound = numred
                else:
                    numred = unifint(diff_lb, diff_ub, (0, min(oh * ow, bound - 1)))
                cc = canvas(choice(ccols), (oh, ow))
                cci = asindices(cc)
                subs = sample(totuple(cci), numred)
                obj1 = {(choice(ccols), ij) for ij in cci - set(subs)}
                obj2 = {(2, ij) for ij in subs}
                obj = obj1 | obj2
                gi = paint(gi, shift(obj, (loci, locj)))
                if go is None:
                    go = paint(cc, obj)
                succ += 1
                indss = (indss - bd) - outbox(bd)
            tr += 1

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # The rule: among the solid rectangular blocks scattered on the background,
    # find the one holding the most cells of colour 2, and keep only that block.
    #
    # The reviewer asked whether this route admits an honest second version.  It does
    # not.  There is exactly one region to deal with (the winning block), nothing is
    # swept in a direction, nothing is drawn on top of anything, and no object is
    # relocated: the whole transformation is "the canvas becomes this rectangle".
    # That is one CropGrid, then Submit.  Any alternative route -- trimming rows and
    # then columns in two crops, or erasing the losing blocks before cropping -- adds
    # an operation the answer is reached without (the crop discards those cells
    # regardless), which is exactly the padding the instructions forbid.  So the
    # function is returned unchanged.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = dominant color on the grid border (generator paints canvas with bgc)
    border = (list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1]))
    bgc = Counter(border).most_common(1)[0][0]

    # blocks are solid rectangles of non-bgc cells, separated by >=1 bgc cell
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    best_cnt = -1
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0] or I[r0, c0] == bgc:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < hi and 0 <= cc < wi and not seen[rr, cc] and I[rr, cc] != bgc:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
            cnt = sum(1 for (r, c) in cells if I[r, c] == 2)
            if cnt > best_cnt:
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                best_cnt = cnt
                best = (min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    br, bc, bh, bw = best

    ops, sels = [], []
    # crop canvas down to that block (selection is exactly the full rectangle)
    ops.append(33); sels.append([br, bc, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e50d258f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e50d258f"
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
                                f"for task e50d258f"
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
                    f"Failed to build a complete episode for task e50d258f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e50d258f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
