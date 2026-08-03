"""
ARC Task: 3eda0437 (RE-ARC) — LLM-generated grid_maker
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
    # Only the hole color (0) and the marker color (6) matter to the rule; the
    # foreground palette is decorative, but keep it fixed across the episode.
    cols = [c for c in range(1, 10) if c != 6]
    palette = list(sample(cols, len(cols)))
    return {"palette": palette}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, palette=None) -> dict:
    if palette is None:
        palette = list(sample([c for c in range(1, 10) if c != 6], 8))

    swap = choice((True, False))
    if swap:
        hb, wb = min(8, max_w), min(30, max_h)
    else:
        hb, wb = min(8, max_h), min(30, max_w)
    hb = max(3, hb)
    wb = max(3, wb)

    h = unifint(diff_lb, diff_ub, (3, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))
    if swap:
        h, w = w, h

    ncols = unifint(diff_lb, diff_ub, (1, min(8, len(palette))))
    fgcs = palette[:ncols]

    gi = canvas(-1, (h, w))
    gi = paint(gi, {(choice(fgcs), ij) for ij in asindices(gi)})
    spac = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = asindices(gi)
    obj = sample(totuple(inds), spac)
    gi = fill(gi, 0, obj)
    locx = (randint(0, h - 1), randint(0, w - 1))
    gi = fill(gi, 0, {locx, add(locx, RIGHT), add(locx, DOWN), add(locx, UNITY)})

    maxsiz = -1
    mapper = dict()
    maxpossw = max([r.count(0) for r in gi])
    maxpossh = max([c.count(0) for c in dmirror(gi)])
    for a in range(2, maxpossh + 1):
        for b in range(2, maxpossw + 1):
            siz = a * b
            if siz < maxsiz:
                continue
            objx = recolor(0, asindices(canvas(-1, (a, b))))
            occs = occurrences(gi, objx)
            if len(occs) > 0:
                if siz == maxsiz:
                    mapper[objx] = occs
                elif siz > maxsiz:
                    mapper = {objx: occs}
                    maxsiz = siz
    go = tuple(e for e in gi)
    for obj, locs in mapper.items():
        go = fill(go, 6, mapply(lbind(shift, obj), locs))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Rule (read off I alone): among all all-zero rectangles a x b with a,b >= 2
    # that occur in I, take the maximal area; every occurrence of every shape at
    # that area becomes a solid 6 block.
    Z = (I == 0).astype(int)
    P = np.zeros((hi + 1, wi + 1), dtype=int)
    P[1:, 1:] = Z.cumsum(0).cumsum(1)

    def zsum(r, c, a, b):
        return int(P[r + a, c + b] - P[r, c + b] - P[r + a, c] + P[r, c])

    maxpossw = max(int(Z[r, :].sum()) for r in range(hi))   # zeros per row
    maxpossh = max(int(Z[:, c].sum()) for c in range(wi))   # zeros per column

    best_area = 0
    rects = []
    for a in range(2, min(maxpossh, hi) + 1):
        for b in range(2, min(maxpossw, wi) + 1):
            area = a * b
            if area < best_area:
                continue
            occ = [(r, c)
                   for r in range(hi - a + 1)
                   for c in range(wi - b + 1)
                   if zsum(r, c, a, b) == area]
            if not occ:
                continue
            if area > best_area:
                best_area = area
                rects = []
            rects.extend([(r, c, a, b) for (r, c) in occ])

    ops, sels = [], []
    painted = set()
    for (r, c, a, b) in rects:
        cells = {(rr, cc) for rr in range(r, r + a) for cc in range(c, c + b)}
        if cells <= painted:
            continue  # this block is already fully 6 from an overlapping occurrence
        ops.append(6)
        sels.append([r, c, a - 1, b - 1])
        painted |= cells

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
                        f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
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
                                f"for task 3eda0437"
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
                    f"Failed to build a complete episode for task 3eda0437 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3eda0437-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
