"""
ARC Task: 28bf18c6 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, **kw) -> dict:
    from random import choice, randint

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None:
        bgc = random.randint(0, 9)
    if fgc is None:
        fgc = random.choice([c for c in range(10) if c != bgc])

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    h = max(3, min(h, max_h))
    w = max(3, min(w, max_w))
    hb = unifint(diff_lb, diff_ub, (1, min(14, h - 1)))
    # the output is 2*wb wide, so wb must also fit inside max_w // 2
    wb = unifint(diff_lb, diff_ub, (1, min(14, w - 1, max(1, max_w // 2))))
    hb = max(1, min(hb, h - 1))
    wb = max(1, min(wb, w - 1, max(1, max_w // 2)))

    bounds = {(i, j) for i in range(hb) for j in range(wb)}
    corners = [(0, 0), (0, wb - 1), (hb - 1, 0), (hb - 1, wb - 1)]
    shp = {choice(corners)}
    mp = (hb * wb) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    nc = choice((dev, hb * wb - dev))
    nc = max(0, min(hb * wb - 1, nc))

    def nbrs(p):
        i, j = p
        return {(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    for _ in range(nc):
        cand = set()
        for p in shp:
            cand |= nbrs(p)
        cand = (bounds - shp) & cand
        if not cand:
            break
        shp.add(choice(sorted(cand)))

    mi = min(i for i, j in shp)
    mj = min(j for i, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    sh = max(i for i, j in shp) + 1
    sw = max(j for i, j in shp) + 1

    di = randint(0, h - sh)
    dj = randint(0, w - sw)
    gi = [[bgc] * w for _ in range(h)]
    for i, j in shp:
        gi[i + di][j + dj] = fgc
    go1 = [[bgc] * sw for _ in range(sh)]
    for i, j in shp:
        go1[i][j] = fgc
    go = [row + row for row in go1]
    return {"input": gi, "output": go}


def _bbox_of_non(I, c):
    rs, cs = np.where(I != c)
    if len(rs) == 0:
        return None
    return rs.min(), rs.max(), cs.min(), cs.max()


def _infer_bgc(I, examples):
    # The background is fixed for the whole episode. In each demonstration it is
    # the colour whose complement's bounding box, doubled in width, gives that
    # demonstration's output shape.
    if examples:
        votes = Counter()
        for ex in examples:
            try:
                Ie, Oe = ex[0], ex[1]
            except Exception:
                Ie, Oe = ex["input"], ex["output"]
            Ie = np.asarray(Ie, dtype=int)
            Oe = np.asarray(Oe, dtype=int)
            for c in set(Ie.flatten().tolist()):
                bb = _bbox_of_non(Ie, c)
                if bb is None:
                    continue
                r0, r1, c0, c1 = bb
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if Oe.shape == (hh, 2 * ww) and np.array_equal(
                        Oe[:, :ww], Ie[r0:r1 + 1, c0:c1 + 1]):
                    votes[c] += 1
        present = set(I.flatten().tolist())
        for c, _ in votes.most_common():
            if c in present and _bbox_of_non(I, c) is not None:
                return c
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    bgc = _infer_bgc(I, examples)

    r0, r1, c0, c1 = _bbox_of_non(I, bgc)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops, sels = [], []
    # 1. Crop the canvas to the object's bounding rectangle (the whole rectangle,
    #    background included).
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
    # 2. Widen the canvas to h x 2w. The new right half is filled with 0s and
    #    will hold the copy.
    ops.append(33); sels.append([0, 0, h - 1, 2 * w - 1])
    # 3. Build the copy one row at a time: copy row i of the object's rectangle
    #    from the INPUT and paste it at (i, w) in the right half.
    #    Copy and Paste skip 0s. Because the right half already holds 0s, a 0
    #    cell in a row comes out right either way. A row with no nonzero cell
    #    would paste nothing, so it is skipped (this can be read from I alone).
    for i in range(h):
        row = I[r0 + i, c0:c1 + 1]
        if not np.any(row != 0):
            continue
        ops.append(28); sels.append([r0 + i, c0, 0, w - 1])   # CopyI: one full row
        ops.append(30); sels.append([i, w, 0, 0])             # Paste at the row's right-half start
    ops.append(34); sels.append([0, 0, h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 28bf18c6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 28bf18c6"
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
                                f"for task 28bf18c6"
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
                    f"Failed to build a complete episode for task 28bf18c6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"28bf18c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
