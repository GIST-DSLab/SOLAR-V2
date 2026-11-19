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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ncols = random.randint(1, 9)
    ccols = random.sample(rem, ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, ccols=None, mirror=None) -> dict:
    # output canvas is always 10 rows (fixed by the task rule)
    if bgc is None:
        bgc = choice(interval(0, 10, 1))
    if ccols is None:
        ccols = remove(bgc, interval(0, 10, 1))
    ccols = list(ccols)
    if mirror is None:
        mirror = choice((True, False))
    hcap = max(2, min(6, max_h))
    wcap = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, hcap))
        w = unifint(diff_lb, diff_ub, (8, wcap))
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
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
    if mirror:
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: the input is the top slice of a 10-row canvas on which one small
    object (the 'unit') has been stamped repeatedly at multiples of a period
    vector (dr, dc).  Trajectory: grow the canvas to 10 rows, lay the
    background base over the rows the resize added, then stamp the unit
    (CopyI of the unit from the input, Paste at k*(dr, dc)) once per period.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    # background = the colour the generator paints the canvas with (majority here)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    cellset = set(cells)

    # ---- measure the period vector (dr, dc) from I ------------------------
    cand = []
    for i in range(1, 6):
        for j in range(-10, 10):
            inter = [(r, c) for (r, c) in cells if (r - i, c - j) in cellset]
            if not inter:
                continue
            if any(I[r, c] != I[r - i, c - j] for (r, c) in inter):
                continue
            cand.append(((len(inter), i * j), (i, j)))
    cand.sort(key=lambda t: t[0], reverse=True)

    def predict(dr, dc):
        base = [(r, c) for r in range(min(dr, h)) for c in range(w) if I[r, c] != bgc]
        G = np.full((ho, wo), bgc, dtype=int)
        for k in range(0, ho + 1):
            for (r, c) in base:
                rr, cc = r + k * dr, c + k * dc
                if 0 <= rr < ho and 0 <= cc < wo:
                    G[rr, cc] = I[r, c]
        return G

    dr, dc = (cand[0][1] if cand else (1, 0))
    for _key, (i, j) in cand:
        if np.array_equal(predict(i, j), O):
            dr, dc = i, j
            break

    # ---- the repeating unit: the stamp occupying the first dr rows --------
    unit = [(r, c) for r in range(min(dr, h)) for c in range(w) if I[r, c] != bgc]
    if unit:
        c0 = min(c for _, c in unit)
        c1 = max(c for _, c in unit)
    else:
        c0, c1 = 0, 0

    ops, sels = [], []
    G = np.zeros((ho, wo), dtype=int)
    G[:h, :w] = I                      # ResizeGrid keeps the input rows as they are

    # 1. grow the canvas to the full 10-row height (selection == whole new canvas)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    # 2. lay the background base over the rows the resize just added
    if bgc != 0:
        ops.append(bgc)
        sels.append([h, 0, ho - 1 - h, wo - 1])     # exactly the added rows
        G[h:, :] = bgc

    # 3. stamp the unit at each further period offset k*(dr, dc)
    last_src = None
    for k in range(1, ho + 1):
        if k * dr > ho - 1:
            break
        rr0, rr1 = max(k * dr, h), min(k * dr + dr - 1, ho - 1)
        if rr0 > rr1:
            continue                                 # this stamp is already in I
        cc0, cc1 = max(c0 + k * dc, 0), min(c1 + k * dc, wo - 1)
        if cc0 > cc1:
            continue                                 # this stamp fell off the grid
        sr0, sr1 = rr0 - k * dr, rr1 - k * dr
        sc0, sc1 = cc0 - k * dc, cc1 - k * dc
        sub = I[sr0:sr1 + 1, sc0:sc1 + 1]
        if not (sub != bgc).any():
            continue                                 # nothing of the unit lands here

        src = (sr0, sc0, sr1, sc1)
        if src != last_src:
            ops.append(28)                           # CopyI: the unit, as it is in I
            sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
            last_src = src

        # Paste cannot carry colour-0 cells; clear the destination first so the
        # unit's 0 cells are already correct there.
        if bgc != 0 and (sub == 0).any():
            before = G.copy()
            G[rr0:rr1 + 1, cc0:cc1 + 1] = 0
            if not np.array_equal(before, G):
                ops.append(0)
                sels.append([rr0, cc0, rr1 - rr0, cc1 - cc0])   # exact dest rect

        before = G.copy()
        dest = G[rr0:rr1 + 1, cc0:cc1 + 1]
        dest[sub != 0] = sub[sub != 0]
        if not np.array_equal(before, G):
            ops.append(30)
            sels.append([rr0, cc0, 0, 0])            # paste origin

    # safety net (should not trigger): place any unit cell still missing
    if not np.array_equal(G, O):
        for col in sorted(set(O.flatten().tolist())):
            miss = [(int(r), int(c)) for r, c in zip(*np.where((G != O) & (O == col)))]
            if miss:
                ops.append(int(col))
                sels.append(sel_of(miss))
                for (r, c) in miss:
                    G[r, c] = col

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
