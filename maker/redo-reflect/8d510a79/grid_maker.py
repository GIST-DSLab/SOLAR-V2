"""
ARC Task: 8d510a79 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- variants ---
# The generator ends with `if choice((True, False)): gi = dmirror(gi) ...`
# so the bar is either a horizontal row or a vertical column.  Both structural
# cases must be shown in the examples for the episode to be learnable.
VARIANTS_8D510A79 = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]      # 1 and 2 are hardcoded
    bgc = random.choice(cols)
    barcol = random.choice([c for c in cols if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS_8D510A79):
        examples = [dict(v) for v in VARIANTS_8D510A79]
        examples += [dict(random.choice(VARIANTS_8D510A79))
                     for _ in range(n_ex - len(VARIANTS_8D510A79))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS_8D510A79, n_ex)]
    plan = examples + [dict(random.choice(examples))]     # test mirrors a shown case
    return {"bgc": bgc, "barcol": barcol, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, barcol=None, transposed=None, **kwargs) -> dict:
    if transposed is None:
        transposed = choice((True, False))
    cols = difference(interval(0, 10, 1), (1, 2))
    if bgc is None:
        bgc = choice(cols)
    if barcol is None:
        barcol = choice(remove(bgc, cols))

    # after a dmirror the grid is (w, h): respect the caller's canvas limits
    hlim = max_w if transposed else max_h
    wlim = max_h if transposed else max_w
    hlim = max(5, min(30, int(hlim)))
    wlim = max(3, min(30, int(wlim)))

    h = unifint(diff_lb, diff_ub, (5, hlim))
    w = unifint(diff_lb, diff_ub, (3, wlim))
    barloci = randint(2, h - 3)

    gi = canvas(bgc, (h, w))
    bar = connect((barloci, 0), (barloci, w - 1))
    gi = fill(gi, barcol, bar)
    go = tuple(e for e in gi)

    jinds = interval(0, w, 1)
    numtop = unifint(diff_lb, diff_ub, (1, w - 1))
    numbot = unifint(diff_lb, diff_ub, (1, w - 1))
    tops = sample(jinds, numtop)
    bots = sample(jinds, numbot)

    for t in tops:
        loci = randint(0, barloci - 2)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (0, t)))
        else:
            go = fill(go, col, connect(loc, (barloci - 1, t)))

    for t in bots:
        loci = randint(barloci + 2, h - 1)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (h - 1, t)))
        else:
            go = fill(go, col, connect(loc, (barloci + 1, t)))

    if transposed:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: one bar spans the grid.  Every 1-marker shoots a ray AWAY from the bar
    until the grid edge; every 2-marker shoots a ray TOWARD the bar, stopping on
    the cell just before it.

    The rule is stated for a *horizontal* bar.  When the instance is the
    dmirrored one (bar vertical), the trajectory first performs that reflection
    on the whole canvas (Rotate90 + FlipV == transpose), draws every ray in the
    canonical frame, then reflects the canvas back.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- identify background and bar colour ---------------------------------
    # markers are always 1/2; background and bar are the only other colours,
    # and the background is by far the more frequent of the two.
    cnt = Counter(I.flatten().tolist())
    others = sorted([c for c in cnt if c not in (1, 2)], key=lambda c: -cnt[c])
    barcol = others[1] if len(others) > 1 else None

    bar_row, bar_col = None, None
    if barcol is not None:
        for r in range(hi):
            if bool(np.all(I[r, :] == barcol)):
                bar_row = r
                break
        if bar_row is None:
            for c in range(wi):
                if bool(np.all(I[:, c] == barcol)):
                    bar_col = c
                    break

    vertical = (bar_row is None)
    sq = max(hi, wi)

    # --- reflect the canvas so the bar lies horizontal -----------------------
    if vertical:
        if hi != wi:
            ops.append(33); sels.append([0, 0, sq - 1, sq - 1])   # square canvas
        ops.append(24); sels.append([0, 0, sq - 1, sq - 1])       # rot CCW (square sel)
        ops.append(27); sels.append([0, 0, sq - 1, sq - 1])       # flip up/down -> transpose
        if hi != wi:
            ops.append(33); sels.append([0, 0, wi - 1, hi - 1])   # crop to transposed size
        G = I.T.copy()
        b = bar_col
    else:
        G = I.copy()
        b = bar_row
    gh, gw = G.shape

    # --- draw one ray per marker, nearest the bar first ---------------------
    targets = []
    for r in range(gh):
        if r == b:
            continue
        for c in range(gw):
            v = int(G[r, c])
            if v not in (1, 2):
                continue
            if r < b:
                if v == 1:
                    cells = [(rr, c) for rr in range(0, r)]          # up to top edge
                else:
                    cells = [(rr, c) for rr in range(r + 1, b)]      # down to the bar
                side = 0
            else:
                if v == 1:
                    cells = [(rr, c) for rr in range(r + 1, gh)]     # down to bottom edge
                else:
                    cells = [(rr, c) for rr in range(b + 1, r)]      # up to the bar
                side = 1
            if cells:                                                # marker itself already holds v
                targets.append((side, abs(r - b), c, v, cells))
    targets.sort(key=lambda t: (t[0], t[1], t[2]))

    for _side, _dist, _c, v, cells in targets:
        ops.append(v)                                                # Color1 / Color2
        sels.append(sel_of(cells))

    # --- reflect the canvas back --------------------------------------------
    if vertical:
        if hi != wi:
            ops.append(33); sels.append([0, 0, sq - 1, sq - 1])
        ops.append(24); sels.append([0, 0, sq - 1, sq - 1])
        ops.append(27); sels.append([0, 0, sq - 1, sq - 1])
        if hi != wi:
            ops.append(33); sels.append([0, 0, hi - 1, wi - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8d510a79"
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
                                f"for task 8d510a79"
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
                    f"Failed to build a complete episode for task 8d510a79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8d510a79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
