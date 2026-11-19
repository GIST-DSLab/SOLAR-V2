"""
ARC Task: b7249182 (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------- colors / plan

VARIANTS = [
    {"mirrored": False},   # figure grows top -> bottom  (portrait)
    {"mirrored": True},    # dmirror'd: figure grows left -> right (landscape)
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ca, cb = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


# ---------------------------------------------------------------- generator

def generate(diff_lb, diff_ub, max_h, max_w, bgc, ca, cb, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])

    # after dmirror the emitted grid is (w, h), so swap the caps in that case
    if mirrored:
        h_cap, w_cap = max_w, max_h
    else:
        h_cap, w_cap = max_h, max_w

    h = unifint(diff_lb, diff_ub, (7, max(7, h_cap)))
    w = unifint(diff_lb, diff_ub, (5, max(5, w_cap)))
    ih = unifint(diff_lb, diff_ub, (3, max(3, (h - 1) // 2)))

    subg = canvas(bgc, (ih, 5))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))

    subg = fill(subg, ca, connect((0, 2), (ih - 2, 2)))
    subg = fill(subg, ca, connect((ih - 2, 0), (ih - 2, 4)))
    subg = fill(subg, ca, {(ih - 1, 0)})
    subga = fill(subg, ca, {(ih - 1, 4)})
    subgb = replace(subga, ca, cb)
    subg = vconcat(subga, hmirror(subgb))

    loci = randint(0, h - 2 * ih)
    locj = randint(0, w - 5)
    obj = asobject(subg)
    obj = shift(obj, (loci, locj))

    gi = fill(gi, ca, {(loci, locj + 2)})
    gi = fill(gi, cb, {(loci + 2 * ih - 1, locj + 2)})
    go = paint(go, obj)

    if mirrored:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- derivation

def derive_operations(I, O):
    """
    Two markers sit on one axis.  The rule draws the same 'T with feet' figure
    starting at each marker: the first one in its own colour, the second one is
    the MIRROR IMAGE of that figure (generator: vconcat(fig, hmirror(fig))).
    So: draw figure A, draw figure B in the same orientation, then FLIP the
    half B lives in -- the reflection is performed, not merely reproduced.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: the canvas colour the generator paints before placing markers
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    pts = sorted([(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc])
    (ra, ca_r), (rb, cb_r) = pts[0], pts[1]
    col_a = int(I[ra, ca_r])
    col_b = int(I[rb, cb_r])

    portrait = (ca_r == cb_r)            # markers share a column -> figure runs downward
    span = (rb - ra + 1) if portrait else (cb_r - ca_r + 1)
    ih = span // 2                       # height of one half-figure

    # (u, v): u = distance along the marker axis, v = offset across it
    if portrait:
        def C(u, v):
            return (ra + u, ca_r + v)
    else:
        def C(u, v):
            return (ra + v, ca_r + u)

    G = I.copy()
    ops, sels = [], []

    def color_cells(cells, col):
        cells = [p for p in cells if G[p] != col]      # skip cells already that colour
        if not cells:
            return
        ops.append(int(col))
        sels.append(sel_of(cells))
        for p in cells:
            G[p] = col

    def figure(base, col):
        # stem running along the axis
        color_cells([C(base + u, 0) for u in range(0, ih - 1)], col)
        # crossbar across the axis
        color_cells([C(base + ih - 2, v) for v in (-2, -1, 1, 2)], col)
        # the two feet
        color_cells([C(base + ih - 1, -2), C(base + ih - 1, 2)], col)

    # 1. figure A, anchored on the first marker
    figure(0, col_a)

    # 2. the second marker is absorbed into figure B: clear it so the flip of
    #    the lower/right half has nothing stray inside it
    color_cells([C(2 * ih - 1, 0)], bgc)

    # 3. figure B drawn in the SAME orientation as A, in the second half
    figure(ih, col_b)

    # 4. reflect that half -- this is the rule
    half = [C(u, v) for u in range(ih, 2 * ih) for v in (-2, -1, 0, 1, 2)]
    # (this selection is exactly the full rectangle of the second half,
    #  background included, because the whole region is being mirrored)
    ops.append(27 if portrait else 26)   # FlipV (up/down) / FlipH (left/right)
    sels.append(sel_of(half))

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(ho) for c in range(wo)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task b7249182"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b7249182"
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
                                f"for task b7249182"
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
                    f"Failed to build a complete episode for task b7249182 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b7249182-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
