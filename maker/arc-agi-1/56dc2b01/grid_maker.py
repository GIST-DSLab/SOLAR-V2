"""
ARC Task: 56dc2b01 (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------- #
# Discrete structural variants: bar orientation + which side the
# object sits on.  Base construction = vertical bar, object to the
# RIGHT of it; the other three are reached by mirroring.
# ---------------------------------------------------------------- #
VARIANTS = [
    {"axis": "v", "side": "right"},   # identity
    {"axis": "v", "side": "left"},    # vmirror
    {"axis": "h", "side": "below"},   # dmirror
    {"axis": "h", "side": "above"},   # dmirror then hmirror
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc = random.choice(cols)
    # objc must be non-zero: ARCLE object-grab only picks up nonzero cells
    objc = random.choice([c for c in cols if c != bgc and c != 0])

    n_ex = num_examples if num_examples else 4
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, axis=None, side=None) -> dict:
    if axis is None or side is None:
        v = random.choice(VARIANTS)
        axis, side = v["axis"], v["side"]

    while True:
        h = unifint(diff_lb, diff_ub, (4, max_h))
        w = unifint(diff_lb, diff_ub, (6, max_w))
        oh = unifint(diff_lb, diff_ub, (1, h))
        ow = unifint(diff_lb, diff_ub, (1, (w - 1) // 2 - 1))
        bb = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bb))
        obj = {sp}
        bb = remove(sp, bb)
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(0, ncells), oh * ow - 1)
        for k in range(ncells):
            obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        loci = randint(0, h - oh)
        locj = unifint(diff_lb, diff_ub, (1, w - ow))
        barlocji = unifint(diff_lb, diff_ub, (0, locj))
        barlocj = locj - barlocji
        barlocj = min(max(0, barlocj), locj - 1)
        # the 8-frontier must actually fit on the canvas, else I == O
        if barlocj + ow + 1 > w - 1:
            continue
        break

    gi = canvas(bgc, (h, w))
    gi = fill(gi, 2, connect((0, barlocj), (h - 1, barlocj)))
    go = fill(gi, objc, shift(obj, (loci, barlocj + 1)))
    go = fill(go, 8, connect((0, barlocj + ow + 1), (h - 1, barlocj + ow + 1)))
    gi = fill(gi, objc, shift(obj, (loci, locj)))

    if axis == "v":
        if side == "left":
            gi, go = vmirror(gi), vmirror(go)
    else:
        gi, go = dmirror(gi), dmirror(go)
        if side == "above":
            gi, go = hmirror(gi), hmirror(go)

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = the colour the generator paints the canvas with; it is a
    # strict majority here (object width < half the grid).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    twos = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == 2]
    two_rows = set(r for r, c in twos)
    two_cols = set(c for r, c in twos)
    vertical_bar = (len(two_cols) == 1)

    obj = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] not in (bgc, 2)]
    r0 = min(r for r, c in obj); r1 = max(r for r, c in obj)
    c0 = min(c for r, c in obj); c1 = max(c for r, c in obj)

    if vertical_bar:
        b = next(iter(two_cols))
        ow = c1 - c0 + 1
        if c0 > b:                      # object right of the bar -> slide left
            steps = c0 - (b + 1)
            move_op, dstep = 23, (0, -1)
            line_idx = b + ow + 1
        else:                           # object left of the bar -> slide right
            steps = (b - 1) - c1
            move_op, dstep = 22, (0, 1)
            line_idx = b - ow - 1
        line_cells = [(r, line_idx) for r in range(hi) if 0 <= line_idx < wi]
    else:
        b = next(iter(two_rows))
        oh = r1 - r0 + 1
        if r0 > b:                      # object below the bar -> slide up
            steps = r0 - (b + 1)
            move_op, dstep = 20, (-1, 0)
            line_idx = b + oh + 1
        else:                           # object above the bar -> slide down
            steps = (b - 1) - r1
            move_op, dstep = 21, (1, 0)
            line_idx = b - oh - 1
        line_cells = [(line_idx, c) for c in range(wi) if 0 <= line_idx < hi]

    # 1) slide the object until it touches the 2-line: one grab, then empties
    cur = list(obj)
    if steps > 0:
        ops.append(move_op); sels.append(sel_of(cur))          # grab the object
        cur = [(r + dstep[0], c + dstep[1]) for r, c in cur]
        for _ in range(steps - 1):
            ops.append(move_op); sels.append(sel_of([]))        # keep it grabbed
            cur = [(r + dstep[0], c + dstep[1]) for r, c in cur]

        # 2) repair only the footprint the object no longer covers (ARCLE
        #    zeroed the grabbed cells; the path it glided over is restored)
        hole = sorted(set(obj) - set(cur))
        if bgc != 0 and hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))

    # 3) draw the 8 frontier just beyond the object's far edge
    if line_cells:
        ops.append(8); sels.append(sel_of(line_cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
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
                                f"for task 56dc2b01"
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
                    f"Failed to build a complete episode for task 56dc2b01 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"56dc2b01-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
