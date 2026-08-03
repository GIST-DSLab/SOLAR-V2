"""
ARC Task: 1b60fb0c (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# 1. colors: generator samples bgc and objc from cols = [0..9] minus 2
#    (2 is reserved for the completed quarter). Fix both per episode.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    import random
    cols = [c for c in range(10) if c != 2]
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


# ----------------------------------------------------------------------------
# 2. generator: 4-fold (90 deg) rotationally symmetric object built from one
#    random quadrant; one quadrant's exclusive part is missing in the input and
#    is color 2 in the output.
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, objc: int) -> dict:
    from random import randint, choice, sample

    hcap = max(10, min(30, int(max_h)))
    wcap = max(10, min(30, int(max_w)))

    h = unifint(diff_lb, diff_ub, (10, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
    odh = unifint(diff_lb, diff_ub, (2, min(h, w) // 2))
    loci = randint(0, h - 2 * odh)
    locj = randint(0, w - 2 * odh)
    loc = (loci, locj)

    quad = canvas(bgc, (odh, odh))
    ncellsd = unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
    ncells = choice((ncellsd, odh ** 2 - ncellsd))
    ncells = min(max(1, ncells), odh ** 2 - 1)
    cells = sample(totuple(asindices(canvas(-1, (odh, odh)))), ncells)
    g1 = fill(quad, objc, cells)
    g2 = rot90(g1)
    g3 = rot90(g2)
    g4 = rot90(g3)
    c1 = shift(ofcolor(g1, objc), (0, 0))
    c2 = shift(ofcolor(g2, objc), (0, odh))
    c3 = shift(ofcolor(g3, objc), (odh, odh))
    c4 = shift(ofcolor(g4, objc), (odh, 0))
    shftamt = randint(0, odh)
    c1 = shift(c1, (0, shftamt))
    c2 = shift(c2, (shftamt, 0))
    c3 = shift(c3, (0, -shftamt))
    c4 = shift(c4, (-shftamt, 0))
    cs = (c1, c2, c3, c4)
    rempart = choice(cs)
    inobjparts = remove(rempart, cs)
    inobj = merge(set(inobjparts))
    rempart = rempart - inobj
    inobj = shift(inobj, loc)
    rempart = shift(rempart, loc)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, inobj)
    go = fill(gi, 2, rempart)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
#
# RULE (measured from I only):
#   The full object F is invariant under a 90 degree CW rotation about the
#   centre of its (even-sided, square) bounding box.  A CW rotation about a
#   half-integer centre is exactly the integer map  (r, c) -> (c + p, q - r)
#   with p + q odd.  The input holds F minus the exclusive part R of ONE
#   quadrant, so  rot(X) \ X == R  for the correct (p, q).
#   We therefore SEARCH I for the (p, q) whose rotation makes X completable:
#   X u rot(X) must itself be rotation-invariant and lie inside the grid.
#   Among valid centres take the one with maximal overlap |X n rot(X)|
#   (= smallest added quarter), exactly as the task's argmax rule.
#   R (the missing rotational quarter) is then painted with a single Color2
#   op selecting that quarter's true cells.  O is never read to build R.
# ----------------------------------------------------------------------------
def _rot_cw(cells, p, q):
    return {(c + p, q - r) for (r, c) in cells}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # degenerate instance: nothing was missing
    if I.shape == O.shape and np.array_equal(I, O):
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    cnt = Counter(I.flatten().tolist())
    # object colour = the non-background one; background is the bulk colour,
    # so try the colours least-frequent-first and keep the first that admits
    # a valid rotational completion.
    color_order = [c for c, _ in sorted(cnt.items(), key=lambda kv: kv[1])]

    found = None
    for objc in color_order:
        X = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == objc}
        if len(X) < 2:
            continue
        rs = [r for r, _ in X]
        cs = [c for _, c in X]
        # rotated row  = c + p  must stay in [0, hi-1]
        p_lo, p_hi = -min(cs), hi - 1 - max(cs)
        # rotated col  = q - r  must stay in [0, wi-1]
        q_lo, q_hi = max(rs), wi - 1 + min(rs)

        best = None
        for p in range(p_lo, p_hi + 1):
            for q in range(q_lo, q_hi + 1):
                if (p + q) % 2 == 0:      # centre must be half-integer
                    continue
                RX = _rot_cw(X, p, q)
                R = RX - X
                if not R:                 # nothing would be added
                    continue
                F = X | RX
                if _rot_cw(F, p, q) != F:  # completed object not 4-fold symmetric
                    continue
                if any(not (0 <= r < hi and 0 <= c < wi) for r, c in F):
                    continue
                score = (len(R), len(X))
                if best is None or score < best[0]:
                    best = (score, R)
        if best is not None:
            found = best[1]
            break

    if found is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # paint the missing rotational quarter (one semantic object) with color 2
    ops.append(2); sels.append(sel_of(sorted(found)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b60fb0c"
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
                                f"for task 1b60fb0c"
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
                    f"Failed to build a complete episode for task 1b60fb0c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b60fb0c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
