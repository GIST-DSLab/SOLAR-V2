"""
ARC Task: 28e73c20 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# shared geometry: the spiral walk on the INTERNAL canvas (H x W).
# The generator builds an H x W picture, draws the spiral of 3s on it and then
# throws internal column 0 away, so the visible grids are H x (W-1).
# ----------------------------------------------------------------------------
def _spiral_arms(H, W):
    """Arms (straight line segments) of the spiral, in drawing order.

    arm 0 = the whole top row; then down the right edge, left along the bottom,
    up the left edge, right again ... each arm one shorter than the matching
    previous one, until the arm length reaches 0.  Coordinates are INTERNAL.
    """
    nxt = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    arms = [[(0, c) for c in range(W)]]
    sp, direc = (0, W - 1), (1, 0)
    lw, lh, ld = W, H, H
    isverti = False
    guard = 0
    while ld > 0 and guard < 4 * (H + W) + 8:
        guard += 1
        lw -= 1
        lh -= 1
        arms.append([(sp[0] + direc[0] * k, sp[1] + direc[1] * k) for k in range(ld)])
        ep = (sp[0] + direc[0] * (ld - 1), sp[1] + direc[1] * (ld - 1))
        direc = nxt[direc]
        ld = lh if isverti else lw
        isverti = not isverti
        sp = ep
    return arms


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if a > b:
        a, b = b, a
    try:
        return unifint(diff_lb, diff_ub, (a, b))  # noqa: F821  (RE-ARC helper if present)
    except Exception:
        return random.randint(a, b)


# ----------------------------------------------------------------------------
# 1. episode palette
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """The picture is uniform noise drawn from a palette of colours (never 3,
    which is reserved for the spiral).  The rule does not depend on the palette
    at all, but fixing it keeps every instance of the episode on one scheme.
    0 is an ordinary picture colour here and is allowed to be drawn."""
    pool = [c for c in range(10) if c != 3]
    ncols = random.randint(1, 9)
    ccols = random.sample(pool, ncols)
    return {"ccols": ccols}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, ccols=None) -> dict:
    pool = [c for c in range(10) if c != 3]
    if ccols is None:
        ccols = random.sample(pool, _unifint(diff_lb, diff_ub, (1, 9)))
    ccols = [c for c in ccols if c != 3] or [random.choice(pool)]

    h = _unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, max_w)))   # visible width = w - 1

    gi = [[random.choice(ccols) for _ in range(w)] for _ in range(h)]
    go = [row[:] for row in gi]
    for arm in _spiral_arms(h, w):
        for (r, c) in arm:
            if 0 <= r < h and 0 <= c < w:
                go[r][c] = 3

    # drop internal column 0 (the generator's dmirror(dmirror(g)[1:]))
    I = tuple(tuple(row[1:]) for row in gi)
    O = tuple(tuple(row[1:]) for row in go)
    return {'input': I, 'output': O}


# ----------------------------------------------------------------------------
# 3. derivation
# ----------------------------------------------------------------------------
def derive_operations(I, O=None, examples=None):
    """Draw the spiral.  Its geometry is fixed by I's SHAPE alone; the colour it
    is drawn in is read off the worked examples (the colour present in every
    example output and absent from its input).  Nothing is read out of O.

    The ROUTE - how a person would actually lay the lines down - is chosen from
    the input itself, so the same instance always gives the same route while
    different instances get different ones:

        mode 0 : trace the spiral line by line, following the path
        mode 1 : one op per full turn of the spiral (a whole ring at a time)
        mode 2 : all the horizontal lines first, then all the vertical ones
        reverse: start at the centre and work outwards instead of outside-in

    Every op is a Color(fill) over one human-meaningful line/ring.  The route
    never Moves, Copies or Pastes anything, so picture cells holding colour 0
    (an ordinary colour in this task) are never carried and never lost.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # ---- the colour the spiral is drawn in: learned from the demonstrations --
    fill = 3
    if examples:
        for ex in examples:
            if isinstance(ex, dict):
                ei, eo = ex.get('input'), ex.get('output')
            else:
                ei, eo = ex[0], ex[1]
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            cand = set(np.unique(eo).tolist()) - set(np.unique(ei).tolist())
            if len(cand) == 1:
                fill = int(cand.pop())
                break

    # ---- the spiral's arms, mapped from the internal canvas onto the grid ----
    arms = []
    for arm in _spiral_arms(hi, wi + 1):
        seen, cells = set(), []
        for (r, c) in arm:
            rc = (r, c - 1)                      # internal column 0 is cropped
            if 0 <= rc[0] < hi and 0 <= rc[1] < wi and rc not in seen:
                seen.add(rc)
                cells.append(rc)
        if cells:
            arms.append(cells)

    # ---- pick this instance's route out of the input ------------------------
    corner = int(I[0, 0])
    tail = int(I[hi - 1, wi - 1])
    npal = len(set(I.reshape(-1).tolist()))      # how many colours the picture uses
    mode = (corner + npal) % 3
    reverse = ((hi + wi + tail) % 2 == 1)

    if mode == 0:                                # line by line along the path
        groups = [list(a) for a in arms]
    elif mode == 1:                              # one whole turn of the spiral
        groups = []
        for i in range(0, len(arms), 4):
            seen, g = set(), []
            for a in arms[i:i + 4]:
                for rc in a:
                    if rc not in seen:
                        seen.add(rc)
                        g.append(rc)
            groups.append(g)
    else:                                        # all rows, then all columns
        groups = [list(a) for a in arms[0::2]] + [list(a) for a in arms[1::2]]

    if reverse:
        groups = groups[::-1]

    # ---- drop any piece the rest of the route already covers entirely -------
    groups = [g for g in groups if g]
    changed = True
    while changed and groups:
        changed = False
        for i in range(len(groups)):
            others = set()
            for j, g in enumerate(groups):
                if j != i:
                    others.update(g)
            if set(groups[i]) <= others:
                del groups[i]
                changed = True
                break

    ops, sels = [], []
    for g in groups:
        ops.append(fill)                 # Color<fill> over this line / ring
        sels.append(sel_of(g))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 28e73c20"
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
                                f"for task 28e73c20"
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
                    f"Failed to build a complete episode for task 28e73c20 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"28e73c20-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
