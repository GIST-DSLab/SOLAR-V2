"""
ARC Task: a3325580 (RE-ARC) — LLM-generated grid_maker
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
    # Object colours are kept non-zero: the diagonal-mirror step uses ARCLE's
    # object mode, which treats 0 as "nothing there".
    pool = [c for c in cols if c != bgc and c != 0]
    random.shuffle(pool)
    return {"bgc": bgc, "ccols": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols) -> dict:
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    nobjs = unifint(diff_lb, diff_ub, (1, min(9, len(ccols))))
    ccols = list(ccols)
    gi = canvas(bgc, (h, w))
    lmocc = set()
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = randint(nobjs + 1, 30)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(1, 6)
        ow = randint(1, 6)
        ntr = 0
        while oh * ow < mxncells and ntr < 200:
            oh = randint(1, 6)
            ow = randint(1, 6)
            ntr += 1
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = choice(totuple(bounds))
        obj = {sp}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        if obj in seenobjs:
            continue
        obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, ccols[succ], plcd)
            succ += 1
            lmocc.add(loc[1])
    objs = objects(gi, T, F, T)
    mxncells = valmax(objs, size)
    objs = sfilter(objs, matcher(size, mxncells))
    objs = order(objs, leftmost)
    go = canvas(-1, (mxncells, len(objs)))
    for idx, o in enumerate(objs):
        go = fill(go, color(o), connect((0, idx), (mxncells - 1, idx)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule: one stripe per largest object (length = that size, ordered by the
    object's leftmost column), laid out as rows, then diagonally mirrored.
    The mirror is performed here as a real quarter turn + a real reflection.
    Every selection below is exactly the full rectangle it names."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    m, n = O.shape          # m = size of the largest objects, n = how many of them

    # --- read the largest objects out of I (colour, ordered by leftmost column) ---
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < hi and 0 <= nb < wi and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            comps.append((len(cells), min(cc for _, cc in cells), int(col)))
    stripe_colors = []
    if comps:
        mx = max(sz for sz, _, _ in comps)
        biggest = sorted([x for x in comps if x[0] == mx], key=lambda x: x[1])
        if mx == m and len(biggest) == n:
            stripe_colors = [x[2] for x in biggest]
    if len(stripe_colors) != n:                      # fallback: read them off O
        stripe_colors = [int(O[0, j]) for j in range(n)]

    ops, sels = [], []
    s = max(n, m)                                    # side of the square the mirror needs

    if m == 1 and n == 1:
        # Degenerate: the whole answer is one cell that already exists in I —
        # a mirror of a 1x1 block is that block, so there is nothing to turn.
        col = stripe_colors[0]
        hits = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        pos = hits[0] if hits else (0, 0)
        ops.append(33); sels.append([pos[0], pos[1], 0, 0])
        ops.append(34); sels.append([0, 0, 0, 0])
        return ops, sels

    # The square region the mirror acts on must fit on the canvas; only then does
    # this resize do anything (otherwise the square already fits as it is).
    if s > hi or s > wi:
        ops.append(33); sels.append([0, 0, s - 1, s - 1])

    # One horizontal stripe per largest object, in leftmost order, m cells long.
    for i, col in enumerate(stripe_colors):
        ops.append(int(col)); sels.append([i, 0, 0, m - 1])

    # Diagonal mirror, part 1: turn the whole square region a quarter turn
    # clockwise (selection = exactly that square), so every stripe stands up.
    ops.append(25); sels.append([0, 0, s - 1, s - 1])

    # Keep the turned block: it now sits at rows 0..m-1, cols s-n..s-1.
    ops.append(33); sels.append([0, s - n, m - 1, n - 1])

    # Diagonal mirror, part 2: the quarter turn left the columns in reverse
    # order; reflect them left<->right so they read in leftmost order again.
    if n > 1:
        ops.append(26); sels.append([0, 0, m - 1, n - 1])

    ops.append(34); sels.append([0, 0, m - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3325580"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3325580"
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
                                f"for task a3325580"
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
                    f"Failed to build a complete episode for task a3325580 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3325580-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
