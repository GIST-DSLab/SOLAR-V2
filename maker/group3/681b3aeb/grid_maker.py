"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np

def sample_colors(num_examples=None) -> dict:
    bgc, ca, cb = random.sample(range(10), 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, ca=1, cb=2) -> dict:
    from dsl import interval, canvas, totuple, asindices, remove, mapply, dneighbors, height, width, fill, normalize, shape, shift, difference, sfilter
    from utils import unifint
    from random import choice, randint
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, max(2, min(8, max_h // 3))))
        wi = unifint(diff_lb, diff_ub, (2, max(2, min(8, max_w // 3))))
        h = unifint(diff_lb, diff_ub, (min(3 * hi, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(3 * wi, max_w), max_w))
        c = canvas(-1, (hi, hi))
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
        go = fill(c, ca, A)
        go = fill(go, cb, B)
        fullocs = totuple(asindices(gi))
        A = normalize(A)
        B = normalize(B)
        ha, wa = shape(A)
        hb, wb = shape(B)
        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locai = randint(0, h - ha)
                locaj = randint(0, w - wa)
                plcda = shift(A, (locai, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(remlocs2)
                plcdb = shift(B, (ch))
                if set(plcdb).issubset(set(remlocs2)):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {"input": gi, "output": go}

def derive_operations(I, O=None, examples=None):
    """Two interlocking pieces form a solid rectangle.  Instead of carrying the
    pieces together, build the rectangle directly: cut the canvas down to the
    rectangle's size, lay the whole canvas in one piece's colour (the 'base'
    piece), then paint the other piece's shape into its slot on top.
    Everything is measured from I (bgc confirmed by the example inputs)."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background colour: the dominant colour of the episode's input grids
    grids = [I]
    for pair in (examples or []):
        try:
            grids.append(np.asarray(pair[0], dtype=int))
        except Exception:
            pass
    votes = Counter(Counter(g.flatten().tolist()).most_common(1)[0][0] for g in grids)
    bgc = votes.most_common(1)[0][0]
    if not bool((I == bgc).any()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the pieces: one colour each (each piece is univalued, colours differ)
    pieces = {}
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc:
                pieces.setdefault(int(I[r, c]), []).append((r, c))
    if len(pieces) < 2:
        return [34], [[0, 0, h - 1, w - 1]]
    items = sorted(pieces.items(), key=lambda t: -len(t[1]))[:2]

    def norm(cells):
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        return {(r - r0, c - c0) for r, c in cells}

    (colB, cellsB), (colA, cellsA) = items   # B = larger piece -> base colour
    nA, nB = norm(cellsA), norm(cellsB)
    span = 2 + max(max(max(p) for p in nA), max(max(p) for p in nB))

    # relative placement that makes A ∪ B a solid rectangle without overlap
    sol = None
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            sa = {(r + dr, c + dc) for r, c in nA}
            if sa & nB:
                continue
            u = sa | nB
            r0 = min(r for r, _ in u); c0 = min(c for _, c in u)
            r1 = max(r for r, _ in u); c1 = max(c for _, c in u)
            if len(u) == (r1 - r0 + 1) * (c1 - c0 + 1):
                sol = ({(r - r0, c - c0) for r, c in sa}, r1 - r0 + 1, c1 - c0 + 1)
                break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    slotA, hr, wr = sol

    ops, sels = [], []
    # 1. cut the canvas to the rectangle's size (full rectangle selection)
    ops.append(33); sels.append([0, 0, hr - 1, wr - 1])
    grid = I[:hr, :wr].copy()          # CropGrid keeps that corner of the grid
    # 2. lay the base piece's colour over the whole canvas (full rectangle)
    if not bool((grid == colB).all()):
        ops.append(colB); sels.append([0, 0, hr - 1, wr - 1])
        grid[:, :] = colB
    # 3. paint the other piece's shape into its slot
    slot = sorted(slotA)
    if any(grid[r, c] != colA for r, c in slot):
        ops.append(colA); sels.append(sel_of(slot))
        for r, c in slot:
            grid[r, c] = colA
    ops.append(34); sels.append([0, 0, hr - 1, wr - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
