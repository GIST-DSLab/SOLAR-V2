"""
ARC Task: 4522001f (RE-ARC) — LLM-generated grid_maker
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

ROTS = ['identity', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    # the rotation is a discrete structural variant (it decides which corner of the
    # 2x2 seed holds the dot, hence the diagonal direction of the two big squares)
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # output canvas is 3x the input in both dimensions (and rot90/rot270 transpose it)
    if rot in ('rot90', 'rot270'):
        hub = max(3, min(10, max_w // 3))
        wub = max(3, min(10, max_h // 3))
    else:
        hub = max(3, min(10, max_h // 3))
        wub = max(3, min(10, max_w // 3))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, wub))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (3 * h, 3 * w))
    sqi = {(dotc, (1, 1))} | recolor(sqc, {(0, 0), (0, 1), (1, 0)})
    sqo = backdrop(frozenset({(0, 0), (3, 3)}))
    sqo |= shift(sqo, (4, 4))
    loci = randint(0, min(h - 2, 3 * h - 8))
    locj = randint(0, min(w - 2, 3 * w - 8))
    loc = (loci, locj)
    plcdi = shift(sqi, loc)
    plcdo = shift(sqo, loc)
    gi = paint(gi, plcdi)
    go = fill(go, sqc, plcdo)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    iinds = ofcolor(gi, bgc) - mapply(dneighbors, toindices(plcdi))
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(iinds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcdi = shift(sqi, loc)
        plcdo = shift(sqo, loc)
        plcdii = toindices(plcdi)
        if plcdii.issubset(iinds):
            succ += 1
            iinds = (iinds - plcdii) - mapply(dneighbors, plcdii)
            gi = paint(gi, plcdi)
            go = fill(go, sqc, plcdo)
    rotf = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    ho_chk, wo_chk = np.asarray(O).shape
    hi, wi = I.shape
    # rule: the answer canvas is exactly 3x the input in each dimension
    ho, wo = 3 * hi, 3 * wi

    def components(mask):
        seen = np.zeros((hi, wi), dtype=bool)
        comps = []
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cells = []
                    while stack:
                        y, x = stack.pop()
                        cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                    comps.append(cells)
        return comps

    def analyze(bg):
        """Every non-background component must be a 2x2 seed: 3 cells of the square
        colour + 1 dot cell.  The dot's corner gives the growth direction."""
        comps = components(I != bg)
        if not comps:
            return None
        blocks = []
        sqc = None
        dd = None
        for cells in comps:
            if len(cells) != 4:
                return None
            r0 = min(y for y, x in cells)
            c0 = min(x for y, x in cells)
            if set(cells) != {(r0 + a, c0 + b) for a in (0, 1) for b in (0, 1)}:
                return None
            cnt = Counter(int(I[y, x]) for y, x in cells)
            if sorted(cnt.values()) != [1, 3]:
                return None
            sq = [k for k, v in cnt.items() if v == 3][0]
            dot = [k for k, v in cnt.items() if v == 1][0]
            dcell = [(y, x) for y, x in cells if I[y, x] == dot][0]
            d = (dcell[0] - r0, dcell[1] - c0)
            if sqc is None:
                sqc, dd = sq, d
            elif sq != sqc or d != dd:
                return None
            blocks.append((r0, c0))
        return sorted(blocks), sqc, dd[0], dd[1]

    order = [c for c, _ in Counter(I.flatten().tolist()).most_common()]
    parsed = None
    for cand in order:
        parsed = analyze(cand)
        if parsed is not None:
            bgc = cand
            break
    if parsed is None:
        bgc = order[0]
        parsed = ([], bgc, 1, 1)
    blocks, sqc, di, dj = parsed

    # where the input content lands inside the 3x canvas, per the rule
    roff = (1 - di) * 2 * hi
    coff = (1 - dj) * 2 * wi
    # diagonal offset of the second big square, from the dot's corner
    sr = 4 * (2 * di - 1)
    sc = 4 * (2 * dj - 1)

    ops, sels = [], []
    g = np.zeros((ho, wo), dtype=int)
    g[:hi, :wi] = I

    # 1. grow the canvas to 3h x 3w (full rectangle -> bbox selection)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. lay the background base
    if bgc != 0:
        ops.append(int(bgc)); sels.append([0, 0, ho - 1, wo - 1])  # full canvas rectangle
        g[:, :] = bgc
    else:
        rects = []
        for (r, c) in blocks:
            ra = r + roff - 2 * (1 - di)
            ca = c + coff - 2 * (1 - dj)
            rects.append((ra, ca))
            rects.append((ra + sr, ca + sc))
        covered = set()
        for (ra, ca) in rects:
            for a in range(4):
                for b in range(4):
                    covered.add((ra + a, ca + b))
        leftover = sorted({(r, c) for r in range(hi) for c in range(wi)
                           if I[r, c] != bgc} - covered)
        if leftover:
            ops.append(0); sels.append(sel_of(leftover))
            for (r, c) in leftover:
                g[r, c] = 0

    # 3. every 2x2 seed grows into two 4x4 squares, diagonal along the dot's corner
    for (r, c) in blocks:
        ra = r + roff - 2 * (1 - di)
        ca = c + coff - 2 * (1 - dj)
        for (br, bc) in ((ra, ca), (ra + sr, ca + sc)):
            cells = [(br + a, bc + b) for a in range(4) for b in range(4)
                     if 0 <= br + a < ho and 0 <= bc + b < wo]
            if not cells or all(g[y, x] == sqc for y, x in cells):
                continue
            ops.append(int(sqc)); sels.append(sel_of(cells))
            for (y, x) in cells:
                g[y, x] = sqc

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
                        f"num_examples+1 ({num_examples + 1}) for task 4522001f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4522001f"
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
                                f"for task 4522001f"
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
                    f"Failed to build a complete episode for task 4522001f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4522001f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
