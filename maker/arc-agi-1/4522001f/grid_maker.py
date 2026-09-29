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


def derive_operations(I, O=None, examples=None):
    """
    Rule (from the demonstrations): the canvas grows to 3h x 3w. Each input seed is a
    2x2 block: 3 cells of the square colour + 1 dot. All seeds share one orientation; the
    dot's corner is the diagonal direction the output grows in. The picture is anchored in
    the canvas corner opposite the dot's direction, i.e. shifted by ((1-dr)*2h, (1-dc)*2w)
    where (dr, dc) is the dot's corner inside its block. Each seed becomes a 4x4 square
    starting at its anti-dot corner and growing toward the dot, plus a second 4x4 square
    4 cells further along that diagonal. Everything else is background.
    Ops: resize -> paint new canvas background -> erase seeds the shift vacates ->
    paint each seed's two squares.  Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = 3 * hi, 3 * wi

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = [c for c, _ in cnt.most_common() if c != bgc]
    sqc = nonbg[0]                                   # 3 cells per seed
    dotc = nonbg[1] if len(nonbg) > 1 else None      # 1 cell per seed

    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]
    blocks = []
    for (rd, cd) in dots:
        dr = 1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0
        dc = 1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0
        blocks.append((rd - dr, cd - dc, dr, dc))
    DR = Counter(b[2] for b in blocks).most_common(1)[0][0]
    DC = Counter(b[3] for b in blocks).most_common(1)[0][0]
    offr, offc = (1 - DR) * 2 * hi, (1 - DC) * 2 * wi
    sr, sc = 2 * DR - 1, 2 * DC - 1

    def squares(i, j):
        ar = i + 1 - DR + offr
        ac = j + 1 - DC + offc
        cells = set()
        for k in (0, 1):
            ra, ca = ar + 4 * k * sr, ac + 4 * k * sc
            for a in range(4):
                for b in range(4):
                    r, c = ra + a * sr, ca + b * sc
                    if 0 <= r < ho and 0 <= c < wo:
                        cells.add((r, c))
        return cells

    sq_cells = [squares(i, j) for (i, j, _, _) in blocks]
    all_sq = set().union(*sq_cells) if sq_cells else set()

    plan = []
    G = np.zeros((ho, wo), dtype=int)
    G[:hi, :wi] = I

    def apply(col, cells, sel):
        if not any(G[r, c] != col for r, c in cells):
            return
        for r, c in cells:
            G[r, c] = col
        plan.append((col, sel))

    ops, sels = [33], [[0, 0, ho - 1, wo - 1]]           # grow canvas to 3h x 3w
    if bgc != 0:
        # the rows / columns the resize just added become background (full rectangles)
        apply(bgc, [(r, c) for r in range(hi, ho) for c in range(wo)],
              [hi, 0, ho - 1 - hi, wo - 1])
        apply(bgc, [(r, c) for r in range(hi) for c in range(wi, wo)],
              [0, wi, hi - 1, wo - 1 - wi])

    for (i, j, _, _), sqs in zip(blocks, sq_cells):
        seed = [(r, c) for r in (i, i + 1) for c in (j, j + 1)]
        if offr or offc:
            # the shift vacates the seed's original spot (unless a square lands there)
            vac = [p for p in seed if p not in all_sq]
            if vac:
                apply(bgc, vac, sel_of(vac))
        cells = sorted(sqs)
        apply(sqc, cells, sel_of(cells))

    # self-check on my own plan: drop any op whose removal leaves the result unchanged
    def run(pl):
        g = np.zeros((ho, wo), dtype=int)
        g[:hi, :wi] = I
        for col, sel in pl:
            if isinstance(sel, dict):
                for r, c in sel["cells"]:
                    g[r, c] = col
            else:
                r, c, h, w = sel
                g[r:r + h + 1, c:c + w + 1] = col
        return g
    final = run(plan)
    k = 0
    while k < len(plan):
        trial = plan[:k] + plan[k + 1:]
        if np.array_equal(run(trial), final):
            plan = trial
        else:
            k += 1

    for col, sel in plan:
        ops.append(col); sels.append(sel)
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
