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

# The only structural variant is the global rotation applied to both grids: it decides
# which corner of every 2x2 block holds the dot, and therefore the diagonal direction in
# which the 4x4 squares grow. All four must be demonstrated when there are example slots.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    hlim, wlim = max_h // 3, max_w // 3
    if rot % 2 == 1:                      # rot90/rot270 swap the output dims
        hlim, wlim = max_w // 3, max_h // 3
    hb, wb = min(10, hlim), min(10, wlim)
    if hb < 3 or wb < 3:
        raise ValueError("max grid dims too small for this task")

    h = unifint(diff_lb, diff_ub, (3, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * (3 * w) for _ in range(3 * h)]

    def cells_of(loc):
        i, j = loc
        return {(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)}

    def place(loc):
        i, j = loc
        gi[i][j] = sqc
        gi[i][j + 1] = sqc
        gi[i + 1][j] = sqc
        gi[i + 1][j + 1] = dotc
        for (br, bc) in ((i, j), (i + 4, j + 4)):
            for r in range(br, br + 4):
                for c in range(bc, bc + 4):
                    go[r][c] = sqc

    def dnb(cells):
        out = set(cells)
        for (r, c) in cells:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    iinds = {(r, c) for r in range(h) for c in range(w)}
    loc = (random.randint(0, min(h - 2, 3 * h - 8)),
           random.randint(0, min(w - 2, 3 * w - 8)))
    place(loc)
    iinds -= dnb(cells_of(loc))

    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    succ, tr, maxtr = 0, 0, 10 * noccs
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = [ij for ij in iinds if ij[0] <= h - 2 and ij[1] <= w - 2]
        if not cands:
            break
        loc = random.choice(cands)
        cs = cells_of(loc)
        if cs <= iinds:
            succ += 1
            iinds = iinds - dnb(cs)
            place(loc)

    def rotcw(g, k):
        for _ in range(k % 4):
            g = [list(t) for t in zip(*g[::-1])]
        return [list(r) for r in g]

    return {"input": rotcw(gi, rot), "output": rotcw(go, rot)}


def derive_operations(I, O):
    """
    Rule read off I -> O:
      * canvas grows 3x in both axes;
      * every 2x2 block (3 cells of sqc + 1 dot) points diagonally away from its dot;
      * the whole picture slides to the canvas corner the dot points at
        (offset = ((1-dr)*2h, (1-dc)*2w));
      * each block becomes a 4x4 square growing from its anti-corner in the dot's
        diagonal direction, plus a second 4x4 square 4 further along that diagonal;
      * everything else is background.
    So: expand canvas -> paint the new background -> per object: erase what it vacated,
    then draw its two squares.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    nonbg = Counter([v for v in I.flatten().tolist() if v != bgc]).most_common()
    sqc = nonbg[0][0]                       # 3 cells per block
    dotc = nonbg[1][0] if len(nonbg) > 1 else sqc

    # simulated working grid
    G = np.zeros((ho, wo), dtype=int)
    G[:hi, :wi] = I

    # 1. expand canvas to 3h x 3w (input stays at top-left)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    # 2. the newly exposed canvas is 0; make it background (skip when bgc already is 0)
    if bgc != 0:
        ops.append(bgc); sels.append([hi, 0, ho - 1 - hi, wo - 1]); G[hi:, :] = bgc
        ops.append(bgc); sels.append([0, wi, hi - 1, wo - 1 - wi]); G[:hi, wi:] = bgc

    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == dotc]

    for (rd, cd) in dots:
        # dot's corner inside its own 2x2 (blocks are never 4-adjacent, so its sqc
        # neighbours can only belong to this block)
        dr = 1 if (rd - 1 >= 0 and I[rd - 1, cd] == sqc) else 0
        dc = 1 if (cd - 1 >= 0 and I[rd, cd - 1] == sqc) else 0
        i, j = rd - dr, cd - dc

        # cells of this block that the transformation leaves empty
        need = set()
        for r in (i, i + 1):
            for c in (j, j + 1):
                if O[r, c] == bgc and G[r, c] != bgc:
                    need.add((r, c))
        if len(need) == 4:
            ops.append(bgc); sels.append([i, j, 1, 1]); G[i:i + 2, j:j + 2] = bgc
        elif need:
            for r in (i, i + 1):
                if (r, j) in need and (r, j + 1) in need:
                    ops.append(bgc); sels.append([r, j, 0, 1])
                    G[r, j:j + 2] = bgc
                    need -= {(r, j), (r, j + 1)}
            for c in (j, j + 1):
                if (i, c) in need and (i + 1, c) in need:
                    ops.append(bgc); sels.append([i, c, 1, 0])
                    G[i:i + 2, c] = bgc
                    need -= {(i, c), (i + 1, c)}
            for (r, c) in sorted(need):
                ops.append(bgc); sels.append([r, c, 0, 0]); G[r, c] = bgc

        # the block's two 4x4 squares, at the destination corner of the canvas
        sr, sc = 2 * dr - 1, 2 * dc - 1                       # diagonal the dot points to
        ar = i + 1 - dr + (1 - dr) * 2 * hi                   # anti-corner, canvas coords
        ac = j + 1 - dc + (1 - dc) * 2 * wi
        for k in (0, 1):
            ra, ca = ar + 4 * k * sr, ac + 4 * k * sc
            rb, cb = ra + 3 * sr, ca + 3 * sc
            r0, r1 = min(ra, rb), max(ra, rb)
            c0, c1 = min(ca, cb), max(ca, cb)
            if np.all(G[r0:r1 + 1, c0:c1 + 1] == sqc):        # already covered
                continue
            ops.append(sqc); sels.append([r0, c0, r1 - r0, c1 - c0])
            G[r0:r1 + 1, c0:c1 + 1] = sqc

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
