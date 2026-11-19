"""
ARC Task: b0c4d837 (RE-ARC) — LLM-generated grid_maker
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

ROT_VARIANTS = [{"rot": "identity"}, {"rot": "rot90"}, {"rot": "rot180"}, {"rot": "rot270"}]


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, fillc = random.sample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [dict(v) for v in ROT_VARIANTS]
        examples += [dict(random.choice(ROT_VARIANTS)) for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, fillc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(("identity", "rot90", "rot180", "rot270"))
    if rot in ("rot90", "rot270"):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    h = unifint(diff_lb, diff_ub, (5, max(5, hb)))
    w = unifint(diff_lb, diff_ub, (5, max(5, wb)))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh - 1, ow - 2))
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow - 2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(asobject(subg), (loci, locj)))
    go = repeat(fillc, ntofill) + repeat(bgc, 9 - ntofill)
    go = (go[:3], go[3:6][::-1], go[6:])
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Reading: count the empty (background) lines left inside the open box -> n.
    Lay a 1x9 tally strip: n cells of the fill colour, then background.  Fold the
    strip into 3x3: cells 3..5 are reversed (FlipH) and dropped into row 1,
    cells 6..8 are dropped into row 2.  Then crop to the 3x3 answer.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # colour roles by bounding-box area: background > frame > fill
    info = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        info.append((int((r1 - r0 + 1) * (c1 - c0 + 1)), int(col),
                     (int(r0), int(c0), int(r1), int(c1))))
    info.sort(key=lambda t: -t[0])
    bgc = info[0][1]
    boxc, bbox = info[1][1], info[1][2]
    fillc, fbox = info[2][1], info[2][2]

    r0, c0, r1, c1 = bbox
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    if left_full and right_full:
        n = (r1 - r0 + 1) - (fbox[2] - fbox[0] + 1) - 1
    else:
        n = (c1 - c0 + 1) - (fbox[3] - fbox[1] + 1) - 1
    n = max(0, min(9, n))

    strip_cols = [fillc] * n + [bgc] * (9 - n)   # the 1x9 tally
    # Paste cannot carry colour 0: the 0 colour (if any) is the painted base.
    if fillc != 0:
        base_col, rep_col = bgc, fillc
        src = (fbox[0], fbox[1])
    else:
        base_col, rep_col = fillc, bgc
        bg_cells = np.argwhere(I == bgc)
        src = (int(bg_cells[0][0]), int(bg_cells[0][1]))
    rep_idx = [k for k in range(9) if strip_cols[k] == rep_col]

    ops, sels = [], []
    out = [(r, c) for r in range(3) for c in range(3)]
    snake = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]

    # thirds of the strip that carry a mark and so must be folded down
    moving = {a: any(strip_cols[k] != base_col for k in range(a, a + 3)) for a in (3, 6)}
    # folding table: the strip row plus the rows it folds into
    table = [(r, c) for r in range(3) for c in range(9)]   # full 3x9 rectangle

    def fold_plan():
        steps = []
        for a, drow in ((3, 1), (6, 2)):
            if not moving[a]:
                continue
            seg = strip_cols[a:a + 3]
            steps.append((a, drow, seg != seg[::-1] and drow == 1))
        return steps

    def run(with_base):
        # forward replay of the table ops on I alone (no answer involved);
        # also reports whether some fold step would leave the grid unchanged
        g = np.zeros((max(hi, 3), max(wi, 9)), dtype=int)
        g[:hi, :wi] = I
        if with_base:
            for (r, c) in table:
                g[r, c] = base_col
        for k in rep_idx:
            g[0, k] = rep_col
        stalled = False
        for a, drow, flip in fold_plan():
            obj = {(0, a + j): int(g[0, a + j]) for j in range(3) if g[0, a + j] != 0}
            snap = g.copy()
            for (r, c) in obj:
                snap[r, c] = 0
            moves = []
            if flip:
                moves.append(lambda o: {(r, 2 * a + 2 - c): v for (r, c), v in o.items()})
            moves += [lambda o: {(r + 1, c): v for (r, c), v in o.items()}] * drow
            moves += [lambda o: {(r, c - 1): v for (r, c), v in o.items()}] * a
            for f in moves:
                obj = f(obj)
                ng = snap.copy()
                for (r, c), v in obj.items():
                    ng[r, c] = v
                if np.array_equal(ng, g):
                    stalled = True
                g = ng
        return g[:3, :3], stalled

    fin_b, _ = run(True)
    fin_n, stall_n = run(False)
    need_base = not np.array_equal(fin_b, fin_n)
    narrow = wi < 9 or (not need_base and stall_n)

    if narrow:
        # No room for the 9-long strip (or its fold would slide invisibly over
        # identical input cells): write the strip directly in folded snake form.
        if rep_idx:
            ops.append(28); sels.append(sel_of([src]))
        ops.append(33); sels.append(sel_of(out))           # full 3x3 rectangle
        if any(I[r, c] != base_col and (r, c) not in {snake[k] for k in rep_idx}
               for (r, c) in out):
            ops.append(int(base_col)); sels.append(sel_of(out))
        for k in rep_idx:
            ops.append(30); sels.append(sel_of([snake[k]]))
        ops.append(34); sels.append(sel_of(out))
        return ops, sels

    # 1. pick up the counted colour from the input
    if rep_idx:
        ops.append(28); sels.append(sel_of([src]))

    # 2. lay the base colour over the folding table
    if need_base:
        ops.append(int(base_col)); sels.append(sel_of(table))

    # 3. write the tally strip along row 0, one paste per counted unit
    for k in rep_idx:
        ops.append(30); sels.append(sel_of([(0, k)]))

    # 4. fold: middle third reversed into row 1, last third into row 2
    for a, drow in ((3, 1), (6, 2)):
        if not moving[a]:
            continue
        seg = strip_cols[a:a + 3]
        cells = [(0, a + j) for j in range(3)]
        first = True
        if drow == 1 and seg != seg[::-1]:
            ops.append(26); sels.append(sel_of(cells))       # reverse the middle third
            first = False
        for _ in range(drow):
            ops.append(21); sels.append(sel_of(cells) if first else sel_of([]))
            first = False
        for _ in range(a):
            ops.append(23); sels.append(sel_of([]))

    # 5. crop to the folded 3x3 answer (full rectangle)
    ops.append(33); sels.append(sel_of(out))
    ops.append(34); sels.append(sel_of(out))
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
                        f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
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
                                f"for task b0c4d837"
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
                    f"Failed to build a complete episode for task b0c4d837 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b0c4d837-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
