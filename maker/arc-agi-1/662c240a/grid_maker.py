"""
ARC Task: 662c240a (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------------------
# Task 662c240a
# The input is a concatenation (vertical or horizontal) of n square d x d
# blocks.  Every block is built symmetric about its main diagonal except ONE,
# which has been perturbed.  The output is that one block: the block that is
# NOT equal to its own diagonal mirror.
#
# Trajectory idea (the reflection is actually performed):
#   1. expand the canvas to a square so a diagonal mirror can be done in place
#   2. mirror the WHOLE grid across its main diagonal (rotate + flip).  Every
#      symmetric block survives unchanged in its mirrored position; only the
#      special block visibly changes -> the rule is performed, not asserted.
#   3. crop to that block (it is now sitting transposed)
#   4. mirror it back across its diagonal -> the answer.
# ---------------------------------------------------------------------------

VARIANTS = [{"concat_dir": "v"}, {"concat_dir": "h"}]


def sample_colors(num_examples=None) -> dict:
    # No background colour exists for this task (the canvas is fully painted
    # with random colours) and the rule depends only on the symmetry pattern,
    # not on which colours are used.  The only discrete structural choice is
    # the concatenation direction, which is planned per instance.
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, concat_dir=None, **kwargs) -> dict:
    if concat_dir is None:
        concat_dir = random.choice(["v", "h"])

    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        if hi < lo:
            hi = lo
        if lo < a:
            lo = a
        if hi > b:
            hi = b
        return random.randint(lo, hi)

    def sym_block(d, colset):
        g = [[0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                c = random.choice(colset)
                g[i][j] = c
                g[j][i] = c
        return g

    def is_sym(g):
        d = len(g)
        for i in range(d):
            for j in range(d):
                if g[i][j] != g[j][i]:
                    return False
        return True

    vertical = (concat_dir == "v")
    long_max = min(30, max_h if vertical else max_w)
    short_max = min(30, max_w if vertical else max_h)
    d_ub = max(2, min(7, short_max, long_max // 2))

    while True:
        d = unifint(diff_lb, diff_ub, (2, d_ub))
        ng_ub = max(2, min(30 // d, long_max // d))
        ng = unifint(diff_lb, diff_ub, (2, ng_ub))
        nc = unifint(diff_lb, diff_ub, (2, min(9, d * d)))
        tcolset = random.sample(range(10), nc)

        g = sym_block(d, tcolset)

        npairs = d * (d - 1) // 2
        ndistinv = unifint(diff_lb, diff_ub, (0, max(0, npairs - 1)))
        ndist = max(1, npairs - ndistinv)
        offdiag = [(i, j) for i in range(d) for j in range(d) if i != j]
        distinds = random.sample(offdiag, min(ndist, len(offdiag)))
        for (i, j) in distinds:
            if g[i][j] == g[j][i]:
                opts = [c for c in tcolset if c != g[i][j]]
                if opts:
                    g[i][j] = random.choice(opts)
            else:
                g[i][j] = g[j][i]

        # the special block MUST end up asymmetric, otherwise the episode is
        # unsolvable (no block would stand out)
        if is_sym(g):
            continue

        out = [row[:] for row in g]
        grid = [row[:] for row in g]

        for _ in range(ng - 1):
            tset = random.sample(range(10), nc)
            b = sym_block(d, tset)
            first_new = random.choice([True, False])
            if vertical:
                grid = (b + grid) if first_new else (grid + b)
            else:
                if first_new:
                    grid = [b[r] + grid[r] for r in range(d)]
                else:
                    grid = [grid[r] + b[r] for r in range(d)]

        # sanity: exactly one block asymmetric and it is the output
        h = len(grid)
        w = len(grid[0])
        dd = min(h, w)
        vert = h > w
        n = (h // dd) if vert else (w // dd)
        asym = []
        for idx in range(n):
            if vert:
                B = [row[:] for row in grid[idx * dd:(idx + 1) * dd]]
            else:
                B = [row[idx * dd:(idx + 1) * dd] for row in grid]
            if not is_sym(B):
                asym.append(B)
        if len(asym) != 1 or asym[0] != out:
            continue

        return {"input": grid, "output": out}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    def transpose_ops(M):
        """Two ARCLE ops whose composition is the diagonal mirror of M,
        chosen so that BOTH steps visibly change the grid."""
        cands = [
            (24, 27, lambda A: np.rot90(A, 1), lambda A: np.flipud(A)),
            (25, 26, lambda A: np.rot90(A, 3), lambda A: np.fliplr(A)),
            (27, 25, lambda A: np.flipud(A), lambda A: np.rot90(A, 3)),
            (26, 24, lambda A: np.fliplr(A), lambda A: np.rot90(A, 1)),
        ]
        for o1, o2, f1, f2 in cands:
            X1 = f1(M)
            X2 = f2(X1)
            if (not np.array_equal(X1, M) and not np.array_equal(X2, X1)
                    and np.array_equal(X2, M.T)):
                return [o1, o2]
        return [24, 27]

    # --- the rule: split into d x d blocks, find the one that is NOT its own
    #     diagonal mirror -------------------------------------------------
    d = min(hi, wi)
    vertical = hi > wi
    n = (hi // d) if vertical else (wi // d)
    k = 0
    for idx in range(n):
        if vertical:
            B = I[idx * d:(idx + 1) * d, 0:d]
        else:
            B = I[0:d, idx * d:(idx + 1) * d]
        if not np.array_equal(B, B.T):
            k = idx
            break
    r0 = k * d if vertical else 0
    c0 = 0 if vertical else k * d
    B = I[r0:r0 + d, c0:c0 + d]

    ops, sels = [], []

    # 1) expand the canvas to a square so the diagonal mirror fits in place
    sq = max(hi, wi)
    P = np.zeros((sq, sq), dtype=int)
    P[:hi, :wi] = I
    ops.append(33); sels.append([0, 0, sq - 1, sq - 1])   # full rectangle: whole square canvas

    # 2) mirror the whole grid across its main diagonal.  Symmetric blocks are
    #    reproduced identically in their mirrored slot; only the special block
    #    changes -- this is the rule being carried out.
    for op in transpose_ops(P):
        ops.append(op); sels.append([0, 0, sq - 1, sq - 1])  # full rectangle: whole square canvas

    # 3) crop to that block, which now sits at the mirrored position
    ops.append(33); sels.append([c0, r0, d - 1, d - 1])   # full rectangle: the block's mirrored slot

    # 4) mirror the block back across its own diagonal -> the answer
    for op in transpose_ops(B.T):
        ops.append(op); sels.append([0, 0, d - 1, d - 1])  # full rectangle: the whole block

    ops.append(34); sels.append([0, 0, d - 1, d - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 662c240a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 662c240a"
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
                                f"for task 662c240a"
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
                    f"Failed to build a complete episode for task 662c240a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"662c240a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
