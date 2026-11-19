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


def sample_colors(num_examples=None) -> dict:
    # Rule is purely structural (find the block that is NOT symmetric about its
    # main diagonal), so no color role needs fixing. The one discrete structural
    # variant is the concatenation orientation -> plan it per instance.
    VARIANTS = [{"orientation": "v"}, {"orientation": "h"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]  # test variant was shown
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, orientation=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def is_sym(g):
        n = len(g)
        for i in range(n):
            for j in range(n):
                if g[i][j] != g[j][i]:
                    return False
        return True

    def make_sym_block(d, nc):
        tcolset = random.sample(list(range(10)), nc)
        g = [[0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                v = random.choice(tcolset)
                g[i][j] = v
                g[j][i] = v
        return g, tcolset

    if orientation is None:
        orientation = random.choice(["v", "h"])

    # orientation "v": input is (ng*d, d); "h": input is (d, ng*d)
    cat_lim = max_h if orientation == "v" else max_w
    side_lim = max_w if orientation == "v" else max_h

    d_ub = min(7, side_lim, cat_lim // 2)
    d_ub = max(2, d_ub)
    d = unifint(diff_lb, diff_ub, (2, d_ub))

    ng_ub = min(30 // d, cat_lim // d)
    ng_ub = max(2, ng_ub)
    ng = unifint(diff_lb, diff_ub, (2, ng_ub))

    nc = unifint(diff_lb, diff_ub, (2, min(9, d * d)))
    nc = max(2, nc)

    # ---- the answer block: symmetric, then distorted so it is NOT symmetric ----
    g, tcolset = make_sym_block(d, nc)
    npairs = d * (d - 1) // 2
    ndistinv = unifint(diff_lb, diff_ub, (0, max(0, npairs - 1)))
    ndist = max(1, npairs - ndistinv)
    offdiag = [(i, j) for i in range(d) for j in range(d) if i != j]
    ndist = min(ndist, len(offdiag))
    distinds = random.sample(offdiag, ndist)
    for (i, j) in distinds:
        if g[i][j] == g[j][i]:
            alts = [c for c in tcolset if c != g[i][j]]
            if alts:
                g[i][j] = random.choice(alts)
        else:
            g[i][j] = g[j][i]

    # distortions can cancel out -> force asymmetry so the answer is unique
    if is_sym(g):
        i, j = random.choice(offdiag)
        alts = [c for c in tcolset if c != g[j][i]]
        g[i][j] = random.choice(alts)

    go = [row[:] for row in g]

    # ---- assemble with ng-1 purely symmetric distractor blocks ----
    gi = [row[:] for row in g]
    for _ in range(ng - 1):
        blk, _tc = make_sym_block(d, nc)
        if orientation == "v":
            if random.choice((True, False)):
                gi = gi + [row[:] for row in blk]
            else:
                gi = [row[:] for row in blk] + gi
        else:
            if random.choice((True, False)):
                gi = [gi[r] + blk[r] for r in range(d)]
            else:
                gi = [blk[r] + gi[r] for r in range(d)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ---------------------------------------------------------------------------
# Route variation: deliberately none — returned unchanged.
#
# The transformation is a pure extraction: keep one d x d block of the strip,
# discard the rest. In ARCLE the only op that changes the canvas to a sub-window
# is CropGrid (33), and the window it must take is exactly the target block; any
# other window submits the wrong grid. Submit (34) is mandatory and last. So the
# route contains exactly one operation of substance, acting on exactly one
# region, painting nothing.
#
# That leaves no choice for two solvers to differ over: there is no second
# region to order against the first, no direction to sweep, no base to lay down
# and draw over. The alternatives I considered are all operations the rule does
# not make — trimming the strip in two cuts (the second cut alone already
# reaches the answer), sliding the block to the corner before cropping (the rule
# relocates nothing, and Move would drop the block's 0-coloured cells), or
# mirroring the whole grid across its diagonal and mirroring the block back (a
# turn and the same turn undone). Each would be padding dressed as variety.
#
# What does vary per instance, and is read from the input alone, is the crop
# window itself: the block index is found by testing each block against its own
# transpose, so the selection differs with orientation, block size d, block
# count ng, and which block was distorted. Nothing here is read from O.
#
# On the 0 caveat: CropGrid clears the canvas before the transparent copy, so
# 0-coloured cells inside the kept window land as 0 correctly (verified: 82/200
# sampled outputs contain colour 0, trajectory check 100%). Nothing is pasted or
# moved here, so no explicit Color(0) is warranted — and emitting one would
# repaint cells that already hold that colour.
# ---------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    d = min(hi, wi)

    # blocks lie along the long axis; short axis length = d
    if hi >= wi:
        n = hi // d
        origins = [(k * d, 0) for k in range(n)]
    else:
        n = wi // d
        origins = [(0, k * d) for k in range(n)]

    # measure the rule from I: the target block is the one NOT equal to its
    # main-diagonal mirror (transpose). All other blocks are symmetric.
    r0, c0 = origins[0]
    for (br, bc) in origins:
        block = I[br:br + d, bc:bc + d]
        if not np.array_equal(block, block.T):
            r0, c0 = br, bc
            break

    # crop the working canvas down to that block
    # (bbox == exactly the full d x d rectangle kept, so the bbox form is right)
    ops.append(33)
    sels.append([r0, c0, d - 1, d - 1])
    ops.append(34)
    sels.append([0, 0, d - 1, d - 1])
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
