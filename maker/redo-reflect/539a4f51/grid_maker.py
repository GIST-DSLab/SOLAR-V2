"""
ARC Task: 539a4f51 (RE-ARC) — LLM-generated grid_maker
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

# The generator's `rotf` choice is the one discrete structural variant: it decides
# which corner of the grid the nested-L "corner pattern" converges to.
#   rot 0 -> apex top-left, 1 -> top-right (rot90 CW), 2 -> bottom-right (rot180),
#   rot 3 -> bottom-left (rot270 CCW)
# (the pattern is symmetric about its main diagonal, so these four rotations are
#  exactly the four reflections identity / vmirror / cmirror / hmirror.)
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # background is hardcoded 0 in the generator, so it is not sampled here.
    # The palette is fixed per episode for visual consistency; the rule itself is
    # colour-agnostic (it only propagates whatever colours the corner already has).
    colpool = random.sample(range(1, 10), 9)
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"colpool": colpool, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, colpool=None, rot=None) -> dict:
    if colpool is None:
        colpool = random.sample(range(1, 10), 9)
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    dmax = min(15, max_h // 2, max_w // 2)
    if dmax < 2:
        dmax = 2
    d = unifint(diff_lb, diff_ub, (2, dmax))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = list(colpool[:numc])
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [random.choice(ccols) for _ in range(numocc)]
    while len(set(arr)) == 1:
        arr = [random.choice(ccols) for _ in range(d)]
    m = len(arr)

    # cell (r, c) belongs to the L-shaped shell number max(r, c)
    gi = [[arr[max(r, c)] if max(r, c) < m else 0 for c in range(d)] for r in range(d)]
    go = [[arr[max(r, c) % m] for c in range(2 * d)] for r in range(2 * d)]

    def rotk(g, k):
        n = len(g)
        if k == 0:
            return [list(row) for row in g]
        if k == 1:  # rot90 CW
            return [[g[n - 1 - c][r] for c in range(n)] for r in range(n)]
        if k == 2:  # rot180
            return [[g[n - 1 - r][n - 1 - c] for c in range(n)] for r in range(n)]
        return [[g[c][n - 1 - r] for c in range(n)] for r in range(n)]  # rot270 CCW

    gi = rotk(gi, rot)
    go = rotk(go, rot)
    return {"input": tuple(tuple(row) for row in gi),
            "output": tuple(tuple(row) for row in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    d = I.shape[0]
    n = 2 * d
    ops, sels = [], []

    # --- 1. Locate the apex of the nested-L pattern --------------------------
    # In the canonical (apex top-left) frame the LAST row and LAST column are
    # constant (all background, or all of the outermost shell colour) while the
    # first row / first column never are (the shell colours are not all equal).
    # So the constant edges are the ones FAR from the apex.
    apex_bottom = len(set(I[d - 1].tolist())) != 1
    apex_right = len(set(I[:, d - 1].tolist())) != 1

    # --- 2. Reflect the grid so the apex sits at the top-left ---------------
    # Whole-grid rectangles: the selections really are the entire region,
    # background included, which is exactly what a mirror acts on.
    full_in = [0, 0, d - 1, d - 1]
    N = I
    if apex_right:
        ops.append(26); sels.append(full_in)      # FlipH: apex right -> left
        N = np.fliplr(N)
    if apex_bottom:
        ops.append(27); sels.append(full_in)      # FlipV: apex bottom -> top
        N = np.flipud(N)

    # shell colours, read off the main diagonal of the normalised grid
    m = 0
    while m < d and N[m, m] != 0:
        m += 1
    arr = [int(N[j, j]) for j in range(m)]

    # --- 3. Double the canvas ------------------------------------------------
    full_out = [0, 0, n - 1, n - 1]
    ops.append(33); sels.append(full_out)          # background is 0 -> no fill needed

    # --- 4. Keep growing the nest of L-shells outward, cycling the colours ---
    for j in range(m, n):
        cells = [(j, c) for c in range(j + 1)] + [(r, j) for r in range(j)]
        ops.append(arr[j % m]); sels.append(sel_of(cells))

    # --- 5. Reflect back into the orientation the input came in -------------
    if apex_right:
        ops.append(26); sels.append(full_out)
    if apex_bottom:
        ops.append(27); sels.append(full_out)

    ops.append(34); sels.append(full_out)
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
                        f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 539a4f51"
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
                                f"for task 539a4f51"
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
                    f"Failed to build a complete episode for task 539a4f51 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"539a4f51-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
