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

# The only discrete structural variant this task has is the global rotation the
# generator applies to both grids (identity / rot90 / rot180 / rot270).
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # background is hardcoded 0 in the generator -> not a sampled role.
    # The palette the nested-L colors are drawn from is sampled -> fix it per episode.
    numc = random.randint(2, 9)
    ccols = random.sample(list(range(1, 10)), numc)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown
    return {"ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, rot=None) -> dict:
    if ccols is None:
        ccols = random.sample(list(range(1, 10)), random.randint(2, 9))
    if rot is None:
        rot = random.choice(VARIANTS)["rot"]

    dub = max(2, min(15, max_h // 2, max_w // 2))
    try:
        d = unifint(diff_lb, diff_ub, (2, dub))
        numocc = unifint(diff_lb, diff_ub, (1, d))
    except NameError:
        d = random.randint(2, dub)
        numocc = random.randint(1, d)

    arr = [random.choice(ccols) for _ in range(numocc)]
    while len(set(arr)) == 1:
        arr = [random.choice(ccols) for _ in range(d)]
    n = len(arr)

    # cell (r, c) belongs to the nested L of index max(r, c)
    gi = [[arr[max(r, c)] if max(r, c) < n else 0 for c in range(d)] for r in range(d)]
    go = [[arr[max(r, c) % n] for c in range(2 * d)] for r in range(2 * d)]

    for _ in range(rot):                      # rot CW quarter turns on both grids
        gi = [list(x) for x in zip(*gi[::-1])]
        go = [list(x) for x in zip(*go[::-1])]

    gi = tuple(tuple(int(v) for v in row) for row in gi)
    go = tuple(tuple(int(v) for v in row) for row in go)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    d = I.shape[0]
    N = O.shape[0]                      # N == 2 * d

    def canon_n(G):
        """n if G is the canonical orientation (nested Ls anchored at top-left), else None."""
        nz = np.argwhere(G != 0)
        if nz.size == 0:
            return None
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        if r0 != 0 or c0 != 0 or r1 != c1:
            return None
        n = int(r1) + 1
        for r in range(G.shape[0]):
            for c in range(G.shape[1]):
                m = max(r, c)
                want = G[m, m] if m < n else 0
                if G[r, c] != want:
                    return None
        return n

    k, n = None, None
    for kk in range(4):
        nn = canon_n(np.rot90(I, kk))
        if nn is not None:
            k, n = kk, nn
            break
    if k is None:                        # defensive fallback
        nz = np.argwhere(I != 0)
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        k, n = 0, int(max(r1 - r0, c1 - c0)) + 1

    ops, sels = [], []
    full_i = [0, 0, d - 1, d - 1]        # bbox == the whole input canvas (rotating everything)
    full_o = [0, 0, N - 1, N - 1]        # bbox == the whole output canvas

    # 1. Turn the pattern into its canonical orientation (Ls anchored at top-left).
    if k == 1:
        ops.append(24); sels.append(full_i)                    # CCW
    elif k == 2:
        ops.append(24); sels.append(full_i)
        ops.append(24); sels.append(full_i)
    elif k == 3:
        ops.append(25); sels.append(full_i)                    # CW

    # 2. Double the canvas.
    ops.append(33); sels.append(full_o)

    # 3. Replicate the colour sequence along the top row (period n).
    ops.append(29); sels.append([0, 0, 0, n - 1])              # CopyO the n-colour sequence
    c = n
    while c < N:
        ops.append(30); sels.append([0, c, 0, 0])
        c += n

    # 4. Replicate that periodic row down the whole canvas.
    ops.append(29); sels.append([0, 0, 0, N - 1])              # CopyO the full periodic row
    for r in range(1, N):
        ops.append(30); sels.append([r, 0, 0, 0])

    # 5. Keep only the upper triangle (c >= r); clear the strictly lower one to background 0
    #    so the mirrored copy can be pasted into it.
    lower = [(r, cc) for r in range(N) for cc in range(r)]
    ops.append(0); sels.append(sel_of(lower))

    # 6. The pattern is symmetric about the main diagonal: copy the upper triangle,
    #    mirror the canvas diagonally (CCW rotate + vertical flip == transpose),
    #    then paste the upper triangle back on top of the mirrored half.
    ops.append(29); sels.append(full_o)                        # clipboard = upper triangle
    ops.append(24); sels.append(full_o)
    ops.append(27); sels.append(full_o)
    ops.append(30); sels.append([0, 0, 0, 0])

    # 7. Restore the original orientation.
    if k == 1:
        ops.append(25); sels.append(full_o)
    elif k == 2:
        ops.append(25); sels.append(full_o)
        ops.append(25); sels.append(full_o)
    elif k == 3:
        ops.append(24); sels.append(full_o)

    ops.append(34); sels.append(full_o)
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
