"""
ARC Task: 94f9d214 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(lo_i, hi_i)


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    acol = random.choice(rem)
    rem = [c for c in rem if c != acol]
    bcol = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice((True, False))
    if mirrored:
        hmax, wmax = max_w, max_h // 2
    else:
        hmax, wmax = max_h, max_w // 2
    hmax = max(2, min(30, hmax))
    wmax = max(2, min(14, wmax))
    h = _unifint(diff_lb, diff_ub, (2, hmax))
    w = _unifint(diff_lb, diff_ub, (2, wmax))
    inds = [(i, j) for i in range(h) for j in range(w)]
    numadev = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = random.choice((numadev, h * w - numadev))
    numb = random.choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = random.sample(inds, numa)
    bset = random.sample(inds, numb)
    A = np.full((h, w), bgc, dtype=int)
    B = np.full((h, w), bgc, dtype=int)
    for i, j in aset:
        A[i, j] = acol
    for i, j in bset:
        B[i, j] = bcol
    gi = np.concatenate([A, B], axis=1)
    go = np.full((h, w), bgc, dtype=int)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    for i, j in res:
        go[i, j] = 2
    if mirrored:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- split I into its two halves, using only I: each half is bg + one colour ---
    cands = []
    if wi % 2 == 0:
        cands.append(("h", wi // 2))
    if hi % 2 == 0:
        cands.append(("v", hi // 2))

    def halves(kind, k):
        if kind == "h":
            return I[:, :k], I[:, k:], (0, k)
        return I[:k, :], I[k:, :], (k, 0)

    chosen = None
    for kind, k in cands:
        a, b, off = halves(kind, k)
        pa, pb = set(a.flatten().tolist()), set(b.flatten().tolist())
        sh = pa & pb
        if len(pa) == 2 and len(pb) == 2 and len(sh) == 1:
            chosen = (a, b, off, sh.pop())
            break
    if chosen is None:
        kind, k = cands[0]
        a, b, off = halves(kind, k)
        sh = set(a.flatten().tolist()) & set(b.flatten().tolist())
        bgc = sorted(sh)[0] if sh else int(np.bincount(a.flatten()).argmax())
        chosen = (a, b, off, bgc)
    A, B, (dr, dc), bgc = chosen
    ha, wa = A.shape

    ops, sels = [], []
    grid = I.copy()

    # 1. Base layer: the A half is presumed "empty in both" -> paint it 2 (whole rectangle).
    ops.append(2); sels.append([0, 0, ha - 1, wa - 1])   # full rectangle of half A
    grid[:ha, :wa] = 2

    # 2. Knock out A's own pattern: every cell of A's colour (as in I) back to background.
    a_cells = [(r, c) for r in range(ha) for c in range(wa) if A[r, c] != bgc]
    ops.append(int(bgc)); sels.append(sel_of(a_cells))
    for r, c in a_cells:
        grid[r, c] = bgc

    # 3. Knock out B's pattern laid over A: B's coloured cells, landed on half A.
    b_cells = [(r, c) for r in range(ha) for c in range(wa) if B[r, c] != bgc]
    if any(grid[r, c] != bgc for r, c in b_cells):
        ops.append(int(bgc)); sels.append(sel_of(b_cells))
        for r, c in b_cells:
            grid[r, c] = bgc

    # 4. Half B has been consumed — crop the canvas down to half A.
    ops.append(33); sels.append([0, 0, ha - 1, wa - 1])

    ops.append(34); sels.append([0, 0, ha - 1, wa - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 94f9d214"
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
                                f"for task 94f9d214"
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
                    f"Failed to build a complete episode for task 94f9d214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"94f9d214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
