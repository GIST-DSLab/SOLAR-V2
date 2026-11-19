"""
ARC Task: 8e5a5113 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barc = random.choice(rem)
    patpool = [c for c in rem if c != barc]
    return {"bgc": bgc, "barc": barc, "patpool": patpool}


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, barc=None, patpool=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if barc is None:
        barc = random.choice([c for c in cols if c != bgc])
    if patpool is None:
        patpool = [c for c in cols if c not in (bgc, barc)]
    vertical = random.choice((True, False))
    long_side = max_h if vertical else max_w
    short_side = max_w if vertical else max_h
    d = _unif(diff_lb, diff_ub, (2, 9))
    d = max(2, min(d, short_side, (long_side + 1) // 2 - 1))
    k = 4 if d < 7 else 3
    num = _unif(diff_lb, diff_ub, (2, k))
    while num > 2 and num * (d + 1) - 1 > long_side:
        num -= 1
    ncols = _unif(diff_lb, diff_ub, (1, 8))
    patcols = random.sample(patpool, min(ncols, len(patpool)))
    c = [[bgc] * d for _ in range(d)]
    inds = [(i, j) for i in range(d) for j in range(d)]
    ncells = _unif(diff_lb, diff_ub, (1, d * d - 1))
    for (i, j) in random.sample(inds, ncells):
        c[i][j] = random.choice(patcols)
    bg = [[bgc] * d for _ in range(d)]
    fill = random.randint(0, num - 1)
    gi = [list(r) for r in (c if fill == 0 else bg)]
    go = [list(r) for r in c]
    for j in range(num - 1):
        c = _rot90(c)
        src = c if j + 1 == fill else bg
        gi = [gi[r] + [barc] + list(src[r]) for r in range(d)]
        go = [go[r] + [barc] + list(c[r]) for r in range(d)]
    if vertical:
        gi = _rot90(gi)
        go = _rot90(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # panel geometry: d x d panels separated by 1-cell bars
    vertical = hi > wi
    d = wi if vertical else hi
    num = (max(hi, wi) + 1) // (d + 1)
    origins = [(j * (d + 1), 0) if vertical else (0, j * (d + 1)) for j in range(num)]
    barc = int(I[d, 0]) if vertical else int(I[0, d])

    src, bgc = 0, 0
    for j, (r, c) in enumerate(origins):
        p = I[r:r + d, c:c + d]
        if len(set(p.flatten().tolist())) > 1:
            src = j
        else:
            bgc = int(p[0, 0])

    sr, sc = origins[src]
    has_zero = bool((I[sr:sr + d, sc:sc + d] == 0).any())
    need_clear = has_zero and bgc != 0

    ops, sels = [], []
    # copy the pattern panel (full d x d rectangle) from the input
    ops.append(28); sels.append([sr, sc, d - 1, d - 1])

    # contiguous runs of empty target panels
    runs, cur = [], []
    for t in range(num):
        if t == src:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(cur)

    for run in runs:
        cleared = False
        if need_clear and (barc == 0 or len(run) >= 3):
            # blank the whole run of panels in one stroke (full rectangle incl. bars)
            r0, c0 = origins[run[0]]
            r1, c1 = origins[run[-1]]
            ops.append(0); sels.append([r0, c0, r1 + d - 1 - r0, c1 + d - 1 - c0])
            if barc != 0:
                bar_cells = []
                for t in run[:-1]:
                    tr, tc = origins[t]
                    if vertical:
                        bar_cells += [(tr + d, x) for x in range(wi)]
                    else:
                        bar_cells += [(y, tc + d) for y in range(hi)]
                ops.append(barc); sels.append(sel_of(bar_cells))
            cleared = True
        for t in run:
            tr, tc = origins[t]
            if need_clear and not cleared:
                ops.append(0); sels.append([tr, tc, d - 1, d - 1])  # full panel
            ops.append(30); sels.append([tr, tc, 0, 0])
            kk = (t - src) % 4
            if kk == 3:
                ops.append(24); sels.append([tr, tc, d - 1, d - 1])  # one CCW
            else:
                for _ in range(kk):
                    ops.append(25); sels.append([tr, tc, d - 1, d - 1])  # CW

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e5a5113"
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
                                f"for task 8e5a5113"
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
                    f"Failed to build a complete episode for task 8e5a5113 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e5a5113-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
