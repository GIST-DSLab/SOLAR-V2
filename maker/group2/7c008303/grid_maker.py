"""
ARC Task: 7c008303 (RE-ARC) — LLM-generated grid_maker
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
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    fgc = random.choice(rem)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc, **kw) -> dict:
    from random import choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    m = min(max_h, max_w)
    hmax = max(2, min(13, (m - 3) // 2))
    h = unifint(diff_lb, diff_ub, (2, hmax)) * 2
    w = unifint(diff_lb, diff_ub, (2, hmax)) * 2
    remcols = [c for c in range(10) if c not in (bgc, linc, fgc)]
    nf = unifint(diff_lb, diff_ub, (1, 4))
    nf = max(1, min(nf, len(remcols)))
    fremcols = sample(remcols, nf)
    qc = [choice(fremcols) for _ in range(4)]

    go = np.full((h, w), bgc, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    for (i, j) in sample(inds, nc):
        go[i, j] = fgc

    gi = np.full((h + 3, w + 3), bgc, dtype=int)
    gi[3:, 3:] = go
    gi[2, :] = linc
    gi[:, 2] = linc
    gi[0, 0] = qc[0]; gi[0, 1] = qc[1]; gi[1, 0] = qc[2]; gi[1, 1] = qc[3]

    out = go.copy()
    hh, hw = h // 2, w // 2
    for k, (r0, c0) in enumerate([(0, 0), (0, hw), (hh, 0), (hh, hw)]):
        blk = out[r0:r0 + hh, c0:c0 + hw]
        blk[blk == fgc] = qc[k]

    k = choice((0, 1, 2, 3))  # identity, rot90(CW), rot180, rot270
    gi = np.rot90(gi, -k)
    out = np.rot90(out, -k)
    return {"input": gi.tolist(), "output": out.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Separator lines: the single uniform row and uniform column.
    R = next(r for r in range(hi) if len(set(I[r, :].tolist())) == 1)
    C = next(c for c in range(wi) if len(set(I[:, c].tolist())) == 1)
    r_ranges = [(0, R), (R + 1, hi)]
    c_ranges = [(0, C), (C + 1, wi)]

    best = None
    for i in range(2):
        for j in range(2):
            area = (r_ranges[i][1] - r_ranges[i][0]) * (c_ranges[j][1] - c_ranges[j][0])
            if best is None or area > best[0]:
                best = (area, i, j)
    _, pi, pj = best
    pr0, pr1 = r_ranges[pi]
    pc0, pc1 = c_ranges[pj]
    kr0 = r_ranges[1 - pi][0]
    kc0 = c_ranges[1 - pj][0]
    ph, pw = pr1 - pr0, pc1 - pc0
    bgc = int(I[r_ranges[1 - pi][0], c_ranges[pj][0]])  # fill of an empty quadrant

    key = [[int(I[kr0 + a, kc0 + b]) for b in range(2)] for a in range(2)]
    hh, hw = ph // 2, pw // 2
    P = I[pr0:pr1, pc0:pc1]

    # 1. Crop the canvas to the pattern block.
    ops.append(33); sels.append([pr0, pc0, ph - 1, pw - 1])

    # 2. Quadrant by quadrant: lay the whole quadrant rectangle in its key colour
    #    (a full rectangle, so a bbox selection is exact). Skip quadrants holding
    #    no pattern cells (nothing there changes).
    painted = []
    for a in range(2):
        for b in range(2):
            r0, c0 = a * hh, b * hw
            quad = P[r0:r0 + hh, c0:c0 + hw]
            if not np.any(quad != bgc):
                continue
            ops.append(key[a][b]); sels.append([r0, c0, hh - 1, hw - 1])
            painted.append((r0, c0))

    # 3. Restore the background: every input-background cell of the pattern
    #    block inside the painted quadrants, in one stroke.
    holes = []
    for (r0, c0) in painted:
        for r in range(r0, r0 + hh):
            for c in range(c0, c0 + hw):
                if int(P[r, c]) == bgc:
                    holes.append((r, c))
    if holes:
        ops.append(bgc); sels.append(sel_of(holes))

    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7c008303"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7c008303"
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
                                f"for task 7c008303"
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
                    f"Failed to build a complete episode for task 7c008303 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7c008303-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
