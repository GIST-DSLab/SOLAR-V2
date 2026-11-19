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

ROTS = [0, 1, 2, 3]  # number of clockwise quarter turns applied to the whole instance


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * max(0.0, min(1.0, diff_lb))
    hi = a + (b - a) * max(0.0, min(1.0, diff_ub))
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    fgc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # the input is (h+3) x (w+3) before rotation; odd rotations swap the axes
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    hub = max(2, min(13, (lim_h - 3) // 2))
    wub = max(2, min(13, (lim_w - 3) // 2))
    h = _unifint(diff_lb, diff_ub, (2, hub)) * 2
    w = _unifint(diff_lb, diff_ub, (2, wub)) * 2
    remcols = [c for c in range(10) if c not in (bgc, linc, fgc)]
    k = _unifint(diff_lb, diff_ub, (1, 4))
    fremcols = random.sample(remcols, k)
    qc = [random.choice(fremcols) for _ in range(4)]

    go = np.full((h, w), bgc, dtype=int)
    inds = [(r, c) for r in range(h) for c in range(w)]
    ncd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = random.choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    for (r, c) in random.sample(inds, nc):
        go[r, c] = fgc

    gi = np.full((h + 3, w + 3), bgc, dtype=int)
    gi[3:, 3:] = go
    gi[2, :] = linc
    gi[:, 2] = linc
    gi[0, 0], gi[0, 1], gi[1, 0], gi[1, 1] = qc

    out = go.copy()
    hh, hw = h // 2, w // 2
    for (a, b), q in zip([(0, 0), (0, 1), (1, 0), (1, 1)], qc):
        blk = out[a * hh:(a + 1) * hh, b * hw:(b + 1) * hw]
        blk[blk == fgc] = q

    gi = np.rot90(gi, -rot)
    out = np.rot90(out, -rot)
    return {"input": gi.tolist(), "output": out.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # The two separator lines: the only uniform row and the only uniform column.
    R = next(r for r in range(hi) if len(set(I[r, :].tolist())) == 1)
    C = next(c for c in range(wi) if len(set(I[:, c].tolist())) == 1)
    r_ranges = [(0, R), (R + 1, hi)]
    c_ranges = [(0, C), (C + 1, wi)]

    # The four parts cut out by the lines: the biggest one is the pattern,
    # the opposite corner is the 2x2 colour guide.
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
    bgc = int(I[r_ranges[1 - pi][0], c_ranges[pj][0]])  # fill colour of an empty part
    guide = [[int(I[kr0 + a, kc0 + b]) for b in range(2)] for a in range(2)]

    # 1. Crop the canvas to the pattern (a full rectangle).
    ops.append(33); sels.append([pr0, pc0, ph - 1, pw - 1])

    # 2. Split the pattern into its four sub-blocks. In each one, recolour the
    #    pattern-colour cells with the guide colour in the same relative position.
    hh, hw = ph // 2, pw // 2
    for a in range(2):
        for b in range(2):
            tgt = guide[a][b]
            cells = [(r, c)
                     for r in range(a * hh, (a + 1) * hh)
                     for c in range(b * hw, (b + 1) * hw)
                     if int(I[pr0 + r, pc0 + c]) != bgc]
            if cells:
                ops.append(tgt); sels.append(sel_of(cells))

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
