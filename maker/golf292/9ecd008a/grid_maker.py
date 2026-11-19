"""
ARC Task: 9ecd008a (RE-ARC) — LLM-generated grid_maker
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
    # The rule (recover the punched-out window from the grid's mirror symmetry)
    # does not depend on colours; only the background is fixed per episode.
    bgc = random.choice(list(range(1, 10)))
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, int(round(a + (b - a) * d))), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(1, 10)))
    cols = list(range(1, 10))
    hmax = max(3, min(15, max_h // 2, max_w // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    g = [[bgc] * w for _ in range(h)]
    nc = _unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx = bx - {ch}
    for (i, j), c in obj.items():
        g[i][j] = c
    g = np.array(g, dtype=int)
    # symmetrise about the main diagonal
    for i in range(h):
        for j in range(i, w):
            g[i, j] = g[j, i]
    g = np.hstack([g, np.fliplr(g)])
    g = np.vstack([g, np.flipud(g)])
    locidev = _unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = _unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = _unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = _unifint(diff_lb, diff_ub, (locj, w - 1))
    go = g[loci:loci2 + 1, locj:locj2 + 1].copy()
    gi = g.copy()
    gi[loci:loci2 + 1, locj:locj2 + 1] = 0
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # the hole: the rectangular block of 0s punched into a doubly-symmetric grid
    zr, zc = np.where(I == 0)
    r0, c0 = int(zr.min()), int(zc.min())
    hh = int(zr.max()) - r0 + 1
    ww = int(zc.max()) - c0 + 1

    # Mirror the whole grid so the hole's coordinates get the missing content.
    # The generator keeps the hole inside one half, so at least one mirror
    # puts no 0 in the window. Pick the mirror with the fewest 0s there.
    reg_v = np.fliplr(I)[r0:r0 + hh, c0:c0 + ww]
    reg_h = np.flipud(I)[r0:r0 + hh, c0:c0 + ww]
    if int((reg_v == 0).sum()) <= int((reg_h == 0).sum()):
        flip_op, region = 26, reg_v   # FlipH = fliplr
    else:
        flip_op, region = 27, reg_h   # FlipV = flipud

    ops, sels = [], []
    # whole-grid bbox: the intended cells ARE the full rectangle
    ops.append(flip_op); sels.append([0, 0, hi - 1, wi - 1])
    # crop to the hole's window (a rectangle read from I), which now holds the recovered content
    ops.append(33); sels.append([r0, c0, hh - 1, ww - 1])

    # Safety only: the generator never leaves 0s here (the hole never crosses its mirror axis)
    zeros = [(i, j) for i in range(hh) for j in range(ww) if region[i, j] == 0]
    if zeros:
        ops.append(0); sels.append(sel_of(zeros))

    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
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
                                f"for task 9ecd008a"
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
                    f"Failed to build a complete episode for task 9ecd008a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9ecd008a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
