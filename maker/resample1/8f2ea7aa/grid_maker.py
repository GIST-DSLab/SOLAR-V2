"""
ARC Task: 8f2ea7aa (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background colour role matters; the pattern's own colours are
    # copied verbatim, so they are free per instance.
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return int(round(random.uniform(lo, hi)))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    remcols = [c for c in range(10) if c != bgc]
    dmax = 5
    while dmax > 2 and dmax * dmax > min(max_h, max_w):
        dmax -= 1
    d = _unifint(diff_lb, diff_ub, (2, dmax))
    d = max(2, min(d, dmax))
    d2 = d * d
    gi = np.full((d2, d2), bgc, dtype=int)
    go = np.full((d2, d2), bgc, dtype=int)
    minig = np.full((d, d), bgc, dtype=int)
    inds = [(i, j) for i in range(d) for j in range(d)]
    mp = d2 // 2
    dev = _unifint(diff_lb, diff_ub, (0, mp))
    devs = random.choice((+1, -1))
    num = max(min(mp + devs * dev, d2), 0)
    locs = set(random.sample(inds, num))

    def shp(s):
        if not s:
            return (0, 0)
        rs = [a for a, _ in s]
        cs = [b for _, b in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(locs) != (d, d):
        locs.add(random.choice(sorted(set(inds) - locs)))
    ncols = _unifint(diff_lb, diff_ub, (1, 9))
    cols = random.sample(remcols, max(1, min(ncols, 9)))
    for (i, j) in locs:
        minig[i, j] = random.choice(cols)
    pr = random.randrange(0, d2, d)
    pc = random.randrange(0, d2, d)
    gi[pr:pr + d, pc:pc + d] = minig
    for (i, j) in locs:
        go[i * d:i * d + d, j * d:j * d + d] = minig
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the generator fills d^4 cells with bgc and at most d^2 with pattern
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    d = int(round(hi ** 0.5))
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    r0 = min(r for r, _ in fg)
    c0 = min(c for _, c in fg)
    # the d x d mini-pattern (its bbox is exactly d x d by construction)
    pat = I[r0:r0 + d, c0:c0 + d]
    locs = [(i, j) for i in range(d) for j in range(d) if pat[i, j] != bgc]
    src_blk = (r0 // d, c0 // d) if (r0 % d == 0 and c0 % d == 0) else None
    # colour-0 pattern cells (only possible when bgc != 0) are not carried by Paste
    zero_cells = [(i, j) for i in range(d) for j in range(d)
                  if pat[i, j] == 0 and bgc != 0]
    # Destination blocks are all bgc; Paste is visible iff some nonzero
    # pattern cell differs from bgc.
    paste_visible = any(pat[i, j] != 0 and pat[i, j] != bgc
                        for i in range(d) for j in range(d))
    ops, sels = [], []
    copied = False
    for (i, j) in locs:  # one copy of the pattern per filled pattern cell
        if src_blk == (i, j):
            continue  # pattern already sits in this block
        tr, tc = i * d, j * d
        if paste_visible:
            if not copied:
                # CopyI of the full d x d pattern block (whole rectangle intended)
                ops.append(28); sels.append([r0, c0, d - 1, d - 1])
                copied = True
            ops.append(30); sels.append([tr, tc, 0, 0])
        if zero_cells:
            ops.append(0)
            sels.append(sel_of([(tr + a, tc + b) for a, b in zero_cells]))
    if src_blk not in [tuple(x) for x in locs]:
        # original pattern position is not one of the target blocks: erase it
        ops.append(bgc); sels.append(sel_of(fg))
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
                        f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8f2ea7aa"
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
                                f"for task 8f2ea7aa"
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
                    f"Failed to build a complete episode for task 8f2ea7aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8f2ea7aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
