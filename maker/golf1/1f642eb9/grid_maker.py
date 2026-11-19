"""
ARC Task: 1f642eb9 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background has to stay the same across the episode. The block colour
    # and marker colours change per instance, and the rule does not depend on them
    # (a marker stamps its own colour).
    return {"bgc": random.randrange(10)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        span = b - a
        lo = a + int(round(lb * span))
        hi = a + int(round(ub * span))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    sqc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, sqc)]
    numcells = unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs, ins = [], []
    c1, c2, c3, c4 = [random.choice((True, False)) for _ in range(4)]
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0)); ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1)); ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - c3):
        outs.append((0, b)); ins.append((loci, b))
    for b in range(locj + c2, locj + iw - c4):
        outs.append((h - 1, b)); ins.append((loci + ih - 1, b))
    locs = set(random.sample(range(2 * ih + 2 * iw - 4), numcells))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
    seq = [random.choice(ccols) for _ in range(numcells)]
    for col, (r, c) in zip(seq, outs):
        gi[r][c] = col
    go = [row[:] for row in gi]
    for col, (r, c) in zip(seq, ins):
        go[r][c] = col
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """
    Rule (read from I only): the grid's interior holds one solid rectangle (the block).
    It is the only non-background thing that does not touch the grid border. Every
    marker sits on the border. Each marker projects straight along its row or column
    onto the block cell facing it and stamps its own colour there.
    Larger strokes: ONE Color op per marker colour. Its selection is "the block cells
    that the markers of this colour project onto", which can be named from I alone.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # The grid corners are never markers: markers only face a block that sits at
    # least 2 cells in from the edge.
    bgc = int(I[0, 0])
    inner = I[1:h - 1, 1:w - 1]
    rs, cs = np.nonzero(inner != bgc)
    if len(rs) == 0:
        return [34], [[0, 0, h - 1, w - 1]]
    r0, r1 = int(rs.min()) + 1, int(rs.max()) + 1
    c0, c1 = int(cs.min()) + 1, int(cs.max()) + 1

    groups, order = {}, []

    def add(col, cell):
        if col not in groups:
            groups[col] = []
            order.append(col)
        groups[col].append(cell)

    for r in range(r0, r1 + 1):
        if I[r, 0] != bgc:
            add(int(I[r, 0]), (r, c0))           # left marker -> left edge of block
        if I[r, w - 1] != bgc:
            add(int(I[r, w - 1]), (r, c1))       # right marker -> right edge
    for c in range(c0, c1 + 1):
        if I[0, c] != bgc:
            add(int(I[0, c]), (r0, c))           # top marker -> top edge
        if I[h - 1, c] != bgc:
            add(int(I[h - 1, c]), (r1, c))       # bottom marker -> bottom edge

    ops, sels = [], []
    for col in order:
        cells = sorted(p for p in groups[col] if I[p] != col)
        if cells:
            ops.append(col)
            sels.append(sel_of(cells))
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
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
                                f"for task 1f642eb9"
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
                    f"Failed to build a complete episode for task 1f642eb9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f642eb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
