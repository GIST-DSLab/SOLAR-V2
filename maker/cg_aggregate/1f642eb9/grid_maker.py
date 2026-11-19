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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return int(min(max(a, round(a + (b - a) * d)), b))


def sample_colors(num_examples=None) -> dict:
    # The rule never depends on which colour a marker has (each marker keeps its own colour).
    # Only the background colour and the block colour are fixed for the episode.
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(cols, 2)
    max_h = max(6, min(30, max_h))
    max_w = max(6, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (6, max_h))
    w = _unifint(diff_lb, diff_ub, (6, max_w))
    ih = _unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = _unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    remcols = [c for c in cols if c not in (bgc, sqc)]
    numcells = _unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs, ins = [], []
    c1 = random.choice((True, False))
    c2 = random.choice((True, False))
    c3 = random.choice((True, False))
    c4 = random.choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0)); ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1)); ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - c3):
        outs.append((0, b)); ins.append((loci, b))
    for b in range(locj + c2, locj + iw - c4):
        outs.append((h - 1, b)); ins.append((loci + ih - 1, b))
    inds = list(range(2 * ih + 2 * iw - 4))
    locs = set(random.sample(inds, min(numcells, len(inds))))
    numc = _unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    gi = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
    seq = [random.choice(ccols) for _ in range(len(outs))]
    for col, (r, c) in zip(seq, outs):
        gi[r][c] = col
    go = [row[:] for row in gi]
    for col, (r, c) in zip(seq, ins):
        go[r][c] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule, read from I alone:
      * the canvas colour (the generator keeps the grid corners as background) is bgc;
      * the solid block is every non-background cell off the border;
      * every border cell that isn't background is a marker. It projects along its row or
        column onto the block cell facing it, which takes the marker's colour. The marker
        itself stays where it is.
    The golf entry reads the task the same way the maker does. It sweeps one side at a time
    (rotating 4 times), and on each line it carries the marker inward until the line meets
    the block. So the unit is one marker on its line, grouped by side, and that is what we
    emit. Copy/Paste cannot be used here: markers may be colour 0, which Copy/Paste treats
    as nothing. So each landing is one Color op on exactly one cell.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(I[0, 0])  # corners are never markers (block starts at row/col >= 2)

    # block = non-background cells strictly inside the border
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if I[r, c] != bgc]
    ops, sels = [], []
    if not inner:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    r0 = min(r for r, _ in inner); r1 = max(r for r, _ in inner)
    c0 = min(c for _, c in inner); c1 = max(c for _, c in inner)

    left = [(r, int(I[r, 0]), (r, c0)) for r in range(r0, r1 + 1) if I[r, 0] != bgc]
    right = [(r, int(I[r, w - 1]), (r, c1)) for r in range(r0, r1 + 1) if I[r, w - 1] != bgc]
    top = [(c, int(I[0, c]), (r0, c)) for c in range(c0, c1 + 1) if I[0, c] != bgc]
    bottom = [(c, int(I[h - 1, c]), (r1, c)) for c in range(c0, c1 + 1) if I[h - 1, c] != bgc]

    # sweep one side at a time; each marker lands on the block edge facing it
    for side in (left, right, top, bottom):
        for _, col, cell in side:
            ops.append(col)
            sels.append(sel_of([cell]))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
