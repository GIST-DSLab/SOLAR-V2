"""
ARC Task: 08ed6ac7 (RE-ARC) — LLM-generated grid_maker
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
    # The rule only depends on bar lengths, so bar colours stay random. Only the background is fixed.
    bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 4)])
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi = a + round(ub * (b - a))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    colopts = list(range(10))
    if bgc is None:
        bgc = random.choice([c for c in colopts if c not in (1, 2, 3, 4)])
    h = unifint(diff_lb, diff_ub, (4, max(4, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (4, max(4, min(30, max_w))))
    remcols = [c for c in colopts if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    nbars = unifint(diff_lb, diff_ub, (4, w))
    barlocs = random.sample(range(w), nbars)
    barhs = random.sample(range(h), 4)
    barcols = [random.choice(remcols) for _ in range(nbars)]
    barhsfx = [random.choice(barhs) for _ in range(nbars - 4)] + list(barhs)
    random.shuffle(barhsfx)
    ordered = sorted(barhs)
    for col, loci, locj in zip(barcols, barhsfx, barlocs):
        for r in range(loci, h):
            gi[r][locj] = col
            go[r][locj] = ordered.index(loci) + 1
    return {"input": gi, "output": go}


def _bars(I, bg):
    """Return list of (col, length) assuming background bg, or None if inconsistent."""
    h, w = I.shape
    bars = []
    for c in range(w):
        col = I[:, c]
        nz = np.nonzero(col != bg)[0]
        if len(nz) == 0:
            continue
        top = int(nz[0])
        if len(nz) != h - top:          # must be a contiguous run down to the last row
            return None
        bars.append((c, h - top))
    if len({L for _, L in bars}) != 4:
        return None
    return bars


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background candidates: colours in row 0 that the generator allows as bgc
    order = [v for v, _ in Counter(I.flatten().tolist()).most_common() if v not in (1, 2, 3, 4)]
    if examples:
        ex_cnt = Counter()
        for ei, _ in examples:
            ei = np.asarray(ei, dtype=int)
            ex_cnt.update(ei[0].tolist())
        order.sort(key=lambda v: -ex_cnt.get(v, 0))
    bars = None
    for v in order:
        bars = _bars(I, v)
        if bars is not None:
            break
    if bars is None:
        bars = []

    lengths = sorted({L for _, L in bars}, reverse=True)   # longest -> colour 1
    rank = {L: i + 1 for i, L in enumerate(lengths)}

    ops, sels = [], []
    # one Color op per rank colour, covering every bar cell of that length class
    for L in lengths:
        k = rank[L]
        cells = [(r, c) for (c, LL) in bars if LL == L
                 for r in range(h - L, h) if int(I[r, c]) != k]
        if cells:
            ops.append(k)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # full grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
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
                                f"for task 08ed6ac7"
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
                    f"Failed to build a complete episode for task 08ed6ac7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"08ed6ac7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
