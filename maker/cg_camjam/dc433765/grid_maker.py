"""
ARC Task: dc433765 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"diagonal": True}, {"diagonal": False}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, src = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "src": src, "instance_plan": plan}


def _unif(lb, ub, rng):
    a, b = rng
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, src=1, diagonal=None, **kw) -> dict:
    if diagonal is None:
        diagonal = random.choice([True, False])
    h = _unif(diff_lb, diff_ub, (4, max(4, max_h)))
    w = _unif(diff_lb, diff_ub, (4, max(4, max_w)))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    if diagonal:
        starts = [(ii, 0) for ii in range(h - 2)] + [(0, jj) for jj in range(1, w - 2)]
        lines = []
        for (r0, c0) in starts:
            ln = []
            r, c = r0, c0
            while r < h and c < w:
                ln.append((r, c)); r += 1; c += 1
            lines.append(ln)
        lines.sort(key=len)
        ln = lines[_unif(diff_lb, diff_ub, (0, len(lines) - 1))]
        epi = _unif(diff_lb, diff_ub, (2, len(ln) - 1))
        ep = ln[epi]
        sp = ln[random.randint(0, epi - 2)]
        gi[sp] = src; gi[ep] = 4
        go[sp[0] + 1, sp[1] + 1] = src; go[ep] = 4
    else:
        loci = random.randint(0, h - 1)
        objw = _unif(diff_lb, diff_ub, (3, w))
        locj1 = random.randint(0, w - objw)
        locj2 = locj1 + objw - 1
        gi[loci, locj1] = src; gi[loci, locj2] = 4
        go[loci, locj1 + 1] = src; go[loci, locj2] = 4
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, -1),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(), "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the canvas colour (grid holds only two marker pixels)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    palette = set(I.flatten().tolist()) - {bgc}
    src_col = [c for c in palette if c != 4][0]
    r4, c4 = [int(v) for v in np.argwhere(I == 4)[0]]
    rs, cs = [int(v) for v in np.argwhere(I == src_col)[0]]
    dr = (r4 > rs) - (r4 < rs)
    dc = (c4 > cs) - (c4 < cs)

    # Reading: the empty gap between the source and the beacon contracts by one
    # line on each axis -- the whole band of rows (then columns) on the source's
    # side of the beacon slides one line toward the beacon, swallowing the empty
    # line next to it; a fresh background line appears at the far grid edge.

    # ARCLE object ops only carry non-zero cells: make a colour-0 source visible first
    zero_src = (src_col == 0)
    if zero_src:
        tmp = 1 if bgc != 1 else 2
        ops.append(tmp); sels.append(sel_of([(rs, cs)]))

    def band_move(cells, op, edge_cells):
        ops.append(op); sels.append(sel_of(cells))   # one grab of the whole band
        if bgc != 0:                                  # vacated far-edge line reads 0
            ops.append(bgc); sels.append(sel_of(edge_cells))

    if dr > 0:   # rows 0..r4-2 slide down
        band = [(r, c) for r in range(0, r4 - 1) for c in range(w)]
        band_move(band, 21, [(0, c) for c in range(w)])
    elif dr < 0:  # rows r4+2..h-1 slide up
        band = [(r, c) for r in range(r4 + 2, h) for c in range(w)]
        band_move(band, 20, [(h - 1, c) for c in range(w)])

    if dc > 0:   # columns 0..c4-2 slide right
        band = [(r, c) for r in range(h) for c in range(0, c4 - 1)]
        band_move(band, 22, [(r, 0) for r in range(h)])
    elif dc < 0:  # columns c4+2..w-1 slide left
        band = [(r, c) for r in range(h) for c in range(c4 + 2, w)]
        band_move(band, 23, [(r, w - 1) for r in range(h)])

    if zero_src:  # restore the carried pixel to its true colour 0
        ops.append(0); sels.append(sel_of([(rs + dr, cs + dc)]))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task dc433765"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task dc433765"
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
                                f"for task dc433765"
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
                    f"Failed to build a complete episode for task dc433765 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"dc433765-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
