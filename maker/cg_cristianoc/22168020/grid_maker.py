"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; each figure takes a fresh
    # distinct colour and the rule ("fill between same-colour cells") never
    # depends on which colour that is.
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _mirror(cells, mode):
    if mode == "identity":
        return set(cells)
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    a, b = min(rs), min(cs)
    dr, dc = min(rs) + max(rs), min(cs) + max(cs)
    if mode == "dmirror":
        return {(c - b + a, r - a + b) for r, c in cells}
    if mode == "hmirror":
        return {(dr - r, c) for r, c in cells}
    if mode == "vmirror":
        return {(r, dc - c) for r, c in cells}
    if mode == "cmirror":
        return _mirror(_mirror(_mirror(cells, "vmirror"), "dmirror"), "vmirror")
    return set(cells)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = _unifint(diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    remcols = [c for c in range(10) if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = gi.copy()
    inds = {(r, c) for r in range(h) for c in range(w)}
    succ = tr = 0
    while tr < 6 * num and succ < num:
        tr += 1
        d = _unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if not inds:
            continue
        li, lj = random.choice(sorted(inds))
        io = {(li + k, lj + k) for k in range(d)} | \
             {(li + k, lj + ow - 1 - k) for k in range(d)}
        io |= {(li + d, lj + d - 1), (li + d, lj + d)}
        oo = set()
        for r in {r for r, c in io}:
            cs = [c for rr, c in io if rr == r]
            oo |= {(r, c) for c in range(min(cs), max(cs) + 1)}
        mf = random.choice(["identity", "dmirror", "cmirror", "hmirror", "vmirror"])
        io = _mirror(io, mf)
        oo = _mirror(oo, mf)
        col = random.choice(remcols)
        if oo.issubset(inds):
            for r, c in io:
                gi[r, c] = col
            for r, c in oo:
                go[r, c] = col
            succ += 1
            inds -= oo
            remcols = [x for x in remcols if x != col]
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared by the golf entry; the maker's all-pairs connect agrees on
    # the segments but groups them by cell-pair rather than by scanline):
    # each figure is one colour -- two diagonal wings closing onto a 2x2 block.
    # Along every scanline crossing the figure's opening, the figure has two
    # cells; the rule fills the span between them with that colour.
    # The generator mirrors figures, so the opening runs across rows OR across
    # columns; both axes are scanned, and a line whose span is already solid
    # (block rows, cross-axis lines) is skipped since filling it changes nothing.
    # One op = one scanline span of one figure, swept from one end to the other.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Background: the canvas colour the generator paints first (figures are sparse).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})
    for col in colors:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        for axis in (0, 1):           # 0: row scanlines, 1: column scanlines
            lines = {}
            for r, c in cells:
                key, pos = (r, c) if axis == 0 else (c, r)
                lines.setdefault(key, []).append(pos)
            for key in sorted(lines):
                ps = lines[key]
                lo, hi_ = min(ps), max(ps)
                span = [(key, p) if axis == 0 else (p, key)
                        for p in range(lo, hi_ + 1)]
                if all(I[r, c] == col for r, c in span):
                    continue          # nothing between this line's cells: no-op
                ops.append(col)
                sels.append(sel_of(span))
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
