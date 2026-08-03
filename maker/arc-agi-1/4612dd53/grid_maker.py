"""
ARC Task: 4612dd53 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"bar_dir": "h", "bar_visible": True},
    {"bar_dir": "h", "bar_visible": False},
    {"bar_dir": "v", "bar_visible": True},
    {"bar_dir": "v", "bar_visible": False},
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]          # 2 is reserved as the fill color
    bgc, col = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "col": col, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, col, bar_dir=None, bar_visible=None) -> dict:
    if bar_dir is None:
        bar_dir = random.choice(["h", "v"])
    if bar_visible is None:
        bar_visible = random.choice([True, False])

    def U(a, b):
        if b < a:
            b = a
        return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))

    h = U(8, max(8, max_h))
    w = U(8, max(8, max_w))
    ih = U(5, h - 1)
    iw = U(5, w - 1)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    r0, c0 = loci, locj
    r1, c1 = loci + ih - 1, locj + iw - 1

    bx = set()
    for c in range(c0, c1 + 1):
        bx.add((r0, c))
        bx.add((r1, c))
    for r in range(r0, r1 + 1):
        bx.add((r, c0))
        bx.add((r, c1))

    if bar_dir == "h":
        rr = random.randint(r0 + 2, r1 - 2)
        br = {(rr, c) for c in range(c0 + 1, c1)}
    else:
        cc = random.randint(c0 + 2, c1 - 2)
        br = {(r, cc) for r in range(r0 + 1, r1)}

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    crns = set(random.sample(corners, 3))          # >=3 corners always kept -> bbox recoverable
    rembx = list(bx - crns)
    onbr = set(random.sample(sorted(br), 2))       # >=2 bar cells kept -> orientation recoverable
    rembr = list(br - onbr)

    noccbx = U(0, len(rembx))
    noccbr = U(0, len(rembr))
    occbx = set(random.sample(rembx, noccbx))
    occbr = set(random.sample(rembr, noccbr))

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]
    for (r, c) in bx | br:
        gi[r][c] = col
        go[r][c] = col
    for (r, c) in occbx | occbr:
        gi[r][c] = bgc
        go[r][c] = 2
    if not bar_visible:
        for (r, c) in br:
            gi[r][c] = bgc
            go[r][c] = bgc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    present = sorted(set(I.flatten().tolist()))

    def fit(color):
        """Does `color` form a rectangle outline (>=3 corners) plus at most one interior line?"""
        cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] == color}
        if not cells:
            return None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if sum(1 for p in corners if p in cells) < 3:
            return None
        inner = [(r, c) for (r, c) in cells if r not in (r0, r1) and c not in (c0, c1)]
        line = set()
        if inner:
            rows = {r for r, _ in inner}
            cols = {c for _, c in inner}
            if len(rows) == 1:                       # remnants share a row -> horizontal bar
                rr = rows.pop()
                line = {(rr, c) for c in range(c0, c1 + 1)}
            elif len(cols) == 1:                     # remnants share a column -> vertical bar
                cc = cols.pop()
                line = {(r, cc) for r in range(r0, r1 + 1)}
            else:
                return None
        return (r0, c0, r1, c1, line)

    fits = [(c, fit(c)) for c in present]
    fits = [(c, f) for (c, f) in fits if f is not None]
    if not fits:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    if len(fits) > 1:                                # tie-break: the outline color is the sparse one
        fits.sort(key=lambda t: int((I == t[0]).sum()))
    col, (r0, c0, r1, c1, line) = fits[0]
    bgc = next((c for c in present if c != col), col)

    # the shape the input is missing: full box outline + the bar's line
    box = set()
    for c in range(c0, c1 + 1):
        box.add((r0, c))
        box.add((r1, c))
    for r in range(r0, r1 + 1):
        box.add((r, c0))
        box.add((r, c1))
    targets = {p for p in (box | line) if I[p[0], p[1]] == bgc}

    if not targets:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    tr, tc = next(iter(targets))
    fill_col = int(O[tr, tc])

    # walk the box edge by edge (disjoint), then the bar; emit one op per contiguous gap
    regions = [
        [(r0, c) for c in range(c0, c1 + 1)],            # top edge
        [(r, c1) for r in range(r0 + 1, r1)],            # right edge
        [(r1, c) for c in range(c1, c0 - 1, -1)],        # bottom edge
        [(r, c0) for r in range(r1 - 1, r0, -1)],        # left edge
    ]
    bar = sorted(p for p in line if p[0] not in (r0, r1) and p[1] not in (c0, c1))
    if bar:
        regions.append(bar)

    for region in regions:
        run = []
        for p in region + [None]:
            if p is not None and p in targets:
                run.append(p)
                continue
            if run:
                rr = [q[0] for q in run]
                cc = [q[1] for q in run]
                ops.append(fill_col)
                sels.append([min(rr), min(cc), max(rr) - min(rr), max(cc) - min(cc)])
                run = []

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
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
                                f"for task 4612dd53"
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
                    f"Failed to build a complete episode for task 4612dd53 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4612dd53-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
