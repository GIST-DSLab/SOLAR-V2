"""
ARC Task: f35d900a (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 5]
    bgc, c1, c2 = random.sample(cols, 3)
    return {"bgc": bgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, c1=None, c2=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c != 5]
    if bgc is None or c1 is None or c2 is None:
        bgc, c1, c2 = random.sample(cols, 3)
    H = max(4, min(30, max_h))
    W = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, H))
    w = unifint(diff_lb, diff_ub, (4, W))
    oh = unifint(diff_lb, diff_ub, (4, h))
    ow = unifint(diff_lb, diff_ub, (4, w))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    r0, c0, r1, cc1 = loci, locj, loci + oh - 1, locj + ow - 1
    bx = set()
    for c in range(c0, cc1 + 1):
        bx.add((r0, c)); bx.add((r1, c))
    for r in range(r0, r1 + 1):
        bx.add((r, c0)); bx.add((r, cc1))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def fill(g, col, cells):
        for (r, c) in cells:
            if 0 <= r < h and 0 <= c < w:
                g[r][c] = col

    def nbrs(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    ul, lr, ur, ll = (r0, c0), (r1, cc1), (r0, cc1), (r1, c0)
    for g in (gi, go):
        fill(g, c1, {ul, lr})
        fill(g, c2, {ur, ll})
    fill(go, c1, nbrs(ur) | nbrs(ll))
    fill(go, c2, nbrs(ul) | nbrs(lr))
    crns = [ul, ur, ll, lr]
    man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    for c in crns:
        rem = [x for x in crns if x != c]
        valids = set()
        for ij in bx:
            d = man(c, ij)
            if d <= min(man(ij, x) for x in rem) and d > 1 and d % 2 == 0:
                valids.add(ij)
        fill(go, 5, valids)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    palette = sorted({int(I[r, c]) for r, c in dots})

    def other_of(col):
        if len(palette) < 2:
            return col
        return palette[0] if col == palette[1] else palette[1]

    ops, sels = [], []

    # Phase 1 — colour at a time: all dots of colour A are ringed together
    # (painted with B), then all dots of colour B together (painted with A).
    for col in palette:
        ring = set()
        for (r, c) in dots:
            if int(I[r, c]) != col:
                continue
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) != (r, c) and 0 <= rr < h and 0 <= cc < w:
                        ring.add((rr, cc))
        if ring:
            ops.append(other_of(col))
            sels.append(sel_of(sorted(ring)))

    # Phase 2 — corner at a time: each dot emits its dashed 5-trail along the
    # two box edges leaving it, up to the halfway point toward neighbouring dots.
    rs = [r for r, _ in dots]
    cs = [c for _, c in dots]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    box = set()
    for c in range(c0, c1 + 1):
        box.add((r0, c)); box.add((r1, c))
    for r in range(r0, r1 + 1):
        box.add((r, c0)); box.add((r, c1))
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    done = set(corners)
    for cr in corners:
        others = [x for x in corners if x != cr]
        trail = []
        for p in sorted(box):
            if p in done:
                continue
            d = man(cr, p)
            if d <= min(man(p, x) for x in others) and d > 1 and d % 2 == 0:
                trail.append(p)
        if trail:
            done.update(trail)
            ops.append(5)
            sels.append(sel_of(trail))

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
                        f"num_examples+1 ({num_examples + 1}) for task f35d900a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f35d900a"
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
                                f"for task f35d900a"
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
                    f"Failed to build a complete episode for task f35d900a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f35d900a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
