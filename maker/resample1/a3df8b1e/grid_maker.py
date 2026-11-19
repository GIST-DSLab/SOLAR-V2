"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round((b - a) * d)))

    if bgc is None or linc is None:
        bgc, linc = random.sample(range(10), 2)

    def identity(g): return [list(r) for r in g]
    def dmirror(g): return [list(r) for r in zip(*g)]
    def cmirror(g): return [list(r)[::-1] for r in zip(*g)][::-1]
    def vmirror(g): return [list(r)[::-1] for r in g]
    def hmirror(g): return [list(r) for r in g[::-1]]
    def rot90(g): return [list(r) for r in zip(*g[::-1])]
    def rot180(g): return [list(r)[::-1] for r in g[::-1]]
    def rot270(g): return [list(r) for r in zip(*g)][::-1]

    mfs = [(identity, False), (dmirror, True), (cmirror, True), (vmirror, False),
           (hmirror, False), (rot90, True), (rot180, False), (rot270, True)]
    nmfs = random.choice((1, 2))
    fns = random.sample(mfs, nmfs)
    transposed = sum(1 for f, t in fns if t) % 2 == 1

    lim_h = max_w if transposed else max_h   # portrait height bound
    lim_w = max_h if transposed else max_w   # portrait width bound
    lim_h = min(30, lim_h)
    lim_w = min(10, lim_w, lim_h - 1)
    if lim_w < 2:
        raise ValueError("grid too small")
    w = unifint(diff_lb, diff_ub, (2, lim_w))
    h = unifint(diff_lb, diff_ub, (w + 1, lim_h))

    gi = [[bgc] * w for _ in range(h)]
    gi[h - 1][0] = linc
    go = [row[:] for row in gi]
    r, c, direc = h - 1, 0, 1
    while True:
        r, c = r - 1, c + direc
        if c == w - 1 or c == 0:
            direc *= -1
        if r < 0:
            break
        go[r][c] = linc

    for fn, _ in fns:
        gi = fn(gi)
        go = fn(go)
    gix = [row[:] for row in gi]
    gox = [row[:] for row in go]
    numlins = unifint(diff_lb, diff_ub, (1, 4))

    def overlay(g, src):
        for i in range(len(g)):
            for j in range(len(g[0])):
                if src[i][j] == linc:
                    g[i][j] = linc

    if numlins > 1:
        overlay(gi, hmirror(gix)); overlay(go, hmirror(gox))
    if numlins > 2:
        overlay(gi, vmirror(gix)); overlay(go, vmirror(gox))
    if numlins > 3:
        overlay(gi, hmirror(vmirror(gix))); overlay(go, hmirror(vmirror(gox)))
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    corners = [(0, 0), (0, wi - 1), (hi - 1, 0), (hi - 1, wi - 1)]
    cset = set(corners)
    noncorner = [int(I[r, c]) for r in range(hi) for c in range(wi) if (r, c) not in cset]
    bgc = Counter(noncorner).most_common(1)[0][0]
    others = [v for v in set(I.flatten().tolist()) if v != bgc]
    linc = others[0]

    portrait = hi > wi
    H, W = (hi, wi) if portrait else (wi, hi)

    def to_grid(r, c):
        return (r, c) if portrait else (c, r)

    grid = I.copy()
    ops, sels = [], []
    for (gr, gc) in corners:
        if I[gr, gc] != linc:
            continue
        r0, c0 = (gr, gc) if portrait else (gc, gr)
        dr = 1 if r0 == 0 else -1
        d = 1 if c0 == 0 else -1
        r, c = r0, c0
        path = []
        while 0 <= r < H:
            path.append(to_grid(r, c))
            r += dr
            c += d
            if c == W - 1 or c == 0:
                d *= -1
        # paint the line outward from its corner, skipping cells already linc
        for (pr, pc) in path:
            if grid[pr, pc] != linc:
                ops.append(linc)
                sels.append(sel_of([(pr, pc)]))
                grid[pr, pc] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
