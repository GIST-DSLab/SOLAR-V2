"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    rem = [c for c in cols if c != markercol]
    bgcols = random.sample(rem, random.randint(1, 8))
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h, max_w, markercol=None, bgcols=None) -> dict:
    if markercol is None:
        markercol = random.randint(1, 9)
    if not bgcols:
        rem = [c for c in range(1, 10) if c != markercol]
        bgcols = random.sample(rem, random.randint(1, 8))

    def unifint(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, round(lo + d * (hi - lo))))

    h = unifint(10, max(10, max_h))
    w = unifint(10, max(10, max_w))
    oh = random.randint(2, h // 4)
    ow = random.randint(2, w // 4)
    gi = [[random.choice(bgcols) for _ in range(w)] for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, unifint(0, (h * w) // 2)):
        gi[i][j] = 0
    occs = [(i, j) for i in range(h - oh + 1) for j in range(w - ow + 1)
            if all(gi[a][b] == 0 for a in range(i, i + oh) for b in range(j, j + ow))]
    cover = [(i + random.randrange(oh), j + random.randrange(ow)) for i, j in occs]
    for (a, b) in cover:
        gi[a][b] = random.choice(bgcols)
    go = [r[:] for r in gi]
    noccs = unifint(2, max(2, (h * w) // ((oh + 2) * (ow + 2))))
    free = set(inds)
    succ = 0
    tr = 0
    while tr < 5 * noccs and succ < noccs:
        tr += 1
        i = random.randint(0, h - oh)
        j = random.randint(0, w - ow)
        ring = {(a, b) for a in range(i - 1, i + oh + 1) for b in range(j - 1, j + ow + 1)
                if not (i <= a < i + oh and j <= b < j + ow)}
        if not ring <= free:
            continue
        succ += 1
        free -= ring
        for a in range(i, i + oh):
            for b in range(j, j + ow):
                gi[a][b] = 0
                go[a][b] = 0
        for a, b in ring:
            go[a][b] = markercol
            if succ == 1:
                gi[a][b] = markercol
        if succ > 1:
            sides = [[(i - 1, b) for b in range(j, j + ow)],
                     [(i + oh, b) for b in range(j, j + ow)],
                     [(a, j - 1) for a in range(i, i + oh)],
                     [(a, j + ow) for a in range(i, i + oh)]]
            for s in sides:
                a, b = random.choice(s)
                gi[a][b] = random.choice(bgcols)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def ring(r0, c0, r1, c1):
        s = set()
        for c in range(c0, c1 + 1):
            s.add((r0, c)); s.add((r1, c))
        for r in range(r0, r1 + 1):
            s.add((r, c0)); s.add((r, c1))
        return s

    def frame_of(g, col):
        # colour whose cells form exactly one hollow rectangle with an all-zero interior
        cells = {(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == col}
        if not cells:
            return None
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 < 3 or c1 - c0 < 3 or cells != ring(r0, c0, r1, c1):
            return None
        if (g[r0 + 1:r1, c0 + 1:c1] != 0).any():
            return None
        return (r0, c0, r1, c1)

    # convention from the demonstrations: the colour the rule draws frames with
    pref = []
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                d = eo[ei != eo]
                if d.size:
                    pref.append(int(np.bincount(d).argmax()))
    order = list(dict.fromkeys(pref)) + [c for c in range(1, 10) if c not in pref]

    marker = None
    for col in order:
        f = frame_of(I, col)
        if f is not None:
            marker = (col,) + f
            break

    ops, sels = [], []
    if marker is not None:
        col, r0, c0, r1, c1 = marker
        oh, ow = r1 - r0 - 1, c1 - c0 - 1
        # every hidden oh x ow all-zero block (other than the marker's own) whose frame fits
        targets = [(r, c) for r in range(1, hi - oh) for c in range(1, wi - ow)
                   if (r, c) != (r0 + 1, c0 + 1) and not I[r:r + oh, c:c + ow].any()]
        if targets:
            # copy the marker frame: a full rectangle; its zero interior is never copied
            ops.append(28); sels.append([r0, c0, r1 - r0, c1 - c0])
            g = I.copy()
            for (r, c) in targets:
                rg = ring(r - 1, c - 1, r + oh, c + ow)
                if all(g[a, b] == col for a, b in rg):
                    continue  # stamping here would change nothing
                for a, b in rg:
                    g[a, b] = col
                # stamp the marker frame around this hidden block
                ops.append(30); sels.append([r - 1, c - 1, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
