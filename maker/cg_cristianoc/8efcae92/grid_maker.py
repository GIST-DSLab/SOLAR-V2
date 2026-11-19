"""
ARC Task: 8efcae92 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc, sqc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc, **kwargs) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    num = _unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = [[bgc] * w for _ in range(h)]
    go = None
    inds = {(r, c) for r in range(h) for c in range(w)}
    succ, tr, maxtr = 0, 0, 4 * num
    oho = owo = None
    numdots = None
    while succ < num and tr < maxtr:
        if oho is None:
            oh = random.randint(2, h - 1)
            ow = random.randint(2, w - 1)
            oho, owo = oh, ow
        else:
            ohd = _unifint(diff_lb, diff_ub, (0, min(oho, h - 1 - oho)))
            owd = _unifint(diff_lb, diff_ub, (0, min(owo, w - 1 - owo)))
            ohd = min(oho, h - 1 - oho) - ohd
            owd = min(owo, w - 1 - owo) - owd
            oh = random.choice((oho - ohd, oho + ohd))
            ow = random.choice((owo - owd, owo + owd))
            oh = min(max(2, oh), h - 1)
            ow = min(max(2, ow), w - 1)
        mini = [(r, c) for r in range(oh) for c in range(ow)]
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        tr += 1
        if not cands:
            continue
        loc = random.choice(cands)
        placed = {(r + loc[0], c + loc[1]) for r, c in mini}
        if not placed.issubset(inds):
            continue
        succ += 1
        if go is None:
            numdots = _unifint(diff_lb, diff_ub, (1, (oh * ow) // 2 - 1))
            nd = numdots
        else:
            nd = _unifint(diff_lb, diff_ub, (0, min((oh * ow) // 2 - 1, numdots - 1)))
        minig = [[sqc] * ow for _ in range(oh)]
        for (r, c) in random.sample(mini, nd):
            minig[r][c] = dotc
        if go is None:
            go = [row[:] for row in minig]
        for r in range(oh):
            for c in range(ow):
                gi[r + loc[0]][c + loc[1]] = minig[r][c]
        nb = set()
        for (r, c) in placed:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb.add((r + dr, c + dc))
        inds = (inds - placed) - nb
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared by maker and code-golf entry): the grid holds several
    # solid rectangles of one fill colour sprinkled with a minority "dot"
    # colour; the answer is the rectangle carrying the most dots. The two
    # readings agree (the golf tie-breaks never matter since the generator
    # gives the first block strictly the most dots), so the operation is a
    # single crop to that block's full rectangle.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(cellset):
        seen, comps = set(), []
        for cell in cellset:
            if cell in seen:
                continue
            stack = [cell]
            seen.add(cell)
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in cellset and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    # background: colour whose complement splits into solid rectangles only
    counts = Counter(I.flatten().tolist())
    bgc, best_cnt = None, -1
    for c in sorted(counts):
        non = {(r, cc) for r in range(hi) for cc in range(wi) if I[r, cc] != c}
        if not non:
            continue
        ok = True
        for comp in components(non):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if np.any(I[min(rs):max(rs) + 1, min(cs):max(cs) + 1] == c):
                ok = False
                break
        if ok and counts[c] > best_cnt:
            best_cnt, bgc = counts[c], c
    if bgc is None:
        bgc = counts.most_common(1)[0][0]

    non = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc}
    non_counts = Counter(int(I[r, c]) for r, c in non)
    # dot colour = minority non-background colour (dots < half of every block)
    dotc = min(non_counts, key=lambda k: non_counts[k]) if len(non_counts) >= 2 else None

    best, best_dc = None, -1
    for comp in components(non):
        dc = 0 if dotc is None else sum(1 for (r, c) in comp if I[r, c] == dotc)
        if dc > best_dc:
            best_dc, best = dc, comp

    rs = [p[0] for p in best]
    cs = [p[1] for p in best]
    r0, c0 = min(rs), min(cs)
    h, w = max(rs) - r0 + 1, max(cs) - c0 + 1

    ops, sels = [], []
    # crop to the chosen block: the block is a solid rectangle, so its
    # full bbox is exactly the intended selection
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8efcae92"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8efcae92"
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
                                f"for task 8efcae92"
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
                    f"Failed to build a complete episode for task 8efcae92 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8efcae92-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
