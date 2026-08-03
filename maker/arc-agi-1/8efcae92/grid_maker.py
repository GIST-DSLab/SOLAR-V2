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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    maxtr = 4 * num
    tr = 0
    gi = canvas(bgc, (h, w))
    go = None
    inds = asindices(gi)
    oho, owo = None, None
    numdots = None
    while succ < num and tr < maxtr:
        if oho is None and owo is None:
            oh = randint(2, h - 1)
            ow = randint(2, w - 1)
            oho = oh
            owo = ow
        else:
            ohd = unifint(diff_lb, diff_ub, (0, min(oho, h - 1 - oho)))
            owd = unifint(diff_lb, diff_ub, (0, min(owo, w - 1 - owo)))
            ohd = min(oho, h - 1 - oho) - ohd
            owd = min(owo, w - 1 - owo) - owd
            oh = choice((oho - ohd, oho + ohd))
            ow = choice((owo - owd, owo + owd))
            oh = min(max(2, oh), h - 1)
            ow = min(max(2, ow), w - 1)
        minig = canvas(sqc, (oh, ow))
        mini = asindices(minig)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        if not shift(mini, loc).issubset(inds):
            continue
        succ += 1
        if go is None:
            numdots = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2 - 1))
            nd = numdots
        else:
            nd = unifint(diff_lb, diff_ub, (0, min((oh * ow) // 2 - 1, numdots - 1)))
        locs = sample(totuple(mini), nd)
        minig = fill(minig, dotc, locs)
        if go is None:
            go = minig
        obj = asobject(minig)
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        inds = (inds - toindices(plcd)) - mapply(dneighbors, toindices(plcd))
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def components(cellset):
        seen = set()
        comps = []
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

    # --- identify background color dynamically ---
    # bgc = color c such that every 4-connected component of non-c cells is a
    # solid (fully-filled) rectangle. Blocks are solid rectangles of non-bgc;
    # the background blob is non-rectangular, so only true bgc passes.
    present = sorted(set(I.flatten().tolist()))
    counts = Counter(I.flatten().tolist())
    bgc = None
    best_bg_count = -1
    for c in present:
        noncells = {(r, cc) for r in range(hi) for cc in range(wi) if I[r, cc] != c}
        if not noncells:
            continue
        ok = True
        for comp in components(noncells):
            rs = [r for r, _ in comp]
            cs = [cc for _, cc in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if np.any(I[r0:r1 + 1, c0:c1 + 1] == c):
                ok = False
                break
        if ok and counts[c] > best_bg_count:
            best_bg_count = counts[c]
            bgc = c
    if bgc is None:
        bgc = counts.most_common(1)[0][0]

    # --- segment blocks (solid non-bgc rectangles) ---
    noncells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc}

    # dot color = least common non-bgc color (fill=sqc majority, dots=dotc minority)
    non_counts = Counter(I[r, c] for r, c in noncells)
    dotc = None
    if len(non_counts) >= 2:
        dotc = min(non_counts, key=lambda k: non_counts[k])

    # choose block with MOST dot cells (generator: first block has strictly max dots)
    best = None
    best_dc = -1
    for comp in components(noncells):
        if dotc is None:
            dc = 0
        else:
            dc = sum(1 for (r, c) in comp if I[r, c] == dotc)
        if dc > best_dc:
            best_dc = dc
            best = comp

    rs = [r for r, c in best]
    cs = [c for r, c in best]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops = []
    sels = []
    # crop working grid down to the chosen block's bounding box (full rectangle)
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
