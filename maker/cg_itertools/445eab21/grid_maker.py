"""
ARC Task: 445eab21 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; box colours are arbitrary
    # (the rule is "colour of the box with the largest bbox area").
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None:
        bgc = random.choice(list(range(10)))
    hi_h = max(10, min(30, max_h))
    hi_w = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, hi_h))
    w = unifint(diff_lb, diff_ub, (10, hi_w))
    remcols = [c for c in range(10) if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    succ = 0
    tr = 0
    bigcol, area = 0, 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = random.randint(3, 7)
        ow = random.randint(3, 7)
        if oh * ow == area:
            continue
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        col = random.choice(remcols)
        if bd.issubset(indss):
            remcols = [c for c in remcols if c != col]
            for (a, b) in bd:
                if a in (loci, loci + oh - 1) or b in (locj, locj + ow - 1):
                    gi[a][b] = col
            succ += 1
            indss = indss - bd
            if oh * ow > area:
                bigcol, area = col, oh * ow
        tr += 1
    if succ == 0:
        raise ValueError("no box placed")
    go = [[bigcol] * 2 for _ in range(2)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry: max row-count x number of
    # rows = width x height of a box): the answer is the colour of the hollow box
    # whose bounding box is largest. Instead of cropping an anonymous 2x2 corner of
    # the grid, this route crops onto the winning box itself -- its top-left 2x2
    # corner (3 box-edge cells + 1 interior cell) -- and then paints in what the
    # corner is missing, so the trajectory shows WHICH box was chosen.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the generator paints the canvas with bgc first; boxes are thin
    # outlines, so bgc dominates. Cross-check with the examples when available.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ex_i, _ in examples:
            E = np.asarray(ex_i, dtype=int)
            votes[Counter(E.flatten().tolist()).most_common(1)[0][0]] += 1
        ex_bg = votes.most_common(1)[0][0]
        if ex_bg in set(I.flatten().tolist()):
            bgc = ex_bg

    # Single-colour 4-connected objects; score each by bbox area.
    seen = np.zeros((hi, wi), dtype=bool)
    best = None  # (area, col, r0, c0)
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            rs, cs = [], []
            while q:
                y, x = q.popleft()
                rs.append(y); cs.append(x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            a = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if best is None or a > best[0]:
                best = (a, col, min(rs), min(cs))

    ops, sels = [], []
    _, col, r0, c0 = best

    # 1. Crop the canvas onto the winning box's top-left 2x2 corner
    #    (a full rectangle: the corner of the chosen object).
    ops.append(33); sels.append([r0, c0, 1, 1])

    # Grid after the transparent crop: zeros are not carried over.
    cur = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            v = int(I[r0 + i, c0 + j])
            cur[i, j] = v if v != 0 else 0

    # 2. Fill in the corner cells that do not already carry the box colour
    #    (the interior cell, plus any cells dropped by the crop when col == 0).
    fix = [(i, j) for i in range(2) for j in range(2) if cur[i, j] != col]
    if fix:
        ops.append(col); sels.append(sel_of(fix))

    ops.append(34); sels.append([0, 0, 1, 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 445eab21"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 445eab21"
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
                                f"for task 445eab21"
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
                    f"Failed to build a complete episode for task 445eab21 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"445eab21-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
