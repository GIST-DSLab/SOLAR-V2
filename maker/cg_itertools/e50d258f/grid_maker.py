"""
ARC Task: e50d258f (RE-ARC) — LLM-generated grid_maker
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
    # colour 2 is fixed (it is the counted "red"); bgc and padcol are sampled
    cols = [c for c in range(10) if c != 2]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    padcol = random.choice(remcols)
    return {"bgc": bgc, "padcol": padcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, padcol=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    cols = [c for c in range(10) if c != 2]
    if padcol is None:
        padcol = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, padcol)]

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    gi = [[bgc] * w for _ in range(h)]
    num = unifint(diff_lb, diff_ub, (1, 10))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    bound = None
    go = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        if bd.issubset(indss):
            numcc = unifint(diff_lb, diff_ub, (1, 7))
            ccols = random.sample(remcols, min(numcc, len(remcols)))
            if succ == 0:
                numred = unifint(diff_lb, diff_ub, (1, oh * ow))
                bound = numred
            else:
                numred = unifint(diff_lb, diff_ub, (0, min(oh * ow, bound - 1)))
            base = random.choice(ccols)
            cc = [[base] * ow for _ in range(oh)]
            cci = [(a, b) for a in range(oh) for b in range(ow)]
            reds = set(random.sample(cci, numred))
            for (a, b) in cci:
                cc[a][b] = 2 if (a, b) in reds else random.choice(ccols)
            for (a, b) in cci:
                gi[loci + a][locj + b] = cc[a][b]
            if go is None:
                go = [row[:] for row in cc]
            succ += 1
            outb = {(loci + a, locj + b) for a in range(-1, oh + 1) for b in range(-1, ow + 1)}
            indss = indss - outb
        tr += 1
    if go is None:
        raise ValueError("no block placed")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by maker and golf entry): the scene is a set of solid
    # rectangular blocks on a background; the answer is the single block holding
    # the most red (2) cells (the generator makes that count strictly maximal),
    # cut out unchanged. Both readings treat "one block" as the unit and crop it,
    # so the trajectory is the same: one CropGrid on that block's full rectangle.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: dominant border colour, pooled over the demonstrations (same bgc per episode)
    border = []
    grids = [I]
    if examples:
        grids += [np.asarray(e[0], dtype=int) for e in examples]
    for g in grids:
        border += list(g[0, :]) + list(g[-1, :]) + list(g[:, 0]) + list(g[:, -1])
    bgc = Counter(int(x) for x in border).most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    best, best_cnt = None, -1
    for r0 in range(hi):
        for c0 in range(wi):
            if seen[r0, c0] or I[r0, c0] == bgc:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < hi and 0 <= cc < wi and not seen[rr, cc] and I[rr, cc] != bgc:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
            cnt = sum(1 for (r, c) in cells if I[r, c] == 2)
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            key = (cnt, area)
            if best is None or key > best_cnt:
                best_cnt = key
                best = (min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    br, bc, bh, bw = best
    ops, sels = [], []
    # crop canvas to the winning block — selection is exactly that full rectangle
    ops.append(33); sels.append([br, bc, bh - 1, bw - 1])
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e50d258f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e50d258f"
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
                                f"for task e50d258f"
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
                    f"Failed to build a complete episode for task e50d258f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e50d258f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
