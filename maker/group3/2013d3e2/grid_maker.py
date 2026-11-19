"""
ARC Task: 2013d3e2 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = list(range(1, 10))
    if bgc is None:
        bgc = random.choice(cols)
    lim = max(3, min(10, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, lim))
    h = max(3, min(h, lim))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    numcols = max(1, min(8, numcols))
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    nc = max(2, min(nc, h * w - 1))
    bx = {(r, c) for r in range(h) for c in range(w)}
    start = random.choice(sorted(bx))
    objcells = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (r, c) in objcells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        dns.add((r + dr, c + dc))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        objcells[ch] = random.choice(remcols)
        bx = bx - {ch}
    for (r, c), v in objcells.items():
        g[r, c] = v
    rot90 = lambda a: np.rot90(a, 3)
    rot180 = lambda a: np.rot90(a, 2)
    rot270 = lambda a: np.rot90(a, 1)
    gi1 = np.hstack([g, rot90(g)])
    gi2 = np.hstack([rot270(g), rot180(g)])
    blk = np.vstack([gi1, gi2])
    fullh = unifint(diff_lb, diff_ub, (2 * h, max_h))
    fullw = unifint(diff_lb, diff_ub, (2 * w, max_w))
    fullh = max(2 * h, min(fullh, max_h))
    fullw = max(2 * w, min(fullw, max_w))
    gi = np.full((fullh, fullw), bgc, dtype=int)
    loci = random.randint(0, fullh - 2 * h)
    locj = random.randint(0, fullw - 2 * w)
    gi[loci:loci + 2 * h, locj:locj + 2 * w] = blk
    ys, xs = np.where(gi != bgc)
    sub = gi[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sub = sub[:sub.shape[0] // 2]
    go = sub[:, :sub.shape[1] // 2]
    return {"input": gi.tolist(), "output": go.tolist()}


def _bgc_of(I, examples=None):
    cnt = Counter(I.flatten().tolist())
    cands = []
    for c, n in cnt.most_common():
        ys, xs = np.where(I != c)
        if len(ys) == 0:
            continue
        H = ys.max() - ys.min() + 1
        W = xs.max() - xs.min() + 1
        if H == W and H % 2 == 0:
            cands.append(c)
    if examples:
        common = None
        for ex in examples:
            EI = np.asarray(ex[0], dtype=int)
            s = set(np.unique(EI).tolist())
            common = s if common is None else common & s
        for c in cands:
            if common and c in common:
                return c
    if cands:
        return cands[0]
    return cnt.most_common(1)[0][0]


def derive_operations(I, O, examples=None):
    # Rule: the non-background cells form a 2h x 2w four-fold rotational block;
    # the answer is its top-left quadrant. Here the quadrant (a full rectangle,
    # background cells included) is slid as one object to the grid's top-left
    # corner with unit Moves, then the canvas is shrunk to the quadrant's size.
    I = np.asarray(I, dtype=int)
    bgc = _bgc_of(I, examples)

    ys, xs = np.where(I != bgc)
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    hh = (rmax - rmin + 1) // 2
    ww = (cmax - cmin + 1) // 2

    ops, sels = [], []
    # Whole quadrant rectangle is the intended object (bgc != 0, so all cells move).
    cur = [(r, c) for r in range(rmin, rmin + hh) for c in range(cmin, cmin + ww)]
    grabbed = False
    for _ in range(rmin):
        ops.append(20)
        sels.append(sel_of([]) if grabbed else sel_of(cur))
        grabbed = True
        cur = [(r - 1, c) for r, c in cur]
    for _ in range(cmin):
        ops.append(23)
        sels.append(sel_of([]) if grabbed else sel_of(cur))
        grabbed = True
        cur = [(r, c - 1) for r, c in cur]

    # Shrink the canvas to the quadrant now sitting at the top-left corner.
    ops.append(33); sels.append([0, 0, hh - 1, ww - 1])
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2013d3e2"
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
                                f"for task 2013d3e2"
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
                    f"Failed to build a complete episode for task 2013d3e2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2013d3e2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
