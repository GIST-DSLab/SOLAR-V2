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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 8)
    remcols = random.sample(remcols, numcols)
    return {"bgc": bgc, "remcols": remcols}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + int(round(diff_lb * (hi - lo)))
    b = lo + int(round(diff_ub * (hi - lo)))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if b < a:
        a, b = b, a
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, remcols=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.randint(1, 9)
    if not remcols:
        rc = [c for c in range(1, 10) if c != bgc]
        remcols = random.sample(rc, random.randint(1, 8))
    hub = min(10, max_h // 2, max_w // 2)
    if hub < 3:
        raise ValueError("grid too small")
    h = _unifint(diff_lb, diff_ub, 3, hub)
    w = h
    canv = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, 2, h * w - 1)
    allc = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(allc))
    obj = {start: random.choice(remcols)}
    bx = set(allc)
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx.discard(ch)
    for (i, j), v in obj.items():
        canv[i, j] = v
    gi = canv
    r90 = np.rot90(gi, 3)    # DSL rot90 (CW)
    r180 = np.rot90(gi, 2)
    r270 = np.rot90(gi, 1)
    top = np.hstack([gi, r90])
    bot = np.hstack([r270, r180])
    blk = np.vstack([top, bot])
    fullh = _unifint(diff_lb, diff_ub, 2 * h, max_h)
    fullw = _unifint(diff_lb, diff_ub, 2 * w, max_w)
    big = np.full((fullh, fullw), bgc, dtype=int)
    loci = random.randint(0, fullh - 2 * h)
    locj = random.randint(0, fullw - 2 * w)
    big[loci:loci + 2 * h, locj:locj + 2 * w] = blk
    ys, xs = np.where(big != bgc)
    sub = big[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    go = sub[:sub.shape[0] // 2, :sub.shape[1] // 2]
    return {"input": big.tolist(), "output": go.tolist()}


def _quadrant(I, bgc):
    ys, xs = np.where(I != bgc)
    if len(ys) == 0:
        return None
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    hh = (rmax - rmin + 1) // 2
    ww = (cmax - cmin + 1) // 2
    return rmin, cmin, hh, ww


def derive_operations(I, O, examples=None):
    # Rule: the non-background cells form a 2h x 2w four-fold rotationally
    # symmetric block; the answer is its top-left quadrant. The golf reading
    # (bbox of non-bg, keep the upper half of rows and left half of columns)
    # is the same carving as this, so the route is one crop of that quadrant.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background colour is fixed per episode: read it from the demonstrations.
    bgc = None
    if examples:
        votes = Counter()
        for ex in examples:
            try:
                EI, EO = ex[0], ex[1]
            except Exception:
                EI, EO = ex["input"], ex["output"]
            EI = np.asarray(EI, dtype=int)
            EO = np.asarray(EO, dtype=int)
            for c in set(EI.flatten().tolist()):
                q = _quadrant(EI, c)
                if q is None:
                    continue
                r0, c0, hh, ww = q
                if hh > 0 and ww > 0 and EI[r0:r0 + hh, c0:c0 + ww].shape == EO.shape \
                        and np.array_equal(EI[r0:r0 + hh, c0:c0 + ww], EO):
                    votes[c] += 1
        present = set(I.flatten().tolist())
        for c, _ in votes.most_common():
            if c in present:
                bgc = c
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    rmin, cmin, hh, ww = _quadrant(I, bgc)

    ops, sels = [], []
    # CropGrid to the full top-left-quadrant rectangle of the symmetric block
    # (whole rectangle intended, bgc cells included; bgc != 0 so the
    # transparent copy keeps them).
    ops.append(33); sels.append([rmin, cmin, hh - 1, ww - 1])
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
