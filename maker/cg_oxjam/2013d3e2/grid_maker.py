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

# ---------------------------------------------------------------------------
# DSL helpers (RE-ARC style), local versions
# ---------------------------------------------------------------------------

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    lo_i = max(a, min(b, lo_i))
    hi_i = max(a, min(b, hi_i))
    return random.randint(lo_i, hi_i)


def _neighbors(loc):
    r, c = loc
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)}


def sample_colors(num_examples=None) -> dict:
    # Only the canvas colour is a role the rule cares about (the rule is
    # "crop the top-left quadrant of the non-background block"); the object's
    # palette is free per instance.  Generator samples bgc from 1..9.
    bgc = random.choice(list(range(1, 10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    cols = list(range(1, 10))
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(10, min(max_h, max_w) // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)

    canv = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = {(r, c) for r in range(h) for c in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for loc in obj:
            dns |= _neighbors(loc)
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx = bx - {ch}
    gi = canv.copy()
    for (r, c), v in obj.items():
        gi[r, c] = v

    # DSL rot90 = CW = np.rot90(k=3); rot270 = CCW = np.rot90(k=1)
    gi1 = np.hstack([gi, np.rot90(gi, 3)])
    gi2 = np.hstack([np.rot90(gi, 1), np.rot90(gi, 2)])
    blk = np.vstack([gi1, gi2])

    fullh = _unifint(diff_lb, diff_ub, (2 * h, max(2 * h, max_h)))
    fullw = _unifint(diff_lb, diff_ub, (2 * w, max(2 * w, max_w)))
    out = np.full((fullh, fullw), bgc, dtype=int)
    loci = random.randint(0, fullh - 2 * h)
    locj = random.randint(0, fullw - 2 * w)
    out[loci:loci + 2 * h, locj:locj + 2 * w] = blk

    ys, xs = np.where(out != bgc)
    sub = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sh, sw = sub.shape
    go = sub[:sh // 2, :sw // 2]
    return {"input": out.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading of the task: the non-background cells form a four-fold
    # rotationally symmetric block; the answer is its top-left quadrant.
    # The code-golf entry reads it the same way: drop leading background
    # rows/columns, then keep a corner whose size is fixed by the block.
    # Its loop of transpose-and-trim is a byte-saving trick, not a different
    # way of splitting up the task. So the maker's route stays: one crop of a
    # rectangle measured from I alone.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background = the canvas fill colour (never 0 in this task, since colours
    # are 1..9). Take the most common colour, and check it against the
    # demonstrations when they are given.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ex_in, _ in examples:
            ei = np.asarray(ex_in, dtype=int)
            votes[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
        # the episode shares one canvas colour; use it if it is present in I
        cand = votes.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand

    ys, xs = np.where(I != bgc)
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    hh = (rmax - rmin + 1) // 2      # top half of the symmetric block
    ww = (cmax - cmin + 1) // 2      # left half of the symmetric block

    ops, sels = [], []
    # CropGrid to the full rectangle of the block's top-left quadrant
    # (bg cells included; bgc != 0, so the transparent copy keeps them).
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
