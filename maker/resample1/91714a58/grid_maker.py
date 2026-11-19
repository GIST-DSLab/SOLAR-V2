"""
ARC Task: 91714a58 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, targc = random.sample(cols, 2)
    return {"bgc": bgc, "targc": targc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, targc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if bgc is None or targc is None:
        bgc, targc = random.sample(range(10), 2)
    remcols = [c for c in range(10) if c != bgc]
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    gi = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    noise = random.sample(inds, nnoise)
    ih = random.randint(2, h // 2)
    iw = random.randint(2, w // 2)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    go = [row[:] for row in gi]
    for i in range(loci, loci + ih):
        for j in range(locj, locj + iw):
            go[i][j] = targc
    for (i, j) in noise:
        gi[i][j] = random.choice(remcols)
    for i in range(loci, loci + ih):
        for j in range(locj, locj + iw):
            gi[i][j] = targc
    return {"input": gi, "output": go}


def _max_rect(I, color):
    h, w = I.shape
    best = (0, None)
    heights = [0] * w
    for r in range(h):
        for c in range(w):
            heights[c] = heights[c] + 1 if I[r, c] == color else 0
        for c in range(w):
            if heights[c] == 0:
                continue
            mh = heights[c]
            for c2 in range(c, w):
                if heights[c2] == 0:
                    break
                mh = min(mh, heights[c2])
                area = mh * (c2 - c + 1)
                if area > best[0]:
                    best = (area, (r - mh + 1, c, mh, c2 - c + 1))
    return best


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the majority colour of the demonstration outputs (rect <= 1/4 area).
    bgc = None
    if examples:
        cnt = Counter()
        for ex in examples:
            eo = np.asarray(ex[1], dtype=int)
            cnt.update(eo.flatten().tolist())
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # The solid rectangle = largest all-one-colour (non-bgc) rectangle in I.
    best_area, best_rect = 0, None
    for col in set(I.flatten().tolist()):
        if col == bgc:
            continue
        area, rect = _max_rect(I, col)
        if area > best_area:
            best_area, best_rect = area, rect

    ops, sels = [], []
    if best_rect is not None:
        r0, c0, rh, rw = best_rect
        inside = lambda r, c: r0 <= r < r0 + rh and c0 <= c < c0 + rw
        noise = [(r, c) for r in range(hi) for c in range(wi)
                 if I[r, c] != bgc and not inside(r, c)]
    else:
        noise = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # Erase every noise pixel outside the rectangle back to background.
    if noise:
        ops.append(bgc); sels.append(sel_of(noise))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 91714a58"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 91714a58"
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
                                f"for task 91714a58"
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
                    f"Failed to build a complete episode for task 91714a58 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"91714a58-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
