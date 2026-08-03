"""
ARC Task: 6cdd2623 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter, defaultdict


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = choice(cols)
    linc = choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, linc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    remcols = remove(bgc, cols)
    remcols = remove(linc, remcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, 7))
    noisecols = sample(remcols, nnoisecols)
    c = canvas(bgc, (h, w))
    ininds = totuple(shift(asindices(canvas(-1, (h - 2, w - 1))), (1, 1)))
    fixinds = sample(ininds, nnoisecols)
    fixobj = {(col, ij) for col, ij in zip(list(noisecols), fixinds)}
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, fixobj)
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    noise = sample(totuple(asindices(c) - set(fixinds)), nnoise)
    noise = {(choice(remcols), ij) for ij in noise}
    gi = paint(gi, noise)
    ilocs = interval(1, h - 1, 1)
    jlocs = interval(1, w - 1, 1)
    aa, bb = sample((0, 1), 2)
    nilocs = unifint(diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = unifint(diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = sample(ilocs, nilocs)
    jlocs = sample(jlocs, njlocs)
    for ii in ilocs:
        gi = fill(gi, linc, {(ii, 0)})
        gi = fill(gi, linc, {(ii, w - 1)})
        go = fill(go, linc, connect((ii, 0), (ii, w - 1)))
    for jj in jlocs:
        gi = fill(gi, linc, {(0, jj)})
        gi = fill(gi, linc, {(h - 1, jj)})
        go = fill(go, linc, connect((0, jj), (h - 1, jj)))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = the canvas colour of I (generator paints canvas(bgc) then sprinkles noise
    # over at most a third of it, so bgc is the majority colour of I).
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # group I's cells by colour; the marker colour is the largest colour group whose cells
    # all sit on the border frame of I (noise is interior-or-anywhere but never this colour).
    pts = defaultdict(list)
    for r in range(h):
        for c in range(w):
            pts[int(I[r, c])].append((r, c))

    def on_border(p):
        return p[0] == 0 or p[0] == h - 1 or p[1] == 0 or p[1] == w - 1

    cands = [col for col, ps in pts.items() if col != bgc and all(on_border(p) for p in ps)]

    ops, sels = [], []

    # 1. wipe the noise: the whole canvas returns to its background colour
    ops.append(bgc)
    sels.append([0, 0, h - 1, w - 1])

    if cands:
        linc = max(cands, key=lambda col: len(pts[col]))
        marks = set(pts[linc])
        # 2. connect each matching pair of markers facing each other across the grid
        rows = sorted({r for (r, c) in marks
                       if 0 < r < h - 1 and (r, 0) in marks and (r, w - 1) in marks})
        cols = sorted({c for (r, c) in marks
                       if 0 < c < w - 1 and (0, c) in marks and (h - 1, c) in marks})
        for r in rows:
            ops.append(int(linc))
            sels.append([r, 0, 0, w - 1])
        for c in cols:
            ops.append(int(linc))
            sels.append([0, c, h - 1, 0])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6cdd2623"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6cdd2623"
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
                                f"for task 6cdd2623"
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
                    f"Failed to build a complete episode for task 6cdd2623 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6cdd2623-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
