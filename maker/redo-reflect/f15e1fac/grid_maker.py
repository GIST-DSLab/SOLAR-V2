"""
ARC Task: f15e1fac (RE-ARC) — LLM-generated grid_maker
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

# The 8 orientations of D4 the generator can apply to the finished picture.
TRANSFORMS = ('identity', 'vmirror', 'hmirror', 'rot180',
              'dmirror', 'cmirror', 'rot90', 'rot270')


def _apply_tf(g, name):
    if name == 'identity':
        return g.copy()
    if name == 'vmirror':
        return np.fliplr(g).copy()
    if name == 'hmirror':
        return np.flipud(g).copy()
    if name == 'rot180':
        return np.rot90(g, 2).copy()
    if name == 'dmirror':
        return np.transpose(g).copy()
    if name == 'cmirror':
        return np.rot90(np.transpose(g), 2).copy()
    if name == 'rot90':
        return np.rot90(g, 3).copy()
    if name == 'rot270':
        return np.rot90(g, 1).copy()
    raise ValueError(name)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(max(a, lo), min(b, hi))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(TRANSFORMS):
        examples = [{'transform': t} for t in TRANSFORMS]
        examples += [{'transform': random.choice(TRANSFORMS)}
                     for _ in range(n_ex - len(TRANSFORMS))]
        random.shuffle(examples)
    else:
        examples = [{'transform': t} for t in random.sample(TRANSFORMS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown
    return {'bgc': bgc, 'linc': linc, 'instance_plan': plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, transform=None) -> dict:
    if transform is None:
        transform = random.choice(TRANSFORMS)
    max_h = max(4, min(30, int(max_h)))
    max_w = max(4, min(30, int(max_w)))
    h = _unifint(diff_lb, diff_ub, (4, max_h))
    w = _unifint(diff_lb, diff_ub, (4, max_w))
    nsps = _unifint(diff_lb, diff_ub, (1, (w - 1) // 2))
    ngps = _unifint(diff_lb, diff_ub, (1, (h - 1) // 2))
    spsj = sorted(random.sample(range(1, w - 1), nsps))
    gpsi = sorted(random.sample(range(1, h - 1), ngps))

    gi = np.full((h, w), bgc, dtype=int)
    for jj in spsj:
        gi[0, jj] = linc
    for ii in gpsi:
        gi[ii, 0] = 2

    go = gi.copy()
    ofs = 0
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            cc = jj + ofs
            if 0 <= cc < w:
                go[a:b + 1, cc] = linc
        ofs += 1

    gi = _apply_tf(gi, transform)
    go = _apply_tf(go, transform)
    return {'input': tuple(tuple(int(v) for v in row) for row in gi),
            'output': tuple(tuple(int(v) for v in row) for row in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    # palette is exactly {bgc, linc, 2}; 2 is the divider marker colour
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    linc = [c for c in cnt if c != bgc and c != 2][0]

    two = [(int(r), int(c)) for r, c in zip(*np.where(I == 2))]
    lin = [(int(r), int(c)) for r, c in zip(*np.where(I == linc))]

    def edge_of(cells, hh, ww):
        if all(r == 0 for r, _ in cells):
            return 'top'
        if all(r == hh - 1 for r, _ in cells):
            return 'bottom'
        if all(c == 0 for _, c in cells):
            return 'left'
        return 'right'

    e2 = edge_of(two, h, w)      # edge carrying the 2 dividers
    el = edge_of(lin, h, w)      # perpendicular edge carrying the seeds

    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]  # bbox == the ENTIRE grid rectangle, background included

    # --- 1. reflect the picture into reading orientation: seeds on the leading
    #        edge, dividers on the perpendicular leading edge ---
    flips = []
    vertical_rays = el in ('top', 'bottom')
    if vertical_rays:                 # seeds live on a row -> bring it to row 0, dividers to col 0
        if el == 'bottom':
            flips.append(27)          # FlipV (up<->down)
        if e2 == 'right':
            flips.append(26)          # FlipH (left<->right)
    else:                             # seeds live on a column -> bring it to col 0, dividers to row 0
        if el == 'right':
            flips.append(26)          # FlipH
        if e2 == 'bottom':
            flips.append(27)          # FlipV

    J = I.copy()
    for op in flips:
        ops.append(op)
        sels.append(list(full))       # whole-grid rectangle: the reflection acts on everything
        J = np.fliplr(J) if op == 26 else np.flipud(J)

    # --- 2. draw each staircase ray, seed by seed, band by band, away from the seed ---
    if vertical_rays:
        seeds = sorted(int(c) for c in np.where(J[0, :] == linc)[0])
        divs = sorted(int(r) for r in np.where(J[:, 0] == 2)[0])
        span_end, lat_max = h - 1, w - 1
    else:
        seeds = sorted(int(r) for r in np.where(J[:, 0] == linc)[0])
        divs = sorted(int(c) for c in np.where(J[0, :] == 2)[0])
        span_end, lat_max = w - 1, h - 1

    bands = list(zip([0] + divs, [d - 1 for d in divs] + [span_end]))

    for s in seeds:
        for k, (a, b) in enumerate(bands):
            lat = s + k               # each divider crossed shifts the ray one step sideways
            if lat > lat_max:
                break                 # the ray has run off the grid
            if vertical_rays:
                cells = [(t, lat) for t in range(a, b + 1) if J[t, lat] != linc]
            else:
                cells = [(lat, t) for t in range(a, b + 1) if J[lat, t] != linc]
            if not cells:             # segment is just the seed cell itself
                continue
            for r, c in cells:
                J[r, c] = linc
            ops.append(int(linc))
            sels.append(sel_of(cells))

    # --- 3. reflect the drawn picture back into the original orientation ---
    for op in reversed(flips):
        ops.append(op)
        sels.append(list(full))       # whole-grid rectangle

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
                        f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
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
                                f"for task f15e1fac"
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
                    f"Failed to build a complete episode for task f15e1fac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f15e1fac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
