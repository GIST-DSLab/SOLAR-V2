"""
ARC Task: 6d58a25d (RE-ARC) — LLM-generated grid_maker
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

# The scene is a fixed 10-cell "V" template (4 rows x 7 cols) plus scattered
# noise, and the generator rotates the whole thing by 0/90/180/270 -> the ray
# direction becomes down / left / up / right: 4 discrete structural variants.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]

CANON_SHAPE = ((0, 3), (1, 2), (1, 3), (1, 4),
               (2, 1), (2, 2), (2, 4), (2, 5), (3, 0), (3, 6))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    c1 = random.choice(remcols)                               # template colour
    c2 = random.choice([c for c in remcols if c != c1])       # noise / ray colour

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, rot=None) -> dict:
    if rot is None:
        rot = random.choice(VARIANTS)["rot"]

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    # a 90/270 rotation transposes the canvas -> both bounds must hold both ways
    if rot % 2 == 1:
        hub = wub = min(max_h, max_w)
    else:
        hub, wub = max_h, max_w

    shp = [(0, 3), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 4), (2, 5), (3, 0), (3, 6)]

    h = unifint(diff_lb, diff_ub, (5, max(5, hub)))
    w = unifint(diff_lb, diff_ub, (8, max(8, wub)))
    loci = random.randint(0, h - 4)
    locj = random.randint(0, w - 7)
    plcd = {(r + loci, c + locj) for r, c in shp}
    rem = [(r, c) for r in range(h) for c in range(w) if (r, c) not in plcd]
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nnoise = max(1, min(nnoise, len(rem)))
    nois = random.sample(rem, nnoise)

    gi = np.full((h, w), bgc, dtype=int)
    for r, c in nois:
        gi[r, c] = c2
    for r, c in plcd:
        gi[r, c] = c1

    # per template column: a noise cell at/below the template's topmost cell
    # there fires a ray running from under its lowest cell to the grid edge
    span = {}
    for r, c in plcd:
        lo, hg = span.get(c, (r, r))
        span[c] = (min(lo, r), max(hg, r))
    go = gi.copy()
    for c, (rmin, rmax) in span.items():
        if any(gi[r, c] == c2 for r in range(rmin, h)):
            for r in range(rmax + 1, h):
                go[r, c] = c2

    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {'input': gi.tolist(), 'output': go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule, read off I alone:
      * one colour draws a fixed 10-cell 'V' template (4 x 7) somewhere;
      * the other non-background colour is scattered noise;
      * put the grid in the pose where the V opens downward -- rays run down;
      * a template column fires iff some noise cell sits in that column at or
        below the template's topmost cell there;
      * a firing column is filled with the noise colour from just under the
        template's lowest cell in that column all the way to the grid edge.
    Which rays are drawn is fixed by the rule.  The ROUTE is read off the input
    as well, so different inputs get solved the way different people solve them:
    the sweep starts from the edge the template itself sits nearer to, and when
    the firing columns form one block the solver may first lay the shared slab
    beneath all of them and then reach each column up to the template, instead
    of drawing every ray in a single stroke.  Nothing here looks at O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    def submit_only():
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    palette = sorted(set(I.flatten().tolist()))
    flat = np.arange(hi * wi).reshape(hi, wi)

    # locate the template and the pose in which its V opens downward
    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(flat, k)
        for col in palette:
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) != len(CANON_SHAPE):
                continue
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            if tuple(sorted((r - r0, c - c0) for r, c in cells)) == CANON_SHAPE:
                found = (J, Jidx, col, cells)
                break
        if found is not None:
            break
    if found is None:
        return submit_only()

    J, Jidx, shape_col, shape_cells = found
    hJ, wJ = J.shape

    # of the two remaining colours the rarer one is the noise: the generator
    # never lets noise reach half the canvas, so the background always leads
    others = [c for c in palette if c != shape_col]
    if not others:
        return submit_only()
    noise_col = min(others, key=lambda c: int((J == c).sum()))

    by_col = {}
    for (r, c) in shape_cells:
        by_col.setdefault(c, []).append(r)

    rays = []                                    # (column, first row of its ray)
    for j in sorted(by_col):
        first_r, last_r = min(by_col[j]), max(by_col[j])
        if not any(J[i, j] == noise_col for i in range(first_r, hJ)):
            continue                             # nothing under it -> stays quiet
        if last_r + 1 > hJ - 1:
            continue                             # template already on the edge
        rays.append((j, last_r + 1))
    if not rays:
        return submit_only()

    work = J.copy()

    def paint(cells):
        """One Color(noise) over a region of the canonical view, if it shows."""
        cells = list(cells)
        if not cells or all(work[r, c] == noise_col for r, c in cells):
            return
        ops.append(int(noise_col))
        sels.append(sel_of(divmod(int(Jidx[r, c]), wi) for r, c in cells))
        for r, c in cells:
            work[r, c] = noise_col

    # route choice 1 -- sweep from the edge the template leans towards
    tcols = list(by_col)
    near_left = (min(tcols) + max(tcols)) <= (wJ - 1)
    order = rays if near_left else list(reversed(rays))

    # route choice 2 -- shared slab first, then reach each column up; or stroke
    # each ray whole.  Both decided from the input (block shape + noise count).
    fcols = [j for j, _ in rays]
    contiguous = (max(fcols) - min(fcols) + 1) == len(fcols)
    use_base = contiguous and len(rays) > 1 and int((J == noise_col).sum()) % 2 == 0

    if use_base:
        base_r = max(start for _, start in rays)         # deepest ray start
        paint([(i, j) for i in range(base_r, hJ)
               for j in range(min(fcols), max(fcols) + 1)])
        for j, start in order:
            paint([(i, j) for i in range(start, base_r)])
    else:
        for j, start in order:
            paint([(i, j) for i in range(start, hJ)])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
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
                                f"for task 6d58a25d"
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
                    f"Failed to build a complete episode for task 6d58a25d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d58a25d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
