"""
ARC Task: d9f24cd1 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """bgc / linc (border seeds) / dotc (interior dots) + one fixed rotation
    for the whole episode, so the seed-edge (and hence the jog direction)
    is the same in every example and in the test."""
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    dotc = random.choice(rem)
    rot = random.choice(['identity', 'rot90', 'rot180', 'rot270'])
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "rot": rot}


# -------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc,
             rot='identity') -> dict:
    # rot90 / rot270 swap the final dimensions -> sample accordingly
    if rot in ('rot90', 'rot270'):
        hlim, wlim = max_w, max_h
    else:
        hlim, wlim = max_h, max_w
    h = unifint(diff_lb, diff_ub, (5, hlim))
    w = unifint(diff_lb, diff_ub, (5, wlim))

    locopts = interval(1, w - 1, 1)
    maxnloc = max(1, (w - 2) // 2)
    nlins = unifint(diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for k in range(nlins):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        locs.append(loc)

    ndots = unifint(diff_lb, diff_ub, (1, maxnloc))
    locopts = interval(1, w - 1, 1)
    dotlocs = []
    for k in range(ndots):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        dotlocs.append(loc)

    gi = canvas(bgc, (h, w))
    for l in locs:
        gi = fill(gi, linc, {(h - 1, l)})
    dotlocs2 = []
    for l in dotlocs:
        jj = randint(1, h - 2)
        gi = fill(gi, dotc, {(jj, l)})
        dotlocs2.append(jj)

    go = tuple(e for e in gi)
    for linloc in locs:
        if linloc in dotlocs:
            jj = dotlocs2[dotlocs.index(linloc)]
            go = fill(go, linc, connect((h - 1, linloc), (jj + 1, linloc)))
            go = fill(go, linc, connect((jj + 1, linloc + 1), (0, linloc + 1)))
        else:
            go = fill(go, linc, connect((h - 1, linloc), (0, linloc)))

    rotf = {'identity': identity, 'rot90': rot90,
            'rot180': rot180, 'rot270': rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------- derivation ---
def derive_operations(I, O):
    """Rule (measured from I only):
       * bgc = majority colour.
       * linc = the non-bgc colour sitting on the outer frame; all its cells
         lie on ONE edge (never a corner) -> that edge is the 'launch' edge.
       * growth direction g = perpendicular, pointing into the grid.
         jog direction j = g turned 90 deg clockwise on screen
         (up->right, right->down, down->left, left->up).
       * dotc = the non-bgc, non-linc colour, only ever interior.
       For each seed: shoot a linc ray from the seed along g.  If a dotc cell
       is hit, the ray stops one cell short of it, steps once along j into the
       neighbouring lane, and continues along g to the far edge.  If no dot is
       hit, the ray runs the whole way.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- launch edge / seed cells: non-bgc cells on the outer frame ---------
    border = [(r, c) for r in range(h) for c in range(w)
              if r == 0 or r == h - 1 or c == 0 or c == w - 1]
    lin_cells = [(r, c) for (r, c) in border if int(I[r, c]) != bgc]
    if not lin_cells:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    linc = int(I[lin_cells[0][0], lin_cells[0][1]])

    # --- dot colour: the other non-bgc colour, found strictly inside --------
    dotc = None
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v != linc:
                dotc = v
                break
        if dotc is not None:
            break

    # --- orientation (seeds never occupy a corner -> unambiguous) -----------
    if all(r == 0 for r, c in lin_cells):
        g, j = (1, 0), (0, -1)          # seeds on top edge
    elif all(r == h - 1 for r, c in lin_cells):
        g, j = (-1, 0), (0, 1)          # seeds on bottom edge
    elif all(c == 0 for r, c in lin_cells):
        g, j = (0, 1), (1, 0)           # seeds on left edge
    else:
        g, j = (0, -1), (-1, 0)         # seeds on right edge

    # --- draw one ray per seed, seed by seed along the edge ----------------
    for (sr, sc) in sorted(lin_cells):
        ray = []
        r, c = sr + g[0], sc + g[1]
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r += g[0]; c += g[1]

        didx = None
        if dotc is not None:
            for i, (rr, cc) in enumerate(ray):
                if int(I[rr, cc]) == dotc:
                    didx = i
                    break

        if didx is None:
            main, jog = ray, []
        else:
            main = ray[:didx]                       # stop one short of the dot
            last = main[-1] if main else (sr, sc)
            jog = []
            r, c = last[0] + j[0], last[1] + j[1]   # step sideways into the lane
            while 0 <= r < h and 0 <= c < w:
                jog.append((r, c))
                r += g[0]; c += g[1]

        if main:                                    # the ray itself
            ops.append(linc); sels.append(sel_of(main))
        if jog:                                     # its continuation past the dot
            ops.append(linc); sels.append(sel_of(jog))

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
                        f"num_examples+1 ({num_examples + 1}) for task d9f24cd1"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d9f24cd1"
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
                                f"for task d9f24cd1"
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
                    f"Failed to build a complete episode for task d9f24cd1 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d9f24cd1-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
