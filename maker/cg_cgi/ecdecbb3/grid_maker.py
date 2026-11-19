"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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

# How this differs from the maker's reading: the maker treats each dot as one unit.
# It paints a single trail from one line to the other and then stamps a halo at each
# contact point. Here each dot fires one RAY toward every line it faces. A dot outside
# the band of lines fires one ray; a dot between two lines fires one ray up and one
# down. Each ray is handled start to finish: paint the straight segment from the dot
# to where it hits the line, then stamp the line-coloured ring around that point.
# That ring paints over the segment cell next to the line, as in the generator.
# The trail cells, the halo cells and the final grid are the same either way; only
# the units and their order change.


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, linc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    ex = [{"mirror": False}, {"mirror": True}]
    ex += [{"mirror": random.choice([False, True])} for _ in range(max(0, n_ex - 2))]
    ex = ex[:max(1, n_ex)]
    random.shuffle(ex)
    plan = [dict(e) for e in ex] + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, mirror=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a))
        hi_ = a + round(ub * (b - a))
        return random.randint(min(lo, hi_), max(lo, hi_))

    if mirror is None:
        mirror = random.choice([True, False])
    hb, wb = (max_w, max_h) if mirror else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (4, min(30, hb)))
    w = unifint(diff_lb, diff_ub, (4, min(30, wb)))
    gi = [[bgc] * w for _ in range(h)]
    nl = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if not inds:
            break
        idx = random.choice(inds)
        locs.append(idx)
        inds = [x for x in inds if abs(x - idx) > 2]
    locs = sorted(locs)
    for ii in locs:
        gi[ii] = [linc] * w
    iopts = [i for i in range(h) if all(abs(i - l) > 1 for l in locs)]
    jopts = list(range(w))
    if not iopts:
        raise ValueError("no dot rows")
    ndots = unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for _ in range(ndots):
        if not jopts:
            break
        loci = random.choice(iopts)
        locj = random.choice(jopts)
        dlocs.append((loci, locj))
        jopts = [j for j in jopts if abs(j - locj) > 1]
    go = [row[:] for row in gi]

    def nb(r, c):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if (a or b) and 0 <= r + a < h and 0 <= c + b < w:
                    go[r + a][c + b] = linc

    for (li, lj) in dlocs:
        if li < min(locs):
            pairs = [min(locs)]
        elif li > max(locs):
            pairs = [max(locs)]
        else:
            pairs = [max(e for e in locs if e < li), min(e for e in locs if e > li)]
        r0 = min([li] + pairs)
        r1 = max([li] + pairs)
        for r in range(r0, r1 + 1):
            go[r][lj] = dotc
        for t in pairs:
            nb(t, lj)
        gi[li][lj] = dotc
    gi = np.array(gi)
    go = np.array(go)
    if mirror:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Lines are full, single-colour, non-background rows (or columns).
    def full_lines(G):
        return [r for r in range(G.shape[0])
                if G[r, 0] != bgc and len(set(G[r].tolist())) == 1]

    horiz = len(full_lines(I)) > 0
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(full_lines(T))
    linc = int(T[locs[0], 0])
    dotc = int([c for c in set(T.flatten().tolist()) if c not in (bgc, linc)][0])
    mp = (lambda r, c: (r, c)) if horiz else (lambda r, c: (c, r))

    G = T.copy()
    dots = sorted((r, c) for r in range(h) for c in range(w)
                  if T[r, c] == dotc and r not in locs)

    for (dr, dc) in dots:
        above = [e for e in locs if e < dr]
        below = [e for e in locs if e > dr]
        targets = []
        if above:
            targets.append(max(above))   # ray toward nearest line above
        if below:
            targets.append(min(below))   # ray toward nearest line below
        for t in targets:
            step = 1 if t > dr else -1
            # ray: straight segment from the dot to its contact point on the line
            ray = [(r, dc) for r in range(dr + step, t + step, step)]
            ops.append(dotc)
            sels.append(sel_of([mp(r, c) for r, c in ray]))
            for r, c in ray:
                G[r, c] = dotc
            # ring of line colour around the contact point (skip cells already linc)
            ring = [(r, c) for r in range(t - 1, t + 2) for c in range(dc - 1, dc + 2)
                    if 0 <= r < h and 0 <= c < w and (r, c) != (t, dc)
                    and G[r, c] != linc]
            if ring:
                ops.append(linc)
                sels.append(sel_of([mp(r, c) for r, c in ring]))
                for r, c in ring:
                    G[r, c] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
