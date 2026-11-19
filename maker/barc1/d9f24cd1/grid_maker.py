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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of np.rot90-style turns applied to the canonical (seeds on bottom) layout


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    dotc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + d * (b - a)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # after an odd rotation the canonical h becomes the width
    hcap = max_h if rot % 2 == 0 else max_w
    wcap = max_w if rot % 2 == 0 else max_h
    h = _unifint(diff_lb, diff_ub, (5, min(30, hcap)))
    w = _unifint(diff_lb, diff_ub, (5, min(30, wcap)))

    maxnloc = (w - 2) // 2
    locopts = list(range(1, w - 1))
    nlins = _unifint(diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for _ in range(nlins):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if x not in (loc - 1, loc, loc + 1)]
        locs.append(loc)

    ndots = _unifint(diff_lb, diff_ub, (1, maxnloc))
    locopts = list(range(1, w - 1))
    dotlocs = []
    for _ in range(ndots):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if x not in (loc - 1, loc, loc + 1)]
        dotlocs.append(loc)

    gi = np.full((h, w), bgc, dtype=int)
    for l in locs:
        gi[h - 1, l] = linc
    dotrows = []
    for l in dotlocs:
        jj = random.randint(1, h - 2)
        gi[jj, l] = dotc
        dotrows.append(jj)

    go = gi.copy()
    for l in locs:
        if l in dotlocs:
            jj = dotrows[dotlocs.index(l)]
            go[jj + 1:h, l] = linc
            go[0:jj + 2, l + 1] = linc
        else:
            go[:, l] = linc

    # rotf in the original: identity / rot90(CW) / rot180 / rot270(CW)
    k = {0: 0, 1: -1, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Read in the reference's terms: every seed on the launch edge starts a
    line that walks into the grid. While the next cell is clear it keeps
    going forward. When the next cell is an obstacle dot, it steps ONE cell
    sideways (turning clockwise on screen: up->right, right->down,
    down->left, left->up) and then carries on forward to the far edge.
    So each line is drawn as up to three pieces: the forward run up to the
    dot, the single sideways step, and the forward run after the step.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    border = [(r, c) for r in range(h) for c in range(w)
              if r == 0 or r == h - 1 or c == 0 or c == w - 1]
    seeds = [(r, c) for (r, c) in border if int(I[r, c]) != bgc]
    if not seeds:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    linc = int(I[seeds[0]])

    dotc = None
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v != linc:
                dotc = v
                break
        if dotc is not None:
            break

    if all(r == 0 for r, _ in seeds):
        g, j = (1, 0), (0, -1)
    elif all(r == h - 1 for r, _ in seeds):
        g, j = (-1, 0), (0, 1)
    elif all(c == 0 for _, c in seeds):
        g, j = (0, 1), (1, 0)
    else:
        g, j = (0, -1), (-1, 0)

    def inb(r, c):
        return 0 <= r < h and 0 <= c < w

    for (sr, sc) in sorted(seeds):
        # forward run: advance while the next cell is not an obstacle
        r, c = sr, sc
        run = []
        hit = False
        while inb(r + g[0], c + g[1]):
            nr, nc = r + g[0], c + g[1]
            if dotc is not None and int(I[nr, nc]) == dotc:
                hit = True
                break
            run.append((nr, nc))
            r, c = nr, nc
        if run:
            ops.append(linc); sels.append(sel_of(run))
        if not hit:
            continue
        # sideways step around the obstacle
        sr2, sc2 = r + j[0], c + j[1]
        if not inb(sr2, sc2):
            continue
        ops.append(linc); sels.append(sel_of([(sr2, sc2)]))
        # forward run after the step, to the far edge
        r, c = sr2, sc2
        run2 = []
        while inb(r + g[0], c + g[1]):
            r, c = r + g[0], c + g[1]
            run2.append((r, c))
        if run2:
            ops.append(linc); sels.append(sel_of(run2))

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
