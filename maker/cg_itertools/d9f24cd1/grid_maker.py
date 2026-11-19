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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    dotc = random.choice(rem)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, dotc=None, **kw) -> dict:
    from dsl import (interval, remove, canvas, fill, connect,
                     identity, rot90, rot180, rot270)
    from utils import unifint

    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = random.choice(cols)
    if linc is None:
        linc = random.choice([c for c in cols if c != bgc])
    if dotc is None:
        dotc = random.choice([c for c in cols if c not in (bgc, linc)])

    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    rotf = random.choice((identity, rot90, rot180, rot270))
    if rotf in (rot90, rot270):
        # keep rotated shape inside (max_h, max_w)
        h = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
        w = unifint(diff_lb, diff_ub, (5, max(5, max_h)))

    locopts = interval(1, w - 1, 1)
    maxnloc = (w - 2) // 2
    nlins = unifint(diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for k in range(nlins):
        if len(locopts) == 0:
            break
        loc = random.choice(locopts)
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
        loc = random.choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        dotlocs.append(loc)
    gi = canvas(bgc, (h, w))
    for l in locs:
        gi = fill(gi, linc, {(h - 1, l)})
    dotlocs2 = []
    for l in dotlocs:
        jj = random.randint(1, h - 2)
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
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading: every line is a growing front.  Each step, all active lines
    advance by one cell simultaneously along the growth direction g; a tip
    whose next cell is a dot instead steps sideways (jog direction j) and the
    growth continues from there.  One Color(linc) op per wave, selecting the
    new tip cells of all lines (derived from the current tips + dots in I)."""
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
    linc = int(I[seeds[0][0], seeds[0][1]])

    dotc = None
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v != linc:
                dotc = v
                break
        if dotc is not None:
            break

    if all(r == 0 for r, c in seeds):
        g, j = (1, 0), (0, -1)
    elif all(r == h - 1 for r, c in seeds):
        g, j = (-1, 0), (0, 1)
    elif all(c == 0 for r, c in seeds):
        g, j = (0, 1), (1, 0)
    else:
        g, j = (0, -1), (-1, 0)

    cur = I.copy()
    tips = sorted(seeds)
    inb = lambda r, c: 0 <= r < h and 0 <= c < w
    while tips:
        wave, nxt = [], []
        for (r, c) in tips:
            nr, nc = r + g[0], c + g[1]
            if not inb(nr, nc):
                continue                       # reached the far edge
            if dotc is not None and int(cur[nr, nc]) == dotc:
                nr, nc = r + j[0], c + j[1]    # blocked: jog sideways
                if not inb(nr, nc):
                    continue
            if int(cur[nr, nc]) != bgc:
                continue
            wave.append((nr, nc))
            nxt.append((nr, nc))
        if not wave:
            break
        ops.append(linc); sels.append(sel_of(wave))
        for (r, c) in wave:
            cur[r, c] = linc
        tips = nxt

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
