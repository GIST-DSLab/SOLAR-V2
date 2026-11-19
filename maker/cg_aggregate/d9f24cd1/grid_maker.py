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


def _canvas(c, h, w):
    return [[c] * w for _ in range(h)]


ROTS = [0, 1, 2, 3]   # number of 90° CW turns applied after generating (launch-edge side)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    linc = random.choice(rem)
    rem = [c for c in rem if c != linc]
    dotc = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, int(round(lo + d * (hi - lo)))))

    # an odd number of quarter turns swaps height and width
    hb, wb = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = unif(5, min(30, hb))
    w = unif(5, min(30, wb))

    maxnloc = (w - 2) // 2
    locopts = list(range(1, w - 1))
    nlins = unif(1, maxnloc)
    locs = []
    for _ in range(nlins):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if x not in (loc - 1, loc, loc + 1)]
        locs.append(loc)
    ndots = unif(1, maxnloc)
    locopts = list(range(1, w - 1))
    dotlocs = []
    for _ in range(ndots):
        if not locopts:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if x not in (loc - 1, loc, loc + 1)]
        dotlocs.append(loc)

    gi = _canvas(bgc, h, w)
    for l in locs:
        gi[h - 1][l] = linc
    dotrows = []
    for l in dotlocs:
        jj = random.randint(1, h - 2)
        gi[jj][l] = dotc
        dotrows.append(jj)
    go = [row[:] for row in gi]
    for l in locs:
        if l in dotlocs:
            jj = dotrows[dotlocs.index(l)]
            for r in range(jj + 1, h):
                go[r][l] = linc
            for r in range(0, jj + 2):
                go[r][l + 1] = linc
        else:
            for r in range(h):
                go[r][l] = linc

    gi = np.rot90(np.array(gi), -rot)
    go = np.rot90(np.array(go), -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Wavefront reading: every line grows out of the launch edge together,
    one layer (row/column parallel to the edge) at a time.  Each step paints
    the next layer of every growing line in one op.  When a line's next cell
    would be a dot, that layer gets the sideways jog cell, and the line keeps
    growing in the neighbouring lane from there on.  The layer cells come only
    from I (seeds, dots, orientation)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    border = [(r, c) for r in range(h) for c in range(w)
              if r in (0, h - 1) or c in (0, w - 1)]
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
        layer = lambda r, c: r
    elif all(r == h - 1 for r, _ in seeds):
        g, j = (-1, 0), (0, 1)
        layer = lambda r, c: h - 1 - r
    elif all(c == 0 for _, c in seeds):
        g, j = (0, 1), (1, 0)
        layer = lambda r, c: c
    else:
        g, j = (0, -1), (-1, 0)
        layer = lambda r, c: w - 1 - c

    inb = lambda r, c: 0 <= r < h and 0 <= c < w
    by_layer = {}
    for (sr, sc) in sorted(seeds):
        cells = []
        r, c = sr, sc
        while True:
            nr, nc = r + g[0], c + g[1]
            if not inb(nr, nc):
                break
            if dotc is not None and int(I[nr, nc]) == dotc:
                # blocked: step sideways, then keep growing in that lane
                r, c = r + j[0], c + j[1]
                while inb(r, c):
                    cells.append((r, c))
                    r, c = r + g[0], c + g[1]
                break
            cells.append((nr, nc))
            r, c = nr, nc
        for cell in cells:
            by_layer.setdefault(layer(*cell), []).append(cell)

    for k in sorted(by_layer):
        ops.append(linc); sels.append(sel_of(sorted(by_layer[k])))

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
