"""
ARC Task: 508bd3b6 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colours ----
def sample_colors(num_examples=None) -> dict:
    """bgc / barc / linc are the three colours the generator draws with.
    The '3' that the beam is continued in is hardcoded by the task, so it is
    not sampled here.  The plan only fixes the bar's ORIENTATION per instance
    so that both a row-bar and a column-bar are demonstrated."""
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    barc = random.choice([c for c in cols if c != bgc])
    linc = random.choice([c for c in cols if c not in (bgc, barc)])

    n_ex = max(1, num_examples if num_examples else 3)
    if n_ex >= 2:
        ex = [{"transposed": False}, {"transposed": True}]
        ex += [{"transposed": random.random() < 0.5} for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [{"transposed": random.random() < 0.5} for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "barc": barc, "linc": linc, "instance_plan": plan}


# --------------------------------------------------------------- generator ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, barc, linc, transposed=None) -> dict:
    # both dimensions stay inside min(max_h, max_w): a transposing mirror may
    # swap them afterwards and the result still has to fit the canvas budget.
    lim = max(5, min(max_h, max_w))
    h = unifint(diff_lb, diff_ub, (5, lim))
    w = unifint(diff_lb, diff_ub, (h, lim))
    barh = unifint(diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(diff_lb, diff_ub, (2, h - barh))
    gi = canvas(bgc, (h, w))
    for j in range(barloci, barloci + barh):
        gi = fill(gi, barc, connect((j, 0), (j, w - 1)))
    dotlociinv = unifint(diff_lb, diff_ub, (0, barloci - 1))
    dotloci = min(max(0, barloci - 2 - dotlociinv), barloci - 1)
    ln1 = shoot((dotloci, 0), (1, 1))
    ofbgc = ofcolor(gi, bgc)
    ln1 = sfilter(ln1 & ofbgc, lambda ij: ij[0] < barloci)
    ln1 = order(ln1, first)
    ln2 = shoot(ln1[-1], (-1, 1))
    ln2 = sfilter(ln2 & ofbgc, lambda ij: ij[0] < barloci)
    ln2 = order(ln2, last)[1:]
    ln = ln1 + ln2
    k = len(ln1)
    lineleninv = unifint(diff_lb, diff_ub, (0, k - 2))
    linelen = k - lineleninv
    givenl = ln[:linelen]
    reml = ln[linelen:]
    gi = fill(gi, linc, givenl)
    go = fill(gi, 3, reml)

    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    par = (0, 1, 1, 0, 0, 1, 0, 1)          # 1 = swaps the two axes
    want = None if transposed is None else int(bool(transposed))
    while True:
        nmfs = random.choice((1, 2))
        idxs = random.sample(range(8), nmfs)
        if want is None or sum(par[i] for i in idxs) % 2 == want:
            break
    for i in idxs:
        gi = mfs[i](gi)
        go = mfs[i](go)
    return {'input': gi, 'output': go}


# ----------------------------------------------------------- trajectory ------
def derive_operations(I, O, examples=None):
    """Continue the ray until it reaches the wall, bounce it off, continue to
    the edge.  Only the input and the demonstrations are read; O is ignored.

    The ROUTE (where the drawing starts, and whether a straight run is laid
    down in one stroke or dotted in cell by cell) is read off the picture, so
    the same picture always yields the same route while different pictures
    are solved differently."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # colour the continuation is drawn in — read from the demonstrations
    fillc = 3
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            new = set(eo.flatten().tolist()) - set(ei.flatten().tolist())
            if len(new) == 1:
                fillc = int(new.pop())
                break

    colors = sorted(set(I.flatten().tolist()))

    # the wall: the only colour whose cells fill a solid rectangle spanning a
    # whole axis of the grid
    barc, box = None, None
    for c in colors:
        cells = np.argwhere(I == c)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        bh, bw = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if len(cells) == bh * bw and (bh == h or bw == w):
            barc, box = int(c), (int(r0), int(c0), int(r1), int(c1))
            break

    rest = [c for c in colors if c != barc]
    # the ray: the only remaining colour with no 4-adjacent twin (pure diagonal)
    linc = None
    for c in rest:
        s = {(int(r), int(cc)) for r, cc in np.argwhere(I == c)}
        if all((r + 1, cc) not in s and (r, cc + 1) not in s for (r, cc) in s):
            linc = int(c)
            break
    bgc = [c for c in rest if c != linc][0]

    r0, c0, r1, c1 = box
    vertical = (r1 - r0 + 1) == h              # wall of columns -> blocks sideways travel
    thickness = (c1 - c0 + 1) if vertical else (r1 - r0 + 1)
    bar = {(int(r), int(c)) for r, c in np.argwhere(I == barc)}
    line = [(int(r), int(c)) for r, c in np.argwhere(I == linc)]

    def dbar(p):
        return (max(r0 - p[0], 0) + max(p[0] - r1, 0) +
                max(c0 - p[1], 0) + max(p[1] - c1, 0))

    head = min(line, key=dbar)                 # ray tip aimed at the wall
    tail = max(line, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
    dr = (head[0] > tail[0]) - (head[0] < tail[0])
    dc = (head[1] > tail[1]) - (head[1] < tail[1])

    def march(start, d):
        out = []
        r, c = start
        while True:
            nr, nc = r + d[0], c + d[1]
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in bar:
                return out, (r, c)
            out.append((nr, nc))
            r, c = nr, nc

    # the stub still missing between the drawn ray and the wall
    seg_in, bounce = march(head, (dr, dc))
    # the reflected arm, from the wall out to the edge of the picture
    refl = (dr, -dc) if vertical else (-dr, dc)
    seg_out, _ = march(bounce, refl)

    seg_in = [p for p in seg_in if I[p] == bgc]
    seg_out = [p for p in seg_out if I[p] == bgc]

    # --- where to begin -------------------------------------------------
    # the eye starts at whichever end of the new path sits nearest the
    # top-left corner of the picture; if the bounce itself is that point,
    # the drawing grows outwards from the wall along both arms.
    first = seg_in[0] if seg_in else bounce
    last = seg_out[-1] if seg_out else bounce
    corner = lambda p: p[0] + p[1]
    if corner(bounce) < corner(first) and corner(bounce) < corner(last):
        runs = [list(reversed(seg_in)), list(seg_out)]          # out from the wall
    elif corner(first) <= corner(last):
        runs = [list(seg_in), list(seg_out)]                    # along the beam
    else:
        runs = [list(reversed(seg_out)), list(reversed(seg_in))]  # back from the exit
    runs = [r for r in runs if r]

    # --- how to lay a run down ------------------------------------------
    # one sweeping stroke over the whole straight run, or one cell at a time;
    # settled by the picture (wall thickness together with the two run lengths).
    chunky = ((len(seg_in) + len(seg_out) + thickness) % 2 == 0)

    for run in runs:
        if chunky:
            ops.append(fillc)
            sels.append(sel_of(run))                 # the whole straight run at once
        else:
            for cell in run:
                ops.append(fillc)
                sels.append(sel_of([cell]))          # follow the beam cell by cell

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])                # whole-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 508bd3b6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 508bd3b6"
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
                                f"for task 508bd3b6"
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
                    f"Failed to build a complete episode for task 508bd3b6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"508bd3b6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
