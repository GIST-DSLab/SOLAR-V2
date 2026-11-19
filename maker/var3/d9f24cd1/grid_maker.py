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


# ---------------------------------------------------------------- colours ---
ROTS = [0, 1, 2, 3]          # np.rot90 quarter-turns -> seeds on each of 4 edges


def sample_colors(num_examples=None) -> dict:
    """bgc / linc (the ray colour, sitting on one outer edge) / dotc (interior
    blockers) are all sampled by the original generator, so all three are fixed
    once per episode.  The quarter-turn is a discrete structural variant (it
    decides which edge launches the rays and which way the jog turns), so it is
    planned per instance and every orientation is shown when there is room."""
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    dotc = random.choice([c for c in cols if c != bgc and c != linc])

    n_ex = num_examples if num_examples else 3
    variants = [{"rot": k} for k in ROTS]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown

    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


# --------------------------------------------------------------- generator ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    def unifint(bounds):
        a, b = bounds
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTS)

    # the quarter-turn may swap the axes -> respect max_h / max_w after rotating
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    lim_h = max(5, min(30, lim_h))
    lim_w = max(5, min(30, lim_w))
    h = unifint((5, lim_h))
    w = unifint((5, lim_w))

    maxnloc = max(1, (w - 2) // 2)

    # --- dots first (pairwise separated, strictly interior) ------------------
    ndots = unifint((1, maxnloc))
    locopts = list(range(1, w - 1))
    dotlocs = []
    for _ in range(ndots):
        if not locopts:
            break
        loc = random.choice(locopts)
        for x in (loc - 1, loc, loc + 1):
            if x in locopts:
                locopts.remove(x)
        dotlocs.append(loc)

    # --- launch lanes; force at least one blocked lane so the jog is visible --
    nlins = unifint((1, maxnloc))
    locopts = list(range(1, w - 1))
    locs = []
    if dotlocs:
        forced = random.choice(dotlocs)
        locs.append(forced)
        for x in (forced - 1, forced, forced + 1):
            if x in locopts:
                locopts.remove(x)
    while len(locs) < nlins and locopts:
        loc = random.choice(locopts)
        for x in (loc - 1, loc, loc + 1):
            if x in locopts:
                locopts.remove(x)
        locs.append(loc)

    gi = np.full((h, w), bgc, dtype=int)
    for l in locs:
        gi[h - 1, l] = linc
    dotrow = {}
    for l in dotlocs:
        jj = random.randint(1, h - 2)
        gi[jj, l] = dotc
        dotrow[l] = jj

    go = gi.copy()
    for l in locs:
        if l in dotrow:
            jj = dotrow[l]
            go[jj + 1:h, l] = linc            # run up to one cell short of the dot
            go[0:jj + 2, l + 1] = linc        # step sideways, continue to the far edge
        else:
            go[0:h, l] = linc

    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


# ------------------------------------------------------------------ solver ---
def derive_operations(I, O):
    """Rule (measured from I only):
       * bgc = majority colour.
       * linc = the non-bgc colour on the outer frame; all its cells lie on ONE
         edge (never a corner) -> that is the launch edge.
       * growth direction g = perpendicular, pointing inwards.  jog direction j
         = g turned 90 deg clockwise (up->right, right->down, down->left,
         left->up).
       * dotc = the remaining non-bgc colour, only ever interior.
       Each seed shoots a linc ray along g.  Hitting a dot, it stops one cell
       short, steps once along j into the neighbouring lane and runs on to the
       far edge.  Otherwise it runs the whole way.

       Route freedom the rule really leaves (all read off I, never off O):
         - which end of the launch edge to sweep from: begin at the corner the
           seeds crowd toward;
         - whether to clear all the unobstructed lanes first and come back for
           the bent ones, or take the lanes strictly in sweep order: the open
           lanes go first when they outnumber the bent ones;
         - whether a bent ray is one stroke or two: an early bend is drawn as a
           single gesture, a late one as run-then-turn.
       No background is laid down first: every dot must survive into O, so no
       cell here is ever painted twice."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- launch edge / seed cells: non-bgc cells on the outer frame ----------
    border = [(r, c) for r in range(h) for c in range(w)
              if r == 0 or r == h - 1 or c == 0 or c == w - 1]
    lin_cells = [(r, c) for (r, c) in border if int(I[r, c]) != bgc]
    if not lin_cells:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    linc = int(I[lin_cells[0][0], lin_cells[0][1]])

    # --- dot colour: the other non-bgc colour, found strictly inside ---------
    dotc = None
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(I[r, c])
            if v != bgc and v != linc:
                dotc = v
                break
        if dotc is not None:
            break

    # --- orientation (seeds never occupy a corner -> unambiguous) ------------
    if all(r == 0 for r, c in lin_cells):
        g, j = (1, 0), (0, -1)          # seeds on top edge
    elif all(r == h - 1 for r, c in lin_cells):
        g, j = (-1, 0), (0, 1)          # seeds on bottom edge
    elif all(c == 0 for r, c in lin_cells):
        g, j = (0, 1), (1, 0)           # seeds on left edge
    else:
        g, j = (0, -1), (-1, 0)         # seeds on right edge
    vertical_growth = (g[1] == 0)
    perp = h if vertical_growth else w              # how far a ray can travel
    edge_len = w if vertical_growth else h          # length of the launch edge

    # --- one path per seed, measured from I ----------------------------------
    rays = []
    for (sr, sc) in sorted(lin_cells):
        lane = []
        r, c = sr + g[0], sc + g[1]
        while 0 <= r < h and 0 <= c < w:
            lane.append((r, c))
            r += g[0]; c += g[1]

        didx = None
        if dotc is not None:
            for i, (rr, cc) in enumerate(lane):
                if int(I[rr, cc]) == dotc:
                    didx = i
                    break

        if didx is None:
            main, jog = lane, []
        else:
            main = lane[:didx]                       # stop one short of the dot
            last = main[-1] if main else (sr, sc)
            jog = []
            r, c = last[0] + j[0], last[1] + j[1]    # sideways into the next lane
            while 0 <= r < h and 0 <= c < w:
                jog.append((r, c))
                r += g[0]; c += g[1]

        t = sc if vertical_growth else sr            # position along the launch edge
        rays.append({"t": t, "main": main, "jog": jog})

    if not rays:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # --- choice 1: sweep from whichever corner the seeds crowd toward --------
    ts = [ray["t"] for ray in rays]
    ascending = min(ts) <= (edge_len - 1 - max(ts))
    rays.sort(key=lambda ray: ray["t"], reverse=not ascending)

    # --- choice 2: open lanes first, or strictly in sweep order --------------
    clear = [ray for ray in rays if not ray["jog"]]
    bent = [ray for ray in rays if ray["jog"]]
    if clear and bent and len(clear) >= len(bent):
        order = clear + bent
    else:
        order = rays

    # --- draw: one lane / one bent path at a time ----------------------------
    for ray in order:
        main, jog = ray["main"], ray["jog"]
        if not jog:
            if main:
                ops.append(linc); sels.append(sel_of(main))
            continue
        # choice 3: an early bend reads as a single stroke, a late one as two
        single = (2 * len(main) <= perp)
        if single or not main:
            ops.append(linc); sels.append(sel_of(main + jog))
        else:
            ops.append(linc); sels.append(sel_of(main))   # the straight run
            ops.append(linc); sels.append(sel_of(jog))    # its continuation past the dot

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
