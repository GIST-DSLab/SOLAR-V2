"""
ARC Task: cbded52d (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colours ----
def sample_colors(num_examples=None) -> dict:
    """Only two colour roles are structural: the lattice-line colour and the
    cell/background colour.  The marker colours carry no rule of their own
    (any colour behaves identically), so they stay free per instance."""
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


# --------------------------------------------------------------- generate ----
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int) -> dict:
    cols = interval(0, 10, 1)
    oh_cap = max(1, min(4, (max_h + 1) // 3 - 1))
    ow_cap = max(1, min(4, (max_w + 1) // 3 - 1))
    oh = unifint(diff_lb, diff_ub, (1, oh_cap))
    ow = unifint(diff_lb, diff_ub, (1, ow_cap))
    numh = unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (oh + 1))))
    numw = unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (ow + 1))))
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, max(1, (numh * numh) // 3))))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    for col in ccols:
        inds = ofcolor(go, bgc)
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        narms = randint(1, 4)
        armdirs = sample(totuple(dneighbors((0, 0))), narms)
        succ = 0
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, {endp})
            go = fill(go, col, set(arm[:aidx + 1]))
            succ += 1
        if succ > 0:
            gi = fill(gi, col, {loc})
            go = fill(go, col, {loc})
    return {'input': gi, 'output': go}


# ------------------------------------------------------- derive_operations ----
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # U, R, D, L (clockwise cycle)


def derive_operations(I, O):
    """
    Rule, read from I alone:
      I is a cell lattice — uniform separator lines of colour `linc` cut the grid
      into cells of colour `bgc`, vertical pitch `ph`, horizontal pitch `pw`.
      Every other colour marks one hub cell plus one endpoint cell per arm, each
      endpoint on the hub's row/column an integer number of pitches away.  Each
      arm is completed: every lattice cell from the hub out to its endpoint takes
      that colour.

    The route (which star first, which arm first, which way along an arm, one
    pixel at a time vs. the whole arm / the whole star in one stroke, Colour vs.
    FloodFill) is chosen from measurements of I — pitch, palette, marker counts,
    hub position — so a given puzzle always gets the same route, and different
    puzzles get different ones.  What is painted never changes.
    """
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # ---- lattice geometry: the uniform rows / columns are the separators ----
    hlines = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    vlines = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    if hlines:
        linc = int(I[hlines[0], 0])
    elif vlines:
        linc = int(I[0, vlines[0]])
    else:
        linc = -1
    ph = max(1, (hi + 1) // (len(hlines) + 1))
    pw = max(1, (wi + 1) // (len(vlines) + 1))

    # ---- cell colour = dominant non-line colour; the rest are markers ----
    cnt = {}
    for v in I.flatten().tolist():
        v = int(v)
        if v != linc:
            cnt[v] = cnt.get(v, 0) + 1
    if not cnt:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # whole grid bbox
        return ops, sels
    bgc = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    marker_cols = sorted(c for c in cnt if c != bgc)

    # ---- read every star: hub + one ray of lattice cells per arm ----
    stars = []
    total_markers = 0
    for col in marker_cols:
        pts = [(int(r), int(c)) for r, c in np.argwhere(I == col)]
        if len(pts) < 2:
            continue
        total_markers += len(pts)
        deg = {p: sum(1 for q in pts if q != p and (q[0] == p[0] or q[1] == p[1]))
               for p in pts}
        hub = max(pts, key=lambda p: (deg[p], -p[0], -p[1]))
        rays = {}
        for p in pts:
            if p == hub or (p[0] != hub[0] and p[1] != hub[1]):
                continue
            dr = 0 if p[0] == hub[0] else (1 if p[0] > hub[0] else -1)
            dc = 0 if p[1] == hub[1] else (1 if p[1] > hub[1] else -1)
            steps = (abs(p[0] - hub[0]) // ph) if dr else (abs(p[1] - hub[1]) // pw)
            cells = [(hub[0] + dr * k * ph, hub[1] + dc * k * pw)
                     for k in range(1, steps + 1)]
            if len(cells) > len(rays.get((dr, dc), [])):
                rays[(dr, dc)] = cells
        if rays:
            stars.append({"col": col, "hub": hub, "rays": rays,
                          "load": sum(len(v) for v in rays.values())})
    if not stars:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # whole grid bbox
        return ops, sels

    # ---- route choices, all measured off I ----
    k_order = (linc + bgc + len(hlines) + len(vlines)) % 4
    k_style = (ph + pw + len(stars) + total_markers) % 4
    tiny_cells = (ph == 2 and pw == 2)      # every lattice cell is a single pixel

    if k_order == 0:                       # lowest colour first
        stars.sort(key=lambda s: s["col"])
    elif k_order == 1:                      # highest colour first
        stars.sort(key=lambda s: -s["col"])
    elif k_order == 2:                      # hubs in reading order
        stars.sort(key=lambda s: (s["hub"], s["col"]))
    else:                                   # biggest star first
        stars.sort(key=lambda s: (-s["load"], s["hub"], s["col"]))

    G = I.copy()

    def paint(col, cells, single_ok):
        pend = [(r, c) for (r, c) in cells if G[r, c] != col]
        if not pend:
            return
        if single_ok and len(pend) == 1 and tiny_cells:
            r, c = pend[0]
            ops.append(10 + int(col))           # FloodFill: 1×1 cell = 1 pixel
            sels.append([r, c, 0, 0])           # exactly one seed cell
        else:
            ops.append(int(col))
            sels.append(sel_of(pend))
        for (r, c) in pend:
            G[r, c] = col

    for st in stars:
        col, hub, rays = st["col"], st["hub"], st["rays"]
        # arms taken in a turning order that starts where this hub sits
        start = ((hub[0] // ph) + (hub[1] // pw)) % 4
        turn = 1 if ((hub[0] // ph) + len(stars)) % 2 == 0 else -1
        order = [_DIRS[(start + turn * i) % 4] for i in range(4)]
        arms = [rays[d] for d in order if d in rays]

        if k_style == 3:
            # one stroke for the whole star
            whole = []
            for cells in arms:
                whole.extend(cells)
            paint(col, whole, False)
        elif k_style == 2:
            # one stroke per arm
            for cells in arms:
                paint(col, cells, True)
        elif k_style == 1:
            # walk each arm inward, from its far end back toward the hub
            for cells in arms:
                for cell in reversed(cells):
                    paint(col, [cell], True)
        else:
            # walk each arm outward from the hub
            for cells in arms:
                for cell in cells:
                    paint(col, [cell], True)

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # whole-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task cbded52d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cbded52d"
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
                                f"for task cbded52d"
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
                    f"Failed to build a complete episode for task cbded52d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cbded52d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
