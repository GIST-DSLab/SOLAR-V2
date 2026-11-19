"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # the generator only ever samples one colour at random: the canvas colour,
    # drawn from the colours that are not 1 / 2 / 3 / 7 (those are hardcoded roles)
    cols = [c for c in range(10) if c not in (1, 2, 3, 7)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    rdi = randint(1, h - 2)
    rdj = randint(1, w - 2)
    rd = (rdi, rdj)
    reminds = inds - ({rd} | neighbors(rd))
    reminds = sfilter(reminds, lambda ij: 1 <= ij[0] <= h - 2 and 1 <= ij[1] <= w - 2)
    bd = choice(totuple(reminds))
    bdi, bdj = bd
    go = fill(go, 2, {rd})
    go = fill(go, 1, {bd})
    ngd = unifint(diff_lb, diff_ub, (1, 8))
    gd = sample(totuple(neighbors(rd)), ngd)
    nod = unifint(diff_lb, diff_ub, (1, 8))
    od = sample(totuple(neighbors(bd)), nod)
    go = fill(go, 3, gd)
    go = fill(go, 7, od)
    gdmapper = {d: (3, position({rd}, {d})) for d in gd}
    odmapper = {d: (7, position({bd}, {d})) for d in od}
    mpr = {**gdmapper, **odmapper}
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, ub))
    gi = tuple(e for e in go)
    fullinds = asindices(gi)
    for k in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = add(loc, direc)
            if (ii, jj) in fullinds and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if len(options) == 0:
            break
        loc, col, direc = choice(options)
        del mpr[loc]
        newloc = add(loc, direc)
        mpr[newloc] = (col, direc)
        gi = fill(gi, bgc, {loc})
        gi = fill(gi, col, {newloc})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Every 7-dot slides radially back until it touches the 1; every 3-dot slides
    radially back until it touches the 2.  The rule is fixed, the ROUTE is not:
    which family is tidied first, where the sweep around a marker starts and which
    way it turns, whether a long journey is taken before a short one, how a diagonal
    run is walked, and whether the emptied cells are swept back to the canvas colour
    dot-by-dot or in one pass — all read off the input grid, so one grid always
    yields one route and different grids yield different ones."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # canvas colour: the generator draws it from the colours outside {1,2,3,7}
    cnt = Counter(I.flatten().tolist())
    cands = [c for c in cnt if c not in (1, 2, 3, 7)]
    bgc = max(cands, key=lambda c: cnt[c]) if cands else 0

    sgn = lambda v: (v > 0) - (v < 0)
    p1 = np.argwhere(I == 1)
    p2 = np.argwhere(I == 2)
    c1 = (int(p1[0][0]), int(p1[0][1])) if len(p1) else None   # marker of the 7s
    c2 = (int(p2[0][0]), int(p2[0][1])) if len(p2) else None   # marker of the 3s

    # ---- the rule: each dot sits at marker + k*u, its home is marker + u -------
    dots = []
    for col, ctr in ((7, c1), (3, c2)):
        if ctr is None:
            continue
        for cell in np.argwhere(I == col):
            r, c = int(cell[0]), int(cell[1])
            u = (sgn(r - ctr[0]), sgn(c - ctr[1]))
            dots.append({"src": (r, c), "dst": (ctr[0] + u[0], ctr[1] + u[1]),
                         "col": col, "anchor": ctr, "u": u})

    # the finished picture, read from I and the rule alone (never from O)
    target = I.copy()
    target[(I == 3) | (I == 7)] = bgc
    for d in dots:
        if d["col"] == 7:
            target[d["dst"]] = 7
    for d in dots:
        if d["col"] == 3:
            target[d["dst"]] = 3
    final_cells = {d["dst"] for d in dots}

    moving = [d for d in dots if d["dst"] != d["src"]]
    n7 = sum(1 for d in dots if d["col"] == 7)
    n3 = len(dots) - n7

    # ---- route choices, each read off the input ------------------------------
    # 1. tidy the busier family first; if both are equal, the marker nearer the
    #    top-left of the grid gets attention first
    if n7 != n3:
        first_col = 7 if n7 > n3 else 3
    elif c1 is not None and c2 is not None:
        first_col = 7 if c1 <= c2 else 3
    else:
        first_col = 7
    # 2. which way the sweep around a marker turns, and whether the far dots or
    #    the near ones are fetched first
    cw = (bgc % 2 == 0)
    far_first = cw
    # 3. sweep by angle, or simply longest journey first
    by_distance = ((hi + wi) % 2 == 1)
    # 4. how a diagonal run is walked
    step_mode = ("zigzag", "vfirst", "hfirst")[(hi + wi + len(dots)) % 3]
    # 5. sweeping the emptied cells: after each dot, after each family, or once
    nmov = len(moving)
    repair_mode = "each" if nmov <= 3 else ("group" if nmov <= 6 else "end")

    CW8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def turn_index(u):
        i = CW8.index(u)
        return i if cw else (-i) % 8

    start_idx = {}
    if c1 is not None and c2 is not None:
        start_idx[7] = turn_index((sgn(c2[0] - c1[0]), sgn(c2[1] - c1[1])))
        start_idx[3] = turn_index((sgn(c1[0] - c2[0]), sgn(c1[1] - c2[1])))

    def pref_key(d):
        fam = 0 if d["col"] == first_col else 1
        ang = (turn_index(d["u"]) - start_idx.get(d["col"], 0)) % 8
        dist = max(abs(d["dst"][0] - d["src"][0]), abs(d["dst"][1] - d["src"][1]))
        rank = -dist if far_first else dist
        if by_distance:
            return (fam, rank, ang)
        return (fam, ang, rank)

    # ---- order the slides: never land on a dot that has not moved yet, and let
    #      a 7 reach a shared home before the 3 that also claims it ------------
    pref = sorted(moving, key=pref_key)
    order = []
    while pref:
        occupied = {d["src"] for d in pref}
        pick = None
        for d in pref:
            if d["dst"] in (occupied - {d["src"]}):
                continue
            if d["col"] == 3 and any(e["col"] == 7 and e["dst"] == d["dst"]
                                     for e in pref if e is not d):
                continue
            pick = d
            break
        if pick is None:
            pick = pref[0]
        pref.remove(pick)
        order.append(pick)

    def plan_steps(vr, vc):
        vop = 20 if vr < 0 else 21
        hop = 23 if vc < 0 else 22
        nv, nh = abs(vr), abs(vc)
        if step_mode == "vfirst":
            return [vop] * nv + [hop] * nh
        if step_mode == "hfirst":
            return [hop] * nh + [vop] * nv
        out = []
        for i in range(max(nv, nh)):
            if i < nv:
                out.append(vop)
            if i < nh:
                out.append(hop)
        return out

    DELTA = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}

    ops, sels = [], []
    state = I.copy()          # mirror of what ARCLE shows after each op
    vacated = []              # footprints still waiting to be swept back

    def flush():
        if vacated:
            keep = [c for c in vacated if state[c] == 0]
            if bgc != 0 and keep:
                ops.append(bgc)
                sels.append(sel_of(keep))
                for c in keep:
                    state[c] = bgc
            del vacated[:]

    prev_col = None
    for d in order:
        if repair_mode == "group" and prev_col is not None and d["col"] != prev_col:
            flush()
        prev_col = d["col"]
        (sr, sc), (tr, tc), col = d["src"], d["dst"], d["col"]
        if state[sr, sc] != col:
            # this dot was covered before its turn came: put it in its home directly
            if state[tr, tc] != col:
                ops.append(col)
                sels.append(sel_of([(tr, tc)]))
                state[tr, tc] = col
            continue
        steps = plan_steps(tr - sr, tc - sc)
        bg = None
        cur = (sr, sc)
        for k, mop in enumerate(steps):
            ops.append(mop)
            if k == 0:
                sels.append(sel_of([(sr, sc)]))   # this grabs the dot
                bg = state.copy()
                bg[sr, sc] = 0
            else:
                sels.append(sel_of([]))           # empty -> same dot glides on,
            dr, dc = DELTA[mop]                   # so its trail is restored for us
            cur = (cur[0] + dr, cur[1] + dc)
            state[:] = bg
            state[cur] = col
        if repair_mode == "each":
            if bgc != 0 and state[sr, sc] == 0:
                ops.append(bgc)
                sels.append(sel_of([(sr, sc)]))
                state[sr, sc] = bgc
        elif (sr, sc) not in final_cells:
            vacated.append((sr, sc))
    flush()

    # anything the rule still asks for that the slides could not deliver
    # (two families claiming one home cell); normally empty
    left = {}
    for r in range(hi):
        for c in range(wi):
            if state[r, c] != target[r, c]:
                left.setdefault(int(target[r, c]), []).append((r, c))
    for col in sorted(left):
        ops.append(col)
        sels.append(sel_of(left[col]))
        for cell in left[col]:
            state[cell] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
