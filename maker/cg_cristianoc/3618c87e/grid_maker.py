"""
ARC Task: 3618c87e (RE-ARC) — LLM-generated grid_maker
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

ROTATIONS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    bgc, linc, dotc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        ex = [{"rot": r} for r in ROTATIONS]
        ex += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None, **kwargs) -> dict:
    from dsl import canvas, connect, fill, interval, remove, identity, rot90, rot180, rot270
    from utils import unifint
    if rot is None:
        rot = random.choice(ROTATIONS)
    # the pre-rotation (h, w) becomes (w, h) when the grid is rotated by 90 or 270
    if rot in ("rot90", "rot270"):
        hmax, wmax = max_w, max_h
    else:
        hmax, wmax = max_h, max_w
    h = unifint(diff_lb, diff_ub, (4, min(30, hmax)))
    w = unifint(diff_lb, diff_ub, (4, min(30, wmax)))
    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch - 1, opts)
        opts = remove(ch + 1, opts)
    gi = fill(c, linc, ln)
    go = fill(c, linc, ln)
    for j in locs:
        hh = random.randint(1, h - 3)
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        go = fill(go, linc, lnx)
        gi = fill(gi, dotc, {(hh + 1, j)})
        go = fill(go, dotc, {(0, j)})
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading: each marker pixel sits one cell beyond the tip of its arm. The marker
    slides back along its own arm until it lands on the border line at the arm's
    base. It is the same pixel in a new place, so the route is a Move chain.
    For each marker: grab it, then keep moving the same grab (empty selections) toward
    the border. ARCLE restores the arm cells it passes over. Afterwards, Color(bgc)
    repairs the cell the marker left. If the marker colour is 0, Move cannot grab it,
    so that case falls back to clearing the tip and painting the base.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []
    NB = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def neigh(r, c):
        out = []
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc, dr, dc))
        return out

    # markers = single-cell components
    isolated = [(r, c) for r in range(h) for c in range(w)
                if all(I[rr, cc] != I[r, c] for rr, cc, _, _ in neigh(r, c))]
    if not isolated:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    dotc = Counter(int(I[r, c]) for r, c in isolated).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]

    # next to a marker, exactly one neighbour is the arm (line colour); the others are background
    ncnt = Counter()
    for r, c in dots:
        for rr, cc, _, _ in neigh(r, c):
            ncnt[int(I[rr, cc])] += 1
    linc = min(ncnt.items(), key=lambda kv: kv[1])[0]
    bgc = max(ncnt.items(), key=lambda kv: kv[1])[0]

    MOVE = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}
    arms = []
    for r, c in dots:
        dvec = None
        for rr, cc, dr, dc in neigh(r, c):
            if I[rr, cc] == linc:
                dvec = (dr, dc)
                break
        if dvec is None:
            continue
        dr, dc = dvec
        if dr == -1:
            base = (0, c)
        elif dr == 1:
            base = (h - 1, c)
        elif dc == -1:
            base = (r, 0)
        else:
            base = (r, w - 1)
        arms.append(((r, c), base, dvec))
    arms.sort(key=lambda t: (t[1][0], t[1][1]))

    for tip, base, (dr, dc) in arms:
        if dotc != 0:
            steps = abs(base[0] - tip[0]) + abs(base[1] - tip[1])
            op = MOVE[(dr, dc)]
            ops.append(op); sels.append(sel_of([tip]))       # grab the marker
            for _ in range(steps - 1):
                ops.append(op); sels.append(sel_of([]))      # continue the same slide
            if bgc != 0:
                ops.append(int(bgc)); sels.append(sel_of([tip]))  # repair the cell the marker left
        else:
            ops.append(int(bgc)); sels.append(sel_of([tip]))
            ops.append(int(dotc)); sels.append(sel_of([base]))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
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
                                f"for task 3618c87e"
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
                    f"Failed to build a complete episode for task 3618c87e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3618c87e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
