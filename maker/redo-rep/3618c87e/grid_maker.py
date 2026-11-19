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
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of

ROTS = ("identity", "rot90", "rot180", "rot270")


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # dot colour must be non-zero: the marker is duplicated with CopyI/Paste,
    # and ARCLE's clipboard treats 0 as "nothing".
    dotc = random.choice([c for c in cols if c != 0])
    rest = [c for c in cols if c != dotc]
    bgc = random.choice(rest)
    linc = random.choice([c for c in rest if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(list(ROTS), n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    # rot90 / rot270 swap the final dimensions, so cap accordingly
    if rot in ("rot90", "rot270"):
        hcap, wcap = max_w, max_h
    else:
        hcap, wcap = max_h, max_w
    hcap = max(4, min(30, int(hcap)))
    wcap = max(4, min(30, int(wcap)))

    h = unifint(diff_lb, diff_ub, (4, hcap))
    w = unifint(diff_lb, diff_ub, (4, wcap))

    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))

    nlocs = unifint(diff_lb, diff_ub, (1, max(1, w // 2)))
    locs = []
    opts = tuple(range(w))
    for _ in range(nlocs):
        if len(opts) == 0:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = tuple(x for x in opts if x not in (ch - 1, ch, ch + 1))

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


def derive_operations(I, O):
    """
    Structure of I: one full border edge is a line (linc); perpendicular arms grow
    inward from that edge, and each arm carries a single marker pixel (dotc) one cell
    beyond its tip.
    Rule: the marker is replicated onto the base of its own arm (the cell on the border
    line) and cleared from the tip.
    Route: CopyI the marker once, then per arm: clear the tip, Paste the marker at the base.
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

    # 1. markers = single-cell (4-connected, univalued) components
    isolated = []
    for r in range(h):
        for c in range(w):
            v = I[r, c]
            if all(I[rr, cc] != v for rr, cc, _, _ in neigh(r, c)):
                isolated.append((r, c))
    if not isolated:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    dotc = Counter(int(I[r, c]) for r, c in isolated).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]

    # 2. around a marker exactly one neighbour is the arm (line colour), the rest is bgc
    ncnt = Counter()
    for r, c in dots:
        for rr, cc, _, _ in neigh(r, c):
            ncnt[int(I[rr, cc])] += 1
    linc = min(ncnt.items(), key=lambda kv: kv[1])[0]
    bgc = max(ncnt.items(), key=lambda kv: kv[1])[0]

    # 3. for each marker: arm direction -> base cell on the border line
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
        arms.append(((r, c), base))
    arms.sort(key=lambda t: (t[1][0], t[1][1]))

    if dotc != 0:
        # grab the marker from the input once; it is the same pixel for every arm
        ops.append(28)
        sels.append(sel_of([arms[0][0]]))
        for tip, base in arms:
            ops.append(int(bgc))          # clear the marker from the arm tip
            sels.append(sel_of([tip]))
            ops.append(30)                # stamp a copy of the marker at the arm base
            sels.append(sel_of([base]))
    else:
        # colour 0 is invisible to the clipboard: place it with Color ops instead
        for tip, base in arms:
            ops.append(int(bgc))
            sels.append(sel_of([tip]))
            ops.append(int(dotc))
            sels.append(sel_of([base]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
