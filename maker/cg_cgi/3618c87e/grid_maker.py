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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc, dotc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, dotc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    if bgc is None or linc is None or dotc is None:
        bgc, linc, dotc = random.sample(range(10), 3)

    rotf = random.choice((0, 1, 2, 3))
    # rot90/rot270 swap dimensions; keep the final grid within max_h x max_w
    if rotf in (1, 3):
        hmax, wmax = max_w, max_h
    else:
        hmax, wmax = max_h, max_w
    h = unifint(diff_lb, diff_ub, (4, min(30, hmax)))
    w = unifint(diff_lb, diff_ub, (4, min(30, wmax)))

    gi = np.full((h, w), bgc, dtype=int)
    gi[0, :] = linc
    go = gi.copy()

    nlocs = unifint(diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = list(range(w))
    for _ in range(nlocs):
        if not opts:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = [o for o in opts if o not in (ch - 1, ch, ch + 1)]

    for j in locs:
        hh = random.randint(1, h - 3)
        gi[0:hh + 1, j] = linc
        go[0:hh + 1, j] = linc
        gi[hh + 1, j] = dotc
        go[0, j] = dotc

    # rotf: 0 identity, 1 rot90 (CW), 2 rot180, 3 rot270
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rotf]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    How I read the rule: each marker is one object that slides along its own arm
    until it lands on the border line. The arm stays where it is. What actually
    moves is the marker, so I write each one as a chain of unit Moves.
    - First Move grabs the marker, and the following Moves use an empty selection.
      ARCLE restores the arm cells the marker passes over.
    - The marker's start cell (the arm tip) is left at 0. If bgc != 0, one
      Color(bgc) repaints it.
    - A 0-coloured marker cannot be grabbed (object buffers drop 0), so in that
      case it is placed with Color ops.
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

    # markers are single-cell components
    isolated = []
    for r in range(h):
        for c in range(w):
            v = I[r, c]
            if all(I[rr, cc] != v for rr, cc, _, _ in neigh(r, c)):
                isolated.append((r, c))
    if not isolated:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    dotc = Counter(int(I[r, c]) for r, c in isolated).most_common(1)[0][0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]

    # one neighbour of each marker is its arm (line colour); the rest are background
    ncnt = Counter()
    for r, c in dots:
        for rr, cc, _, _ in neigh(r, c):
            ncnt[int(I[rr, cc])] += 1
    linc = min(ncnt.items(), key=lambda kv: kv[1])[0]
    bgc = max(ncnt.items(), key=lambda kv: kv[1])[0]

    # per marker: direction along its arm and the number of steps to the border
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
            steps, op, base = r, 20, (0, c)
        elif dr == 1:
            steps, op, base = h - 1 - r, 21, (h - 1, c)
        elif dc == -1:
            steps, op, base = c, 23, (r, 0)
        else:
            steps, op, base = w - 1 - c, 22, (r, w - 1)
        arms.append(((r, c), base, steps, op))
    arms.sort(key=lambda t: (t[1][0], t[1][1]))

    for tip, base, steps, op in arms:
        if steps <= 0:
            continue
        if dotc != 0:
            # grab the marker once, then slide it along its arm onto the border line
            ops.append(op); sels.append(sel_of([tip]))
            for _ in range(steps - 1):
                ops.append(op); sels.append(sel_of([]))
            # the only cell left at 0 is the marker's original cell (the arm tip)
            if bgc != 0:
                ops.append(int(bgc)); sels.append(sel_of([tip]))
        else:
            # a 0-coloured marker cannot be grabbed by Move: clear the tip, paint the base
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
