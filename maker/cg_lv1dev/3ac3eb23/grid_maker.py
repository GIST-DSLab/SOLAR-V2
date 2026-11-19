"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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
    # Only the background is an episode-level colour role; each seed's colour is
    # drawn per seed and the rule copies whatever colour the seed already has.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    remcols = [c for c in range(10) if c != bgc]
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = set(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if not locopts:
            break
        locj = random.choice(sorted(locopts))
        locopts -= set(range(locj - 2, locj + 3))
        col = random.choice(remcols)
        gi[0, locj] = col
        go[0:h:2, locj] = col
        go[1:h:2, locj - 1] = col
        go[1:h:2, locj + 1] = col
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (the golf reading): the whole border line that carries the seeds is ONE
    unit, not one seed at a time. The line next to it is that border line shifted one
    step each way, which gives every seed its two wings at once. The two lines together
    (border line + wing line) then repeat with period 2 across the grid.
      (1) paint the wing line for all seeds together, with one Color op per colour;
      (2) copy the whole full-width 2-line band and paste it again and again across the grid;
      (3) if one line is left over at the far side, paste the border line there.
    Colour 0 does not show up in Copy/Paste, so any 0-coloured seed (possible only when
    bgc != 0) is continued depth by depth. O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour every demonstration input is painted on
    if examples:
        bgc = Counter(int(v) for ex in examples for row in ex[0] for v in row).most_common(1)[0][0]
        if bgc not in set(I.flatten().tolist()):
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # which border carries the seeds -> u = inward direction, v = along the border
    if all(r == 0 for r, _ in seeds):
        u, v, D, base = (1, 0), (0, 1), hi, (0, 0)
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D, base = (-1, 0), (0, 1), hi, (hi - 1, 0)
    elif all(c == 0 for _, c in seeds):
        u, v, D, base = (0, 1), (1, 0), wi, (0, 0)
    else:
        u, v, D, base = (0, -1), (1, 0), wi, (0, wi - 1)
    L = wi if v == (0, 1) else hi       # length of one line along the border

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    def lines_rect(d0, d1):
        # full-length rectangle covering depths d0..d1 -> [r, c, h-1, w-1]
        cells = [(base[0] + u[0] * d, base[1] + u[1] * d) for d in (d0, d1)]
        if v == (0, 1):
            r0 = min(p[0] for p in cells); r1 = max(p[0] for p in cells)
            return [r0, 0, r1 - r0, L - 1]
        c0 = min(p[1] for p in cells); c1 = max(p[1] for p in cells)
        return [0, c0, L - 1, c1 - c0]

    ops, sels = [], []

    # (1) the wing line: every seed spreads one step sideways into depth 1
    bycol = {}
    for s in seeds:
        bycol.setdefault(int(I[s]), []).extend([cell(s, 1, -1), cell(s, 1, 1)])
    for col in sorted(bycol):
        ops.append(col); sels.append(sel_of(sorted(bycol[col])))

    visible = any(int(I[s]) != 0 for s in seeds)   # Copy/Paste can carry something
    P = D // 2
    if visible and P > 1:
        # (2) the 2-line band is the period; the full rectangle of both lines is meant
        ops.append(29); sels.append(lines_rect(0, 1))
        for k in range(1, P):
            r, c, _, _ = lines_rect(2 * k, 2 * k + 1)
            ops.append(30); sels.append([r, c, 0, 0])
    if visible and D % 2 == 1:
        # (3) one even-depth line left at the far border: paste the border line there
        ops.append(29); sels.append(lines_rect(0, 0))   # whole border line (full rectangle)
        r, c, _, _ = lines_rect(D - 1, D - 1)
        ops.append(30); sels.append([r, c, 0, 0])

    # colour-0 seeds are skipped by Copy/Paste: continue their fan one depth at a time
    for s in seeds:
        if int(I[s]) == 0 and bgc != 0:
            for d in range(2, D):
                ts = (0,) if d % 2 == 0 else (-1, 1)
                ops.append(0); sels.append(sel_of([cell(s, d, t) for t in ts]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
