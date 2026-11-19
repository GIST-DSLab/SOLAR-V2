"""
ARC Task: d406998b (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colours ----
def sample_colors(num_examples=None) -> dict:
    # 3 is the marker colour the rule paints with, so it is never a picture colour.
    cols = [c for c in range(10) if c != 3]
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


# --------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None) -> dict:
    cols = remove(3, interval(0, 10, 1))
    if bgc is None or dotc is None:
        bgc, dotc = sample(cols, 2)

    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    itv = interval(0, h, 1)
    for j in range(w):
        nilocs = unifint(diff_lb, diff_ub, (1, h // 2 - 1 if h % 2 == 0 else h // 2))
        ilocs = sample(itv, nilocs)
        locs = {(ii, j) for ii in ilocs}
        gi = fill(gi, dotc, locs)
        go = fill(go, dotc if (j - w) % 2 == 0 else 3, locs)
    return {"input": gi, "output": go}


# ------------------------------------------------------------ derivation -----
def derive_operations(I, O, examples=None):
    """
    Rule (read off the examples, measured on I alone):
      every other column, counting from the RIGHT edge (the rightmost column is
      always included), has its dots repainted with the marker colour 3.
      Untouched columns and all background keep their colour.

    Everything below only ever looks at I, at the demonstrations, and at the rule.
    The route (which end to start from, whether to treat each vertical streak as
    its own object, whether to lay the marker across a whole column and draw the
    background back over it) is decided from measurements of I, so one instance
    always gives one route, and different instances give different routes.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # --- colours ------------------------------------------------------------
    counts = Counter(I.flatten().tolist())
    bgc = counts.most_common(1)[0][0]          # dots never fill half a column
    others = [c for c in counts if c != bgc]
    dotc = others[0] if others else bgc

    # marker colour: the colour the demonstrations introduce (never read from O)
    fillc = 3
    if examples:
        seen_in, seen_out = set(), set()
        for ex_i, ex_o in examples:
            seen_in |= set(np.asarray(ex_i).flatten().tolist())
            seen_out |= set(np.asarray(ex_o).flatten().tolist())
        fresh = sorted(seen_out - seen_in)
        if len(fresh) == 1:
            fillc = int(fresh[0])

    # --- the columns the rule touches ---------------------------------------
    targets = [j for j in range(wi) if (wi - 1 - j) % 2 == 0]
    dots = {j: [(r, j) for r in range(hi) if I[r, j] == dotc] for j in targets}
    live = [j for j in targets if dots[j]]

    ops, sels = [], []
    if not live:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    def streaks(cells):
        """maximal vertical runs of dots inside one column -- the little bars a
        person sees as single things."""
        out = []
        for (r, c) in cells:                      # cells are already top-down
            if out and out[-1][-1][0] == r - 1:
                out[-1].append((r, c))
            else:
                out.append([(r, c)])
        return out

    n_dots = sum(len(dots[j]) for j in live)
    density = n_dots / float(hi * len(live))
    all_streaks = {j: streaks(dots[j]) for j in live}
    n_streaks = sum(len(all_streaks[j]) for j in live)
    longest = max(len(s) for j in live for s in all_streaks[j])

    # --- route choice 1: how to treat a column ------------------------------
    if density >= 0.35:
        mode = "cover"      # crowded columns: sweep the marker down the whole
                            # column, then draw the background back over it
    elif longest >= 3 and n_streaks <= 40:
        mode = "streaks"    # a few clear bars: do them one bar at a time
    else:
        mode = "column"     # sparse scatter: one stroke per column

    # --- route choice 2: which end to sweep from ----------------------------
    right_first = len(dots[live[-1]]) >= len(dots[live[0]])
    sweep = list(reversed(live)) if right_first else list(live)

    # --- route choice 3: a standout column gets dealt with first ------------
    if len(live) >= 3:
        avg = n_dots / float(len(live))
        best = max(live, key=lambda j: (len(dots[j]), -j))
        if best not in (live[0], live[-1]) and len(dots[best]) >= avg + 2:
            sweep = [best] + [j for j in sweep if j != best]

    # --- route choice 4: which way up to work inside a column ---------------
    top_half = sum(1 for j in live for (r, _) in dots[j] if r < hi // 2)
    downward = top_half >= (n_dots - top_half)

    # --- emit ---------------------------------------------------------------
    for j in sweep:
        if mode == "cover":
            # lay the marker over the whole column ...
            ops.append(int(fillc))
            sels.append(sel_of([(r, j) for r in range(hi)]))
            # ... then put the background back where it belongs.  This is an
            # explicit Color(bgc), so it also works when bgc happens to be 0.
            back = [(r, j) for r in range(hi) if I[r, j] != dotc]
            if back:
                ops.append(int(bgc))
                sels.append(sel_of(back))
        elif mode == "streaks":
            bars = all_streaks[j]
            if not downward:
                bars = list(reversed(bars))
            for bar in bars:
                ops.append(int(fillc))
                sels.append(sel_of(bar if downward else list(reversed(bar))))
        else:
            cells = dots[j] if downward else list(reversed(dots[j]))
            ops.append(int(fillc))
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task d406998b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d406998b"
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
                                f"for task d406998b"
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
                    f"Failed to build a complete episode for task d406998b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d406998b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
