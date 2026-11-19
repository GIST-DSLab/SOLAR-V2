"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------- 1. colors --
# The rule ("complete every concentric ring: corner colour on all 4 corners,
# edge colour along all 4 edges") is colour-agnostic, so only the background
# has to be pinned down for the episode.  The one genuine structural variant is
# which corner of the grid the seed pattern is anchored to (the generator's
# final rot90/rot180/rot270), so that is planned per instance.
ROTATIONS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        examples = [{"rot": r} for r in ROTATIONS]
        examples += [{"rot": random.choice(ROTATIONS)}
                     for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test uses a seen variant
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- 2. generate --
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if rot is None:
        rot = choice(ROTATIONS)

    hub = max(5, min(14, (max_h - 1) // 2))
    wub = max(5, min(14, (max_w - 1) // 2))
    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))
    h = h * 2 + 1
    w = w * 2 + 1

    remcols = remove(bgc, cols)
    ub = min(h, w) // 4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)

    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = sample(linsbase, nlines)
    colors = [choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)

    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2)
                   for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))

    rotf = {"identity": identity, "rot90": rot90,
            "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------ 3. derive_operations --
def derive_operations(I, O, examples=None):
    """Every instance is a set of concentric square-ish rings drawn on the odd
    cells.  The input shows, in ONE corner of the grid, a seed of each ring:
    the ring's corner pixel (its corner colour) and/or the two pixels next to
    that corner along the ring's two edges (its edge colour).  The output is the
    completed picture: each ring gets its corner colour on all four corners and
    its edge colour along all four edges.

    So we walk the rings from the outermost inwards and, for each, paint the
    four corners with one Color op and the whole four-edge outline with another.
    Only Color ops are used, so a role whose colour happens to be 0 is drawn
    just like any other colour (Copy/Paste would silently drop it).
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    # background: the canvas colour, shared by every instance of the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ein, _eout in examples:
            cnt.update(np.asarray(ein, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    ops, sels = [], []
    ub = min(H, W) // 4                      # number of concentric ring slots

    for k in range(ub):
        r1, r2 = 2 * k + 1, H - 2 - 2 * k    # the ring's two rows
        c1, c2 = 2 * k + 1, W - 2 - 2 * k    # the ring's two columns

        # --- the ring's four corners ---------------------------------------
        corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
        seen = [int(I[r, c]) for (r, c) in corners if int(I[r, c]) != bgc]
        if seen:
            ops.append(seen[0])              # Color<corner colour>
            sels.append(sel_of(corners))

        # --- the ring's four edges (odd cells strictly between the corners) --
        edge = []
        ca, cb = 2 * k + 3, W - 4 - 2 * k
        if ca <= cb:
            for c in range(ca, cb + 1, 2):
                edge.append((r1, c))
                edge.append((r2, c))
        ra, rb = 2 * k + 3, H - 4 - 2 * k
        if ra <= rb:
            for r in range(ra, rb + 1, 2):
                edge.append((r, c1))
                edge.append((r, c2))
        if edge:
            seen2 = [int(I[r, c]) for (r, c) in edge if int(I[r, c]) != bgc]
            if seen2:
                ops.append(seen2[0])         # Color<edge colour>
                sels.append(sel_of(sorted(set(edge))))

    ho, wo = O.shape
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])      # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
