"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colours ---
def sample_colors(num_examples=None) -> dict:
    """bgc is the canvas colour AND the colour the answer is padded with, so it
    must be identical across the whole episode.  The mark colours carry no rule
    (any non-bgc colour behaves the same), so they stay free.

    How many marks an instance carries is structural, though: an episode whose
    examples all show 9 marks never shows the padding, so a test with 3 marks
    would be unlearnable.  Plan the counts so padding-heavy, mixed and full
    instances are all demonstrated."""
    bgc = random.choice(list(range(10)))

    n_ex = num_examples if num_examples else 3
    variants = [{"numc_target": 2}, {"numc_target": 5}, {"numc_target": 9}]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [{"numc_target": random.randint(1, 9)}
                     for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, numc_target=None) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    if numc_target is None:
        numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    else:
        numc = max(1, min(int(numc_target), min(9, w)))
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ---
def derive_operations(I, O, examples=None):
    """Read the scattered marks left to right, write them into a 3x3 answer
    boustrophedon (row 0 left->right, row 1 right->left, row 2 left->right),
    pad the rest with the canvas colour, keep only that 3x3.

    Everything below is measured from I.  Three things a solver genuinely gets
    to choose are chosen from I as well, so two instances take different routes:
      * WHERE the answer is built  - the empty corner furthest from the clutter;
      * WHICH WAY the sweep runs   - from whichever side the marks start nearer;
      * HOW the reversed middle row is produced - laid straight and mirrored
        when it is full enough for the reversal to be the thing you notice,
        cell by cell otherwise;
      * and whether a background base is swept over the work patch or only the
        single mark in the way is rubbed out.
    O is never consulted."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # canvas colour: marks are at most one per column of an h>=3 grid, so the
    # colour the marks sit on is the majority colour by a wide margin
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # the marks, in reading order (left to right)
    marks = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) != bgc]
    marks.sort(key=lambda rc: (rc[1], rc[0]))
    colors = [int(I[r, c]) for r, c in marks][:9]
    n = len(colors)
    colors = colors + [bgc] * (9 - n)          # leftover slots are canvas

    # answer slots, in the order the marks are read
    slots = [(0, 0), (0, 1), (0, 2),
             (1, 2), (1, 1), (1, 0),
             (2, 0), (2, 1), (2, 2)]
    target = [[bgc] * 3 for _ in range(3)]
    for k, (dr, dc) in enumerate(slots):
        target[dr][dc] = colors[k]

    # ---- where to build the answer: an empty corner, as far from the marks
    #      as possible so nothing still to be read gets covered up
    if marks:
        cr = sum(r for r, _ in marks) / float(len(marks))
        cc = sum(c for _, c in marks) / float(len(marks))
    else:
        cr, cc = (hi - 1) / 2.0, (wi - 1) / 2.0

    def strays_in(r0, c0):
        return [(r, c) for (r, c) in marks
                if r0 <= r < r0 + 3 and c0 <= c < c0 + 3]

    corners, seen = [], set()
    for p in [(0, 0), (0, wi - 3), (hi - 3, 0), (hi - 3, wi - 3)]:
        if p not in seen:
            seen.add(p)
            corners.append(p)
    corners.sort(key=lambda p: -(abs(p[0] + 1 - cr) + abs(p[1] + 1 - cc)))

    patch = None
    for p in corners:
        if not strays_in(*p):
            patch = p
            break
    if patch is None:                     # no clear corner: take the emptiest,
        best = None                       # furthest patch anywhere
        for rr in range(hi - 2):
            for ccol in range(wi - 2):
                key = (len(strays_in(rr, ccol)),
                       -(abs(rr + 1 - cr) + abs(ccol + 1 - cc)))
                if best is None or key < best[0]:
                    best = (key, (rr, ccol))
        patch = best[1]
    r0, c0 = patch

    ops, sels = [], []
    region = [r0, c0, 2, 2]     # bbox == exactly the whole 3x3 work patch
    cur = [[int(I[r0 + dr, c0 + dc]) for dc in range(3)] for dr in range(3)]
    need_crop = (hi, wi) != (3, 3)

    def paint(dr, dc, col):
        if cur[dr][dc] == col:            # already holds it: nothing to do
            return
        ops.append(col)
        sels.append(sel_of([(r0 + dr, c0 + dc)]))
        cur[dr][dc] = col

    if cur != target:
        # ---- clear what is in the way (only marks that must not stay)
        wrong = [(r, c) for (r, c) in strays_in(r0, c0)
                 if int(I[r, c]) != target[r - r0][c - c0]]
        if len(wrong) >= 2:
            # enough clutter that sweeping the patch clean and drawing on top
            # of the fresh base is the natural move
            ops.append(bgc)
            sels.append(region)
            cur = [[bgc] * 3 for _ in range(3)]
        elif len(wrong) == 1:
            # a single mark in the way: just rub that one out
            ops.append(bgc)
            sels.append(sel_of(wrong))
            (r, c) = wrong[0]
            cur[r - r0][c - c0] = bgc

        # ---- which way the sweep runs: start from the side the marks start on
        forward = (not marks) or (marks[0][1] <= wi - 1 - marks[-1][1])

        mid = colors[3:6]
        # lay the middle row straight and mirror it only when that mirror is a
        # real, visible move and every cell of it can survive the flip (ARCLE
        # does not carry colour 0 through a flip)
        use_mirror = (n >= 5 and mid[0] != mid[2] and all(v != 0 for v in mid))

        if use_mirror:
            jobs = [("plain", 0, colors[0:3]),
                    ("mirror", 1, colors[3:6]),
                    ("plain", 2, colors[6:9])]
            if not forward:
                jobs = jobs[::-1]
            for kind, dr, vals in jobs:
                for j in ([0, 1, 2] if forward else [2, 1, 0]):
                    paint(dr, j, vals[j])
                if kind == "mirror":
                    ops.append(26)                  # the middle row runs back
                    sels.append([r0 + 1, c0, 0, 2])  # bbox == exactly that row
                    cur[1] = cur[1][::-1]
        else:
            for k in (range(9) if forward else range(8, -1, -1)):
                dr, dc = slots[k]
                paint(dr, dc, colors[k])

        # safety net (target is derived from I, never from O)
        for dr in range(3):
            for dc in range(3):
                paint(dr, dc, target[dr][dc])

    # ---- keep only the answer
    if need_crop:
        ops.append(33)
        sels.append(region)

    ops.append(34)
    sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
