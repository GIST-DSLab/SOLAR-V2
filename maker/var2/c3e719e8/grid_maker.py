"""
ARC Task: c3e719e8 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. colours
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """Fix the whole palette for the episode.

    `mc` is the marker colour: the most common colour of the tile, whose cells
    say which blocks of the big canvas get a copy of the tile.  It is kept
    non-zero, because it is the colour the rule is read from and ARCLE treats 0
    as 'nothing there'.  The remaining detail colours DO include 0 — in this
    episode 0 is an ordinary colour of the picture, and the route has to cope
    with cells of colour 0 that no Copy/Paste can carry.
    """
    mc = random.choice(range(1, 10))
    others = [c for c in range(10) if c != mc and c != 0]
    random.shuffle(others)
    if random.random() < 0.5:                      # give 0 a real chance of being
        others.insert(0, 0)                        # one of the first detail colours
    else:
        others.insert(random.randrange(1, len(others) + 1), 0)
    return {"mc": mc, "palette": others}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             mc=None, palette=None) -> dict:
    if mc is None:
        mc = choice(tuple(range(1, 10)))
    if palette is None:
        rest = [c for c in range(10) if c != mc]
        palette = list(sample(tuple(rest), len(rest)))

    # output is (h*h, w*w) -> h, w bounded by the square root of the canvas caps
    hub = 5
    while hub * hub > max_h and hub > 2:
        hub -= 1
    wub = 5
    while wub * wub > max_w and wub > 2:
        wub -= 1

    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))
    gob = canvas(-1, (h ** 2, w ** 2))
    wg = canvas(-1, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mcc = sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    ocols = [c for c in palette if c != mc][:ncols]
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = sample(inds, unifint(diff_lb, diff_ub, (1, ub)))
        inds = difference(inds, locs)
        gi = fill(gi, ocol, locs)
    gi = replace(gi, -1, mc)
    o = asobject(gi)
    gob = replace(gob, -1, 0)
    go = paint(gob, mapply(lbind(shift, o),
                           apply(rbind(multiply, (h, w)), ofcolor(gi, mc))))
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    """The rule (read off the worked examples): the canvas becomes a hi x wi
    arrangement of blocks, each block the size of the tile I, and a block holds
    a copy of I exactly where I shows its most common colour.

    Nothing below looks at O: the finished grid, the block layout, the marker
    colour and every route decision are all functions of I (and of the rule the
    examples teach).  The ROUTE varies from instance to instance the way two
    people differ — stamp the tile with the clipboard, or paint the block's
    dominant colour down and draw the details on top; sweep the blocks in one
    of four directions; copy before or after enlarging the canvas; tidy the
    left-over top-left block first or last — but each variant performs exactly
    the same rule and contains no op the answer could be reached without.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ho, wo = hi * hi, wi * wi                      # output size, from I alone

    cnt = Counter(I.flatten().tolist())
    mc = cnt.most_common(1)[0][0]                  # marker / dominant colour
    colours = sorted(cnt)
    k = len(colours)

    targets = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == mc]
    n_t = len(targets)
    fr, fc = targets[0]                            # first marked cell, reading order

    # ---- route choices, all derived from the input picture -----------------
    sweep = (mc + fr + fc + n_t) % 4               # which way to work through blocks
    draw = (k <= 3) or (n_t <= 3)                  # few colours / few blocks -> draw it
    per_colour = ((mc + n_t) % 2 == 0)             # all of one colour at once?
    copy_first = (mc % 2 == 1)                     # grab the tile before enlarging?
    clear_late = ((hi + wi + mc) % 2 == 1)         # tidy the stale block last?
    by_count = ((hi * wi + mc) % 2 == 0)           # detail colours: boldest or lowest first

    def block(r, c):
        R, C = r * hi, c * wi
        return [(R + i, C + j) for i in range(hi) for j in range(wi)]

    def block_of(r, c, col):
        R, C = r * hi, c * wi
        return [(R + i, C + j) for i in range(hi) for j in range(wi)
                if I[i, j] == col]

    def ordered(blocks):
        if sweep == 0:
            return sorted(blocks, key=lambda t: (t[0], t[1]))    # rows, downwards
        if sweep == 1:
            return sorted(blocks, key=lambda t: (t[1], t[0]))    # columns, rightwards
        if sweep == 2:
            return sorted(blocks, key=lambda t: (-t[0], -t[1]))  # rows, upwards
        return sorted(blocks, key=lambda t: (-t[1], -t[0]))      # columns, leftwards

    tile_rect = [(r, c) for r in range(hi) for c in range(wi)]
    out_rect = [(r, c) for r in range(ho) for c in range(wo)]

    # the enlargement leaves the tile sitting in block (0,0); that block is
    # already finished if it is marked, and must be wiped if it is not.
    work = ordered([t for t in targets if t != (0, 0)])
    clear_needed = (0, 0) not in targets
    use_clipboard = (not draw) and len(work) > 0

    ops, sels = [], []

    # ---- take the tile / enlarge the canvas --------------------------------
    if use_clipboard and copy_first:
        ops.append(28); sels.append(sel_of(tile_rect))   # exactly the tile rectangle

    ops.append(33); sels.append(sel_of(out_rect))        # exactly the new canvas rectangle

    if use_clipboard and not copy_first:
        ops.append(28); sels.append(sel_of(tile_rect))   # exactly the tile rectangle

    def emit_clear():
        # exactly the top-left block rectangle
        ops.append(0); sels.append(sel_of(block(0, 0)))

    if clear_needed and not clear_late:
        emit_clear()

    # ---- fill the marked blocks --------------------------------------------
    if draw:
        others = [c for c in colours if c != mc]
        if by_count:
            others.sort(key=lambda c: (-cnt[c], c))
        else:
            others.sort()

        if per_colour:
            base = [p for t in work for p in block(*t)]
            based = bool(base) and mc != 0
            if based:
                ops.append(mc); sels.append(sel_of(base))
            for col in others:
                cells = [p for t in work for p in block_of(t[0], t[1], col)]
                if not cells:
                    continue
                # colour 0 travels with nothing in ARCLE: it only needs putting
                # down where a base was laid over it; on bare canvas it is there.
                if col == 0 and not based:
                    continue
                ops.append(col); sels.append(sel_of(cells))
        else:
            based = (mc != 0)
            for t in work:
                if based:
                    # exactly this block's rectangle: lay the dominant colour,
                    # then draw the tile's details over it
                    ops.append(mc); sels.append(sel_of(block(*t)))
                for col in others:
                    cells = block_of(t[0], t[1], col)
                    if not cells:
                        continue
                    if col == 0 and not based:
                        continue
                    ops.append(col); sels.append(sel_of(cells))
    else:
        # stamp the clipboard tile into each marked block.  Paste carries only
        # the tile's non-zero cells; its colour-0 cells land on the freshly
        # enlarged canvas, which is already 0 there, so nothing is emitted for
        # them (a Color0 there would change nothing).
        for t in work:
            ops.append(30); sels.append(sel_of([(t[0] * hi, t[1] * wi)]))

    if clear_needed and clear_late:
        emit_clear()

    ops.append(34); sels.append(sel_of(out_rect))
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
                        f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3e719e8"
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
                                f"for task c3e719e8"
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
                    f"Failed to build a complete episode for task c3e719e8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3e719e8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
