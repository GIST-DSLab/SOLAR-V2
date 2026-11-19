"""
ARC Task: 8eb1be9a (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# 1. episode-level colors
#    The rule ("stamp the object band up and down with period oh") depends only
#    on the object's *position/shape*, never on its colors, so only the
#    background needs to be fixed across the episode.
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator (faithful to RE-ARC generate_8eb1be9a, 30 -> max_h/max_w)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)

    h_lo = min(8, max_h)
    w_lo = min(4, max_w)
    h = unifint(diff_lb, diff_ub, (h_lo, max(h_lo, max_h)))
    w = unifint(diff_lb, diff_ub, (w_lo, max(w_lo, max_w)))

    oh = unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
    ow = unifint(diff_lb, diff_ub, (2, max(2, w)))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (2, max(2, (oh * ow) // 3 * 2)))
    ncells = min(ncells, len(bounds))
    obj = normalize(frozenset(sample(totuple(bounds), ncells)))
    oh, ow = shape(obj)

    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = frozenset({(choice(ccols), ij) for ij in obj})

    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    obj = shift(obj, (loci, locj))

    c = canvas(bgc, (h, w))
    gi = paint(c, obj)
    go = paint(c, obj)
    for k in range(h // oh + 1):
        go = paint(go, shift(obj, (-oh * k, 0)))
        go = paint(go, shift(obj, (oh * k, 0)))

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. ARCLE operation derivation
#    Rule: the single foreground "band" (bbox of every non-background cell,
#    height oh) is stamped again and again with vertical period oh, upward and
#    downward, until the whole canvas is covered.  The stamp is transparent:
#    background cells inside the band's bbox never overwrite anything, and the
#    original band is left untouched.
#    Implementation: one CopyI of the band's rectangle, then one Paste per new
#    band, growing outward from the original.  Because Copy/Paste treat 0 as
#    "nothing", any object cell whose colour is literally 0 is stamped by an
#    explicit Color0 right after that band's Paste.
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = colour the canvas was filled with; the object covers at most
    # ~2/9 of the cells (oh <= h//3, ncells <= 2/3 of oh*ow), so it is the
    # strict majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    rows = [r for r, _ in cells]
    colsx = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(colsx), max(colsx)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    zero_cells = [(r, c) for (r, c) in cells if I[r, c] == 0]   # object cells Paste cannot carry
    has_nonzero = any(I[r, c] != 0 for (r, c) in cells)

    # ---- full bands, ordered outward from the original object -----------
    full = []
    k = 1
    while True:
        up = r0 - k * oh
        dn = r0 + k * oh
        added = False
        if 0 <= up < h:
            full.append(up)
            added = True
        if 0 <= dn < h:
            full.append(dn)
            added = True
        if not added and (up < 0 and dn >= h):
            break
        k += 1
        if k > h + 2:
            break

    # ---- grab the object band once (exact rectangle: background included,
    #      and Paste is transparent so bg never clobbers anything) ---------
    if has_nonzero and full:
        ops.append(28)
        sels.append([r0, c0, oh - 1, ow - 1])          # CopyI of the band's bbox

    for o in full:
        if has_nonzero:
            ops.append(30)
            sels.append([o, c0, 0, 0])                 # Paste band at its origin
        if zero_cells:
            z = [(r - r0 + o, c) for (r, c) in zero_cells if 0 <= r - r0 + o < h]
            if z:
                ops.append(0)                          # stamp the 0-coloured cells
                sels.append(sel_of(z))

    # ---- the band clipped by the top edge (only its lower rows are visible)
    rem = r0 % oh
    if rem != 0:
        off = oh - rem                                  # first visible object row index
        src_r = r0 + off                                # its row in I
        src_nonzero = any(I[r, c] != 0 for (r, c) in cells if r >= src_r)
        if src_nonzero:
            ops.append(28)
            sels.append([src_r, c0, (r1 - src_r), ow - 1])   # CopyI of the visible tail
            ops.append(30)
            sels.append([0, c0, 0, 0])                       # Paste flush with the top edge
        z = [(r - src_r, c) for (r, c) in zero_cells if r >= src_r]
        if z:
            ops.append(0)
            sels.append(sel_of(z))

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
                        f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8eb1be9a"
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
                                f"for task 8eb1be9a"
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
                    f"Failed to build a complete episode for task 8eb1be9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8eb1be9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
