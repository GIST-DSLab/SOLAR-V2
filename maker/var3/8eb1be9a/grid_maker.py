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
import numpy as np
from collections import Counter
from random import choice, randint, sample

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the canvas colour is a real role here: the rule (repeat the object's
    # band vertically with period = band height) is purely structural, so the
    # object's own colours may differ freely from instance to instance.
    bgc = choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    cols = interval(0, 10, 1)
    hub = max(8, min(30, max_h))
    wub = max(4, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (8, hub))
    w = unifint(diff_lb, diff_ub, (4, wub))
    oh = unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (2, max(2, (oh * ow) // 3 * 2)))
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


def derive_operations(I, O, examples=None):
    """Rule: the object's band (its bbox, height oh) is stamped up and down the
    whole grid with period oh, keeping its columns.  Every copy is a real band,
    so the route is: grab the band once, then put it down at each band origin.

    The choices a solver actually faces here are the order of the bands and how
    the 0-coloured cells (which no Paste can carry) are dotted in.  Both are
    decided from the input alone:
      * sweep from whichever edge the object sits nearer to, outward from the
        object when it sits equidistant;
      * dot the 0s band-by-band when they make up most of the object (they are
        then the thing being drawn), otherwise collect them into one pass at
        the end (a few leftovers, finished in one go).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # background = the colour the canvas was filled with; the object covers at
    # most ~2/9 of the cells (oh <= h//3, ncells <= 2/3 of oh*ow), so the
    # background is the strict majority colour.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1

    zero_cells = [(r, c) for (r, c) in cells if I[r, c] == 0]      # Paste cannot carry these
    nz_cells = [(r, c) for (r, c) in cells if I[r, c] != 0]

    # ---- band origins (excluding the object's own band, already drawn) ----
    up_list, dn_list, out_list = [], [], []
    k = 1
    while True:
        up = r0 - k * oh
        dn = r0 + k * oh
        if up < 0 and dn >= h:
            break
        if 0 <= up < h:
            up_list.append(up)
            out_list.append(up)
        if 0 <= dn < h:
            dn_list.append(dn)
            out_list.append(dn)
        k += 1

    # the band cut by the top edge: only its lower rows are visible
    rem = r0 % oh
    tail_src = r0 + (oh - rem) if rem != 0 else None

    # ---- the two input-derived route choices ------------------------------
    d_top = r0
    d_bot = h - 1 - r1
    if d_top < d_bot:
        sweep = 'down'          # object hugs the top: start at the top edge, work down
    elif d_bot < d_top:
        sweep = 'up'            # object hugs the bottom: start at the bottom, work up
    else:
        sweep = 'out'           # dead centre: grow outward from the object itself

    per_band_zeros = len(zero_cells) * 2 >= len(cells)

    pending = []
    state = {'grab': None}

    def stamp(zs):
        if not zs:
            return
        if per_band_zeros:
            ops.append(0)
            sels.append(sel_of(sorted(set(zs))))
        else:
            pending.extend(zs)

    def put_band(o):
        land_nz = [(r - r0 + o, c) for (r, c) in nz_cells if 0 <= r - r0 + o < h]
        land_z = [(r - r0 + o, c) for (r, c) in zero_cells if 0 <= r - r0 + o < h]
        if land_nz:
            if state['grab'] != 'band':
                ops.append(28)                                  # CopyI of the band
                sels.append([r0, c0, oh - 1, ow - 1])           # exact rectangle: the whole band bbox
                state['grab'] = 'band'
            ops.append(30)
            sels.append([o, c0, 0, 0])                          # Paste the band at its origin
        stamp(land_z)

    def put_tail():
        if tail_src is None:
            return
        t_nz = [(r - tail_src, c) for (r, c) in nz_cells if r >= tail_src]
        t_z = [(r - tail_src, c) for (r, c) in zero_cells if r >= tail_src]
        if t_nz:
            ops.append(28)                                      # CopyI of the visible tail
            sels.append([tail_src, c0, r1 - tail_src, ow - 1])  # exact rectangle
            state['grab'] = 'tail'
            ops.append(30)
            sels.append([0, c0, 0, 0])                          # Paste flush with the top edge
        stamp(t_z)

    # ---- walk the bands in the chosen order -------------------------------
    if sweep == 'down':
        put_tail()
        for o in sorted(set(out_list)):
            put_band(o)
    elif sweep == 'up':
        for o in sorted(set(out_list), reverse=True):
            put_band(o)
        put_tail()
    else:
        for o in out_list:
            put_band(o)
        put_tail()

    if pending:
        ops.append(0)                                           # all the 0-coloured cells in one pass
        sels.append(sel_of(sorted(set(pending))))

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
