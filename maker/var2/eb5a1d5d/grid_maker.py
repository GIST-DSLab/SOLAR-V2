"""
ARC Task: eb5a1d5d (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colours
def sample_colors(num_examples=None) -> dict:
    """Ring colours, outside-in: box j gets palette[j].

    The rule is purely geometric, but one palette per episode keeps every
    instance of the episode on the same colour scheme.  Colour 0 is deliberately
    IN the pool: here it is an ordinary colour of the picture, and the route has
    to survive ARCLE reading 0 as "nothing there".
    """
    palette = list(range(10))
    random.shuffle(palette)
    return {"palette": palette}


# ---------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, palette=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if palette is None:
        palette = list(range(10))
        random.shuffle(palette)

    max_h = int(max_h)
    max_w = int(max_w)

    # d nested boxes -> a (2d-1)x(2d-1) picture; keep it small enough that at
    # least one row/column duplication still fits on the canvas.
    dcap = max(2, min(len(palette), min(max_h, max_w) // 2))
    d = unifint(diff_lb, diff_ub, (2, dcap))
    n = 2 * d - 1

    go = [[0] * n for _ in range(n)]
    for j in range(d):
        c = palette[j]
        for r in range(j, n - j):
            for cc in range(j, n - j):
                go[r][cc] = c

    vroom = max(0, max_h - n)
    hroom = max(0, max_w - n)
    nv = unifint(diff_lb, diff_ub, (0, vroom))
    nh = unifint(diff_lb, diff_ub, (0, hroom))
    if nv + nh == 0:                       # never emit an identity instance
        if vroom > 0:
            nv = 1
        elif hroom > 0:
            nh = 1

    gi = [row[:] for row in go]
    while nv > 0 or nh > 0:
        opts = []
        if nv > 0 and len(gi) < max_h:
            opts.append(0)
        if nh > 0 and len(gi[0]) < max_w:
            opts.append(1)
        if not opts:
            break
        if random.choice(opts) == 0:
            idx = random.randint(0, len(gi) - 1)
            gi = gi[:idx + 1] + [gi[idx][:]] + gi[idx + 1:]
            nv -= 1
        else:
            idx = random.randint(0, len(gi[0]) - 1)
            gi = [r[:idx + 1] + [r[idx]] + r[idx + 1:] for r in gi]
            nh -= 1

    return {'input': tuple(tuple(r) for r in gi),
            'output': tuple(tuple(r) for r in go)}


# ---------------------------------------------------------------- derivation
def derive_operations(I, O):
    """Squeeze every duplicated row and column out of the stretched nest of boxes.

    The picture is a set of concentric boxes whose rows/columns were copied in
    place; the answer is that picture with every box back to one cell thick.
    Each pair of identical neighbouring lines is consumed by sliding the block
    of picture on one side of it one cell onto the copy (a real Move: the
    picture slides across its own duplicate), and the canvas is finally cropped
    down to what is left.

    The route is read off the input, the way a person's choices would be, so two
    instances are solved two ways and the same instance always the same way:
      * the axis carrying more copies is cleared first,
      * the picture is pushed towards the edge it already sits nearer,
      * the sweep starts at the half where most of the copies are,
      * a duplicated outer frame four lines thick or more is worth one crop, so
        it is trimmed away up front instead of being slid over.
    O is never consulted -- every number below comes from I.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- runs of identical neighbouring rows / columns ----------------------
    def runs(lines):
        out = []
        for i, ln in enumerate(lines):
            if out and lines[out[-1][0]] == ln:
                out[-1].append(i)
            else:
                out.append([i])
        return out

    rrun = runs([tuple(r) for r in I.tolist()])
    crun = runs([tuple(c) for c in I.T.tolist()])
    ho, wo = len(rrun), len(crun)          # size of the squeezed picture

    # window still to be squeezed: one line of the outermost run is kept on each
    # side, everything outside it is a copy the final crop drops for free.
    r0, r1 = rrun[0][-1], rrun[-1][0]
    c0, c1 = crun[0][-1], crun[-1][0]

    # --- the choices, all read off the input --------------------------------
    dup_rows = [i for run in rrun for i in run[1:]]
    dup_cols = [i for run in crun for i in run[1:]]
    rows_first = len(dup_rows) >= len(dup_cols)          # messier axis first
    margin = r0 + (hi - 1 - r1) + c0 + (wi - 1 - c1)
    pre_crop = margin >= 4                               # thick frame: trim it
    pull_up = r0 <= hi - 1 - r1                          # towards the near edge
    pull_left = c0 <= wi - 1 - c1
    top_dups = sum(1 for i in dup_rows if 2 * i < hi)
    left_dups = sum(1 for i in dup_cols if 2 * i < wi)
    fwd_rows = top_dups >= len(dup_rows) - top_dups      # start where the work is
    fwd_cols = left_dups >= len(dup_cols) - left_dups

    ops, sels = [], []

    # --- 1. optionally trim the duplicated outer frame in one crop ----------
    if pre_crop and (r0, c0, r1, c1) != (0, 0, hi - 1, wi - 1):
        # a whole rectangle, picture and all -> bbox selection is exactly it
        ops.append(33)
        sels.append([r0, c0, r1 - r0, c1 - c0])
        G = I[r0:r1 + 1, c0:c1 + 1].copy()
        gh, gw = G.shape
        rlo, rhi, clo, chi = 0, gh - 1, 0, gw - 1
    else:
        G = I.copy()
        gh, gw = hi, wi
        rlo, rhi, clo, chi = r0, r1, c0, c1

    # --- 2. slide the picture over every duplicated line --------------------
    def squeeze(axis, lo, high, olo, ohi, pull_low, forward):
        """Consume every duplicated line of `axis` inside the window [lo, high].

        `pull_low` slides the block on the far side up/left onto the copy,
        otherwise the block on the near side is pushed down/right onto it.
        Returns the shrunken window.
        """
        A = G if axis == 0 else G.T          # .T is a view: writes reach G

        def same(i):
            return np.array_equal(A[i, olo:ohi + 1], A[i + 1, olo:ohi + 1])

        def sel(a, b):                       # a full rectangle -> bbox selection
            return [a, olo, b - a, ohi - olo] if axis == 0 \
                else [olo, a, ohi - olo, b - a]

        def slide(a, b, step):
            """Move the block of lines [a, b] one cell, ARCLE-style."""
            ops.append((20 if step < 0 else 21) if axis == 0
                       else (23 if step < 0 else 22))
            sels.append(sel(a, b))
            block = A[a:b + 1, olo:ohi + 1].copy()
            A[a:b + 1, olo:ohi + 1] = 0                  # grabbed -> background 0
            dest = A[a + step:b + 1 + step, olo:ohi + 1]
            np.copyto(dest, block, where=block > 0)      # 0 cells do not travel
            # a 0 of the picture landing on a cell the grab did not clear would
            # keep the old colour: put those down explicitly.  The only such
            # cells are on the line slid over, and that line is a true copy of
            # the line arriving, so there are normally none and nothing is emitted.
            miss = [(int(r), int(c))
                    for r, c in zip(*np.nonzero((block == 0) & (dest != 0)))]
            if miss:
                if axis == 0:
                    cells = [(a + step + r, olo + c) for r, c in miss]
                else:
                    cells = [(olo + c, a + step + r) for r, c in miss]
                ops.append(0)
                sels.append(sel_of(cells))
                dest[block == 0] = 0

        if forward:
            i = lo
            while i < high:
                if same(i):
                    if pull_low:
                        slide(i + 1, high, -1)
                        high -= 1
                    else:
                        slide(lo, i, +1)
                        lo += 1
                        i += 1
                else:
                    i += 1
        else:
            i = high - 1
            while lo <= i < high:
                if same(i):
                    if pull_low:
                        slide(i + 1, high, -1)
                        high -= 1
                        i -= 1
                    else:
                        slide(lo, i, +1)
                        lo += 1
                else:
                    i -= 1
        return lo, high

    if rows_first:
        rlo, rhi = squeeze(0, rlo, rhi, clo, chi, pull_up, fwd_rows)
        clo, chi = squeeze(1, clo, chi, rlo, rhi, pull_left, fwd_cols)
    else:
        clo, chi = squeeze(1, clo, chi, rlo, rhi, pull_left, fwd_cols)
        rlo, rhi = squeeze(0, rlo, rhi, clo, chi, pull_up, fwd_rows)

    # --- 3. crop the canvas onto the squeezed picture -----------------------
    if (rlo, clo) != (0, 0) or (gh, gw) != (ho, wo):
        ops.append(33)
        sels.append([rlo, clo, ho - 1, wo - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task eb5a1d5d"
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
                                f"for task eb5a1d5d"
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
                    f"Failed to build a complete episode for task eb5a1d5d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"eb5a1d5d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
