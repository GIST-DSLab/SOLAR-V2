"""
ARC Task: 44f52bb0 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ncols = random.randint(2, 9)
    ccols = random.sample(rem, ncols)

    VARIANTS = [{"issymm": True}, {"issymm": False}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, issymm=None, **kwargs) -> dict:
    if issymm is None:
        issymm = choice((True, False))
    A = 1 if issymm else 7
    mh = max(3, min(30, int(max_h)))
    mw = max(3, min(30, int(max_w)))
    ccols = list(ccols)

    while True:
        h = unifint(diff_lb, diff_ub, (3, mh))
        w = unifint(diff_lb, diff_ub, (3, mw))
        ncols = unifint(diff_lb, diff_ub, (2, len(ccols)))
        cc = sample(ccols, ncols)

        gi = canvas(bgc, (h, w))
        numcells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = asindices(gi)
        while gi == hmirror(gi):
            cells = sample(totuple(inds), numcells)
            gi = canvas(bgc, (h, w))
            for ij in cells:
                a, b = ij
                col = choice(cc)
                gi = fill(gi, col, {ij})
                gi = fill(gi, col, {(a, w - 1 - b)})

        if not issymm:
            numpert = unifint(diff_lb, diff_ub, (1, h * (w // 2)))
            cands = asindices(canvas(-1, (h, w // 2)))
            locs = sample(totuple(cands), numpert)
            for a, b in locs:
                col = gi[a][b]
                newcol = choice(totuple(remove(col, insert(bgc, set(cc)))))
                gi = fill(gi, newcol, {(a, b)})

        go = canvas(A, (1, 1))
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        nmfs = choice((1, 2))
        for fn in sample(mfs, nmfs):
            gi = fn(gi)
            go = fn(go)

        # --- validity: verdict must be A, and a usable mirror-probe pair must exist ---
        gl = [list(row) for row in gi]
        hh, ww = len(gl), len(gl[0])
        vs = all(gl[r][c] == gl[r][ww - 1 - c] for r in range(hh) for c in range(ww))
        hs = all(gl[r][c] == gl[hh - 1 - r][c] for r in range(hh) for c in range(ww))
        verdict = 1 if (vs or hs) else 7
        if verdict != A:
            continue
        axes = ['H'] if vs else (['V'] if hs else ['H', 'V'])
        ok = False
        for ax in axes:
            for r in range(hh):
                for c in range(ww):
                    pr, pc = (r, ww - 1 - c) if ax == 'H' else (hh - 1 - r, c)
                    if (pr, pc) != (r, c) and gl[r][c] != A and gl[pr][pc] != A:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            continue

        return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: the grid is tested against its mirror image.  The trajectory PERFORMS that
    reflection: a probe cell is marked with the verdict colour, the whole grid is
    reflected across the tested axis (FlipH = vmirror, FlipV = hmirror), which carries
    the mark to the probe's mirror partner, and that partner cell is what gets cropped
    out as the 1x1 answer.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    A = int(O[0, 0])

    v_sym = np.array_equal(np.fliplr(I), I)   # vmirror symmetry
    h_sym = np.array_equal(np.flipud(I), I)   # hmirror symmetry

    # reflect across the axis the rule is satisfied by; if neither holds, test vmirror
    if v_sym:
        axes = ['H']
    elif h_sym:
        axes = ['V']
    else:
        axes = ['H', 'V']

    chosen = None
    for ax in axes:
        cands = []
        for r in range(h):
            for c in range(w):
                if ax == 'H':
                    pr, pc = r, w - 1 - c
                else:
                    pr, pc = h - 1 - r, c
                if (pr, pc) == (r, c):
                    continue                      # cell maps onto itself: reflection invisible there
                if I[r, c] == A or I[pr, pc] == A:
                    continue                      # probe and target must both start off-verdict
                cands.append((r, c, pr, pc))
        if not cands:
            continue
        if not v_sym and not h_sym:
            # prefer a pair that actually witnesses the broken symmetry
            witness = [t for t in cands if I[t[0], t[1]] != I[t[2], t[3]]]
            pick = witness[0] if witness else cands[0]
        else:
            pick = cands[0]
        chosen = (ax, pick)
        break

    ops, sels = [], []

    if chosen is None:
        # degenerate grid: no usable mirror pair — mark a cell and read it off directly
        rc = None
        for r in range(h):
            for c in range(w):
                if I[r, c] != A:
                    rc = (r, c)
                    break
            if rc:
                break
        r, c = rc
        ops.append(A);  sels.append(sel_of([(r, c)]))          # verdict mark
        ops.append(33); sels.append([r, c, 0, 0])              # crop to that single cell
        ops.append(34); sels.append([0, 0, 0, 0])
        return ops, sels

    ax, (r, c, pr, pc) = chosen

    # 1. mark the probe cell with the verdict colour
    ops.append(A)
    sels.append(sel_of([(r, c)]))

    # 2. perform the reflection on the WHOLE grid (full rectangle, background included)
    ops.append(26 if ax == 'H' else 27)
    sels.append([0, 0, h - 1, w - 1])

    # 3. the mark now sits at the probe's mirror partner — crop it out as the answer
    ops.append(33)
    sels.append([pr, pc, 0, 0])

    ops.append(34)
    sels.append([0, 0, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
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
                                f"for task 44f52bb0"
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
                    f"Failed to build a complete episode for task 44f52bb0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44f52bb0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
