"""
ARC Task: 4c5c2cf0 (RE-ARC) — LLM-generated grid_maker
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
from random import choice, randint, sample
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cc=None, objc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    if bgc is None:
        bgc = choice(cols)
    if cc is None or cc == bgc:
        cc = choice([c for c in cols if c != bgc])
    if objc is None or objc in (bgc, cc):
        objc = choice([c for c in cols if c not in (bgc, cc)])
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 5 and shape(cells) == (3, 3):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _find_marker(I, bgc):
    # the X: a colour whose cells are exactly a centre plus its four diagonal neighbours
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = {(int(r), int(c)) for r, c in np.argwhere(I == col)}
        if len(cells) != 5:
            continue
        for (r, c) in cells:
            if cells == {(r, c), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)}:
                return col, r, c
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    marker = _find_marker(I, bgc)
    if marker is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    cc, ci, cj = marker
    obj = [(int(r), int(k)) for r, k in np.argwhere((I != bgc) & (I != cc))]
    if not obj:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    objc = int(I[obj[0]])

    orows = [r for r, _ in obj]
    ocols = [c for _, c in obj]
    sr = -1 if max(orows) < ci else 1
    sc = -1 if max(ocols) < cj else 1

    # quadrant block: pattern bbox grown to the marker corner next to the centre
    r_lo = min(min(orows), ci + sr); r_hi = max(max(orows), ci + sr)
    c_lo = min(min(ocols), cj + sc); c_hi = max(max(ocols), cj + sc)
    bh, bw = r_hi - r_lo, c_hi - c_lo

    # Reading: fold twice. First mirror the quadrant across the marker's centre
    # column (giving a half), then mirror that whole built half across the centre row.
    d1c = 2 * cj - c_hi                       # mirrored block's left column
    h_c_lo = min(c_lo, d1c)                   # half spans both blocks + centre column
    h_c_hi = max(c_hi, 2 * cj - c_lo)
    h_w = h_c_hi - h_c_lo
    d2r = 2 * ci - r_hi                       # mirrored half's top row

    if cc != 0 and objc != 0:
        # ---- fold 1: quadrant -> half (across centre column) ----
        ops.append(28); sels.append([r_lo, c_lo, bh, bw])           # CopyI block (full rect incl. background)
        if bgc == 0 and bw:
            # transparent paste would leave this side's marker corner to be swept by the flip
            ops.append(0); sels.append([r_lo, d1c, bh, bw])         # clear destination rect
        ops.append(30); sels.append([r_lo, d1c, 0, 0])              # Paste
        if bw:
            ops.append(26); sels.append([r_lo, d1c, bh, bw])        # FlipH whole dest rect
        # ---- fold 2: half -> whole (across centre row) ----
        ops.append(29); sels.append([r_lo, h_c_lo, bh, h_w])        # CopyO the built half (full rect)
        if bgc == 0 and bh:
            ops.append(0); sels.append([d2r, h_c_lo, bh, h_w])      # clear destination rect
        ops.append(30); sels.append([d2r, h_c_lo, 0, 0])            # Paste
        if bh:
            ops.append(27); sels.append([d2r, h_c_lo, bh, h_w])     # FlipV whole dest rect
    else:
        # 0 is real content -> Copy/Paste can't carry it; draw each fold's image instead.
        # fold 1: mirror the pattern across the centre column
        cells1 = [(r, 2 * cj - c) for r, c in obj if 0 <= 2 * cj - c < wi]
        ops.append(objc); sels.append(sel_of(cells1))
        # fold 2: mirror every pattern-coloured cell of the half across the centre row
        half = sorted(set(obj) | set(cells1))
        cells2 = [(2 * ci - r, c) for r, c in half if 0 <= 2 * ci - r < hi]
        ops.append(objc); sels.append(sel_of(cells2))

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
                        f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
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
                                f"for task 4c5c2cf0"
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
                    f"Failed to build a complete episode for task 4c5c2cf0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4c5c2cf0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
