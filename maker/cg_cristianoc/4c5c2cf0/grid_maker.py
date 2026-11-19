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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cc=None, objc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if cc is None:
        cc = random.choice([c for c in cols if c != bgc])
    if objc is None:
        objc = random.choice([c for c in cols if c not in (bgc, cc)])
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = np.full((oh, ow), bgc, dtype=int)
    locc = (oh - 1, ow - 1)
    sg[locc] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]
    while True:
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = random.sample(reminds, ncells)
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        shp = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        if not (ncells == 5 and shp == (3, 3)):
            break
    for r, c in cells:
        sg[r, c] = objc
    G2 = sg[:, ::-1]; G3 = sg[::-1, :]; G4 = sg[::-1, ::-1]
    vbar = np.full((oh, 1), bgc, dtype=int)
    hbar = np.full((1, ow), bgc, dtype=int)
    cp = np.full((1, 1), cc, dtype=int)
    topg = np.hstack([sg, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])
    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_marker(I, bgc):
    """The X-shaped marker: a colour with exactly 5 cells laid out as an X."""
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = {(int(r), int(c)) for r, c in np.argwhere(I == col)}
        if len(cells) != 5:
            continue
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        ci = min(rs) + 1; cj = min(cs) + 1
        x = {(ci, cj), (ci - 1, cj - 1), (ci - 1, cj + 1), (ci + 1, cj - 1), (ci + 1, cj + 1)}
        if cells == x:
            return col, ci, cj
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # background: the canvas colour the generator paints first (always the majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    marker = _find_marker(I, bgc)
    obj = [] if marker is None else [(int(r), int(k)) for r, k in np.argwhere((I != bgc) & (I != marker[0]))]
    if marker is None or not obj:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    cc, ci, cj = marker
    objc = int(I[obj[0]])

    orows = [r for r, _ in obj]; ocols = [c for _, c in obj]
    sr = -1 if max(orows) < ci else 1           # which quadrant holds the pattern
    sc = -1 if max(ocols) < cj else 1

    # Quadrant block: the pattern grown to the marker corner touching the centre.
    r_lo = min(min(orows), ci + sr); r_hi = max(max(orows), ci + sr)
    c_lo = min(min(ocols), cj + sc); c_hi = max(max(ocols), cj + sc)
    bh, bw = r_hi - r_lo, c_hi - c_lo

    # Stage 1: mirror the quadrant (pattern + marker corner) across the centre row.
    d1r = 2 * ci - r_hi
    # Stage 2: the half built so far (both quadrants with the centre row between)
    # is itself one unit, mirrored across the centre column.
    H_lo = min(r_lo, d1r); H_hi = max(r_hi, 2 * ci - r_lo)
    hh = H_hi - H_lo
    d2c = 2 * cj - c_hi

    if cc != 0 and objc != 0:
        # Selections below are whole rectangles on purpose (block incl. background).
        ops.append(28); sels.append([r_lo, c_lo, bh, bw])           # CopyI the quadrant block
        if bgc == 0 and bh:
            # transparent paste would leave this side's marker cell to be dragged by the flip
            ops.append(0); sels.append([d1r, c_lo, bh, bw])          # clear landing block
        ops.append(30); sels.append([d1r, c_lo, 0, 0])              # Paste across centre row
        if bh:
            ops.append(27); sels.append([d1r, c_lo, bh, bw])         # FlipV in place
        ops.append(29); sels.append([H_lo, c_lo, hh, bw])           # CopyO the half just built
        if bgc == 0 and bw:
            ops.append(0); sels.append([H_lo, d2c, hh, bw])          # clear landing block
        ops.append(30); sels.append([H_lo, d2c, 0, 0])              # Paste across centre column
        if bw:
            ops.append(26); sels.append([H_lo, d2c, hh, bw])         # FlipH in place
    else:
        # 0 is real content here, so Copy/Paste cannot carry it: draw each mirror stage.
        # The X marker maps onto itself, so only the pattern needs drawing.
        half = [(2 * ci - r, c) for r, c in obj]
        half = [(r, c) for r, c in half if 0 <= r < hi]
        ops.append(objc); sels.append(sel_of(half))                 # stage 1: across centre row
        full = [(r, 2 * cj - c) for r, c in obj + half]
        full = [(r, c) for r, c in full if 0 <= c < wi]
        ops.append(objc); sels.append(sel_of(full))                 # stage 2: across centre column

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
