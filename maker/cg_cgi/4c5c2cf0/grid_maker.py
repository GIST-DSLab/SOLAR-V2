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
    cc = random.choice([c for c in cols if c != bgc])
    objc = random.choice([c for c in cols if c not in (bgc, cc)])
    return {"bgc": bgc, "cc": cc, "objc": objc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = lo + random.random() * (hi - lo)
    return int(min(b, max(a, round(v))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, cc=1, objc=2) -> dict:
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, max_h))
    w = _unifint(diff_lb, diff_ub, (10, max_w))
    oh = _unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = _unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = np.full((oh, ow), bgc, dtype=int)
    locc = (oh - 1, ow - 1)
    sg[locc] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def _shape(cs):
        rs = [r for r, _ in cs]
        ks = [k for _, k in cs]
        return (max(rs) - min(rs) + 1, max(ks) - min(ks) + 1)

    ncells = _unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = random.sample(reminds, ncells)
    while ncells == 5 and _shape(cells) == (3, 3):
        ncells = _unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = random.sample(reminds, ncells)
    for (i, j) in cells:
        sg[i, j] = objc
    G1 = sg
    G2 = np.fliplr(sg)
    G3 = np.flipud(sg)
    G4 = np.flipud(np.fliplr(sg))
    vbar = np.full((oh, 1), bgc, dtype=int)
    hbar = np.full((1, ow), bgc, dtype=int)
    cp = np.full((1, 1), cc, dtype=int)
    topg = np.hstack([G1, vbar, G2])
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
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_marker(I, bgc):
    """The X-shaped marker: a colour with exactly 5 cells forming a 3x3 X."""
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = [tuple(map(int, p)) for p in np.argwhere(I == col)]
        if len(cells) != 5:
            continue
        rs = [r for r, _ in cells]
        ks = [k for _, k in cells]
        ci, cj = (min(rs) + max(rs)) // 2, (min(ks) + max(ks)) // 2
        want = {(ci, cj), (ci - 1, cj - 1), (ci - 1, cj + 1),
                (ci + 1, cj - 1), (ci + 1, cj + 1)}
        if set(cells) == want:
            return col, ci, cj
    return None


def derive_operations(I, O, examples=None):
    # Reading (as in the golf entry): the grid is FOLDED twice through the X
    # marker's centre, each fold overlaying the grid with its own mirror.
    # Fold 1 mirrors the pattern's block across the centre row; fold 2 mirrors the
    # whole doubled strip (what fold 1 just produced, so CopyO from the current
    # grid) across the centre column.  Two folds, not three independent copies.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the colour the generator fills the canvas with (always majority)
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

    # the pattern's quadrant block, grown to the marker corner touching it
    r_lo = min(min(orows), ci + sr)
    r_hi = max(max(orows), ci + sr)
    c_lo = min(min(ocols), cj + sc)
    c_hi = max(max(ocols), cj + sc)

    G = I.copy()   # tracks the working grid, used only for no-op checks

    if cc != 0 and objc != 0:
        def fold(copy_op, src, dst, flip_op):
            sr0, sc0, sh, sw = src
            dr0, dc0 = dst
            # Copy/Paste carries the whole rectangle (bgc included) — whole rect on purpose
            ops.append(copy_op); sels.append([sr0, sc0, sh, sw])
            clip = G[sr0:sr0 + sh + 1, sc0:sc0 + sw + 1].copy()
            if bgc == 0:
                # transparent Paste would leave an X corner sitting uncovered in the
                # target rectangle and the flip would drag it off place: clear first
                region = G[dr0:dr0 + sh + 1, dc0:dc0 + sw + 1]
                if ((region != 0) & (clip == 0)).any():
                    ops.append(0); sels.append([dr0, dc0, sh, sw])
                    G[dr0:dr0 + sh + 1, dc0:dc0 + sw + 1] = 0
            ops.append(30); sels.append([dr0, dc0, 0, 0])
            region = G[dr0:dr0 + sh + 1, dc0:dc0 + sw + 1]
            region[clip != 0] = clip[clip != 0]
            flipped = np.flipud(region) if flip_op == 27 else np.fliplr(region)
            if not (flipped == region).all():
                # mirror the pasted rectangle in place
                ops.append(flip_op); sels.append([dr0, dc0, sh, sw])
                G[dr0:dr0 + sh + 1, dc0:dc0 + sw + 1] = flipped

        bh, bw = r_hi - r_lo, c_hi - c_lo
        # fold 1: input block across the centre row
        fold(28, (r_lo, c_lo, bh, bw), (2 * ci - r_hi, c_lo), 27)
        # fold 2: the doubled strip (both row-halves, current grid) across the centre column
        R_lo, R_hi = min(r_lo, 2 * ci - r_hi), max(r_hi, 2 * ci - r_lo)
        fold(29, (R_lo, c_lo, R_hi - R_lo, bw), (R_lo, 2 * cj - c_hi), 26)
    else:
        # 0 is real content here, so Copy/Paste would drop it: draw each fold's
        # mirror image of the pattern instead (the X maps onto itself).
        half = [(2 * ci - r, c) for r, c in obj if 0 <= 2 * ci - r < hi]
        if half:
            ops.append(objc); sels.append(sel_of(half))
        both = obj + half
        full = [(r, 2 * cj - c) for r, c in both if 0 <= 2 * cj - c < wi]
        if full:
            ops.append(objc); sels.append(sel_of(full))

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
