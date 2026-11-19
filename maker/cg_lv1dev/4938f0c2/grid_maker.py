"""
ARC Task: 4938f0c2 (RE-ARC) — LLM-generated grid_maker
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
        d = random.uniform(lb, ub)
        return min(b, max(a, round(a + (b - a) * d)))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if cc is None:
        cc = random.choice([c for c in cols if c != bgc])
    if objc is None:
        objc = random.choice([c for c in cols if c not in (bgc, cc)])
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h + 1)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w + 1)))
    oh = unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))
    sg = [[bgc] * ow for _ in range(oh)]
    locc = (oh - 1, ow - 1)
    sg[locc[0]][locc[1]] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def shape_of(cs):
        rr = [r for r, _ in cs]; cc_ = [c for _, c in cs]
        return (max(rr) - min(rr) + 1, max(cc_) - min(cc_) + 1)

    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = random.sample(reminds, ncells)
    while ncells == 4 and shape_of(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = random.sample(reminds, ncells)
    for (i, j) in cells:
        sg[i][j] = objc
    sg = np.array(sg)
    G1, G2, G3, G4 = sg, np.fliplr(sg), np.flipud(sg), np.flipud(np.fliplr(sg))
    vbar = np.full((oh, 1), bgc)
    hbar = np.full((1, ow), bgc)
    cp = np.full((1, 1), cc)
    topg = np.hstack([G1, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])
    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc)
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = np.full((h, w), bgc)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    rs, cs_ = np.where(gi == cc)
    ci, cj = (min(rs) + max(rs)) // 2, (min(cs_) + max(cs_)) // 2
    gi = np.delete(np.delete(gi, ci, axis=0), cj, axis=1)
    go = np.delete(np.delete(go, ci, axis=0), cj, axis=1)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def bg_of(g):
        g = np.asarray(g, dtype=int)
        return int(Counter(g.flatten().tolist()).most_common(1)[0][0])

    def markers(g):
        """colours whose whole cell set is exactly a solid 2x2 block -> {colour: (top,left)}"""
        g = np.asarray(g, dtype=int)
        b = bg_of(g)
        bycol = {}
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v = int(g[r, c])
                if v != b:
                    bycol.setdefault(v, []).append((r, c))
        out = {}
        for col, cs in bycol.items():
            if len(cs) == 4:
                rr = sorted({r for r, _ in cs})
                cc_ = sorted({c for _, c in cs})
                if len(rr) == 2 and len(cc_) == 2 and rr[1] == rr[0] + 1 and cc_[1] == cc_[0] + 1:
                    out[col] = (rr[0], cc_[0])
        return out

    full = [0, 0, hi - 1, wi - 1]
    bgc = bg_of(I)
    cand = markers(I)
    if not cand:
        return [34], [full]
    if len(cand) == 1:
        mcol = next(iter(cand))
    else:
        votes = Counter()
        for ex in (examples or []):
            ei = ex.get("input") if isinstance(ex, dict) else ex[0]
            if ei is None:
                continue
            for k in markers(ei):
                votes[k] += 1
        pool = [c for c in cand if votes.get(c, 0) > 0]
        mcol = max(pool, key=lambda c: votes[c]) if pool else sorted(cand)[0]

    br, bc = cand[mcol]
    ar, ac = 2 * br + 1, 2 * bc + 1        # doubled mirror axes through the marker centre

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and int(I[r, c]) != mcol]
    if not obj:
        return [34], [full]
    objc = int(I[obj[0][0], obj[0][1]])

    # The picture lives in one quadrant around the marker. The unit is that whole
    # quadrant: the rectangle from the picture's far edge up to (and including)
    # the marker cell it touches.
    rows = [r for r, _ in obj]
    cols = [c for _, c in obj]
    if max(rows) <= br:
        r0, r1 = min(rows), br
    else:
        r0, r1 = br + 1, max(rows)
    if max(cols) <= bc:
        c0, c1 = min(cols), bc
    else:
        c0, c1 = bc + 1, max(cols)
    qh, qw = r1 - r0, c1 - c0
    mr0, mc0 = ar - r1, ac - c1            # mirrored rectangle top-left

    ops, sels = [], []
    g = I.copy()

    def emit(op, sel):
        ops.append(op); sels.append(sel)

    if objc == 0 or mcol == 0:
        # 0 cannot travel through the clipboard (a 0 picture would not be copied, and a
        # 0 marker inside the quadrant would be covered by pasted background):
        # paint each reflected copy of the picture instead.
        for image in ([(r, ac - c) for r, c in obj], [(ar - r, c) for r, c in obj],
                      [(ar - r, ac - c) for r, c in obj]):
            cells = sorted({(r, c) for r, c in image
                            if 0 <= r < hi and 0 <= c < wi and int(g[r, c]) != objc})
            if cells:
                emit(objc, sel_of(cells))
                for r, c in cells:
                    g[r, c] = objc
        emit(34, full)
        return ops, sels

    # copy the source quadrant once
    emit(28, [r0, c0, qh, qw])
    clip = I[r0:r1 + 1, c0:c1 + 1].copy()

    def arcle_flip(grid, cells, fop):
        """ARCLE object-mode flip of the selected cells within their bounding box."""
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        a, b, c_, d = min(rs), max(rs), min(cs), max(cs)
        mask = np.zeros((b - a + 1, d - c_ + 1), bool)
        for r, c in cells:
            mask[r - a, c - c_] = True
        objp = np.where(mask, grid[a:b + 1, c_:d + 1], 0)
        out = grid.copy()
        out[a:b + 1, c_:d + 1][mask] = 0
        f = np.fliplr(objp) if fop == 26 else np.flipud(objp)
        reg = out[a:b + 1, c_:d + 1]
        reg[f > 0] = f[f > 0]
        return out

    marker_cells = {(br, bc), (br, bc + 1), (br + 1, bc), (br + 1, bc + 1)}

    def paste_and_flip(tr, tc, flips):
        rect = [(r, c) for r in range(tr, tr + qh + 1) for c in range(tc, tc + qw + 1)]
        start = g.copy()
        reg = g[tr:tr + qh + 1, tc:tc + qw + 1]
        reg[clip != 0] = clip[clip != 0]
        if not np.array_equal(g, start):
            emit(30, [tr, tc, 0, 0])

        def flip_selection(grid):
            standing = [p for p in rect if p in marker_cells and int(grid[p]) == mcol]
            if standing:
                # the standing marker cell still shows (the copy wrote nothing there)
                # while the pasted copy carries its own marker elsewhere in the
                # rectangle: keep the standing one out of the flip, so the flip lays
                # the pasted marker onto it.
                cells = [p for p in rect if p not in standing]
                return cells, sel_of(cells)
            # nothing to protect: flip the full mirrored quadrant rectangle
            return rect, [tr, tc, qh, qw]

        cur = g.copy()
        states = [start, cur]
        steps = []
        for fop in flips:
            cells, sel = flip_selection(cur)
            nxt = arcle_flip(cur, cells, fop)
            if any(np.array_equal(nxt, s_) for s_ in states):
                continue
            steps.append((fop, sel)); states.append(nxt); cur = nxt
        for fop, sel in steps:
            emit(fop, sel)
        g[:, :] = cur

    paste_and_flip(r0, mc0, [26])          # left<->right reflection
    paste_and_flip(mr0, c0, [27])          # up<->down reflection
    paste_and_flip(mr0, mc0, [26, 27])     # point reflection through the marker
    emit(34, full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
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
                                f"for task 4938f0c2"
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
                    f"Failed to build a complete episode for task 4938f0c2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4938f0c2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
