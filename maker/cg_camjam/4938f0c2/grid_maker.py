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
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if cc is None:
        cc = random.choice([c for c in cols if c != bgc])
    if objc is None:
        objc = random.choice([c for c in cols if c not in (bgc, cc)])

    H = max(10, min(31, max_h + 1))
    W = max(10, min(31, max_w + 1))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))

    sg = [[bgc] * ow for _ in range(oh)]
    locc = (oh - 1, ow - 1)
    sg[locc[0]][locc[1]] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def shape_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    ncells = min(ncells, len(reminds))
    cells = random.sample(reminds, ncells)
    while ncells == 4 and shape_of(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        ncells = min(ncells, len(reminds))
        cells = random.sample(reminds, ncells)
    for (i, j) in cells:
        sg[i][j] = objc
    G1 = np.array(sg, dtype=int)
    G2 = np.fliplr(G1)
    G3 = np.flipud(G1)
    G4 = np.flipud(np.fliplr(G1))
    vbar = np.full((oh, 1), bgc, dtype=int)
    hbar = np.full((1, ow), bgc, dtype=int)
    cp = np.full((1, 1), cc, dtype=int)
    topg = np.hstack([G1, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])

    canv = np.full((h, w), bgc, dtype=int)
    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)
    go = canv.copy()
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = canv.copy()
    gi[loci:loci + oh, locj:locj + ow] = G1
    gi[go == cc] = cc

    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    pos = np.argwhere(gi == cc)
    rs = pos[:, 0]
    cs = pos[:, 1]
    ci = (rs.min() + rs.max()) // 2
    cj = (cs.min() + cs.max()) // 2
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
        # colours whose whole cell set is exactly one solid 2x2 block
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
    axis_r2 = 2 * br + 1      # row r <-> axis_r2 - r (marker's horizontal mid-line)
    axis_c2 = 2 * bc + 1      # col c <-> axis_c2 - c (marker's vertical mid-line)

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and int(I[r, c]) != mcol]
    if not obj:
        return [34], [full]
    objc = int(I[obj[0][0], obj[0][1]])

    G = I.copy()
    ops, sels = [], []

    # Doubling 1: reflect the picture across the marker's horizontal axis.
    picture = [(r, c) for r in range(hi) for c in range(wi) if int(G[r, c]) == objc
               and (r, c) in set(obj)]
    land = sorted({(axis_r2 - r, c) for (r, c) in picture
                   if 0 <= axis_r2 - r < hi})
    if any(int(G[r, c]) != objc for r, c in land):
        ops.append(objc); sels.append(sel_of(land))
        for r, c in land:
            G[r, c] = objc

    # Doubling 2: reflect the whole half-picture (original + its mirror) across
    # the marker's vertical axis -- the fourth quadrant comes from this, not separately.
    picture = [(r, c) for r in range(hi) for c in range(wi) if int(G[r, c]) == objc
               and int(G[r, c]) != mcol]
    if objc == bgc:
        picture = []
    land = sorted({(r, axis_c2 - c) for (r, c) in picture
                   if 0 <= axis_c2 - c < wi})
    if any(int(G[r, c]) != objc for r, c in land):
        ops.append(objc); sels.append(sel_of(land))
        for r, c in land:
            G[r, c] = objc

    ops.append(34)
    sels.append(full)   # full-grid rectangle: submit
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
