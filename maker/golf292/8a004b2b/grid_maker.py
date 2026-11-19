"""
ARC Task: 8a004b2b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cornc=None, ac1=None, ac2=None, objc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))
    if bgc is None:
        bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = unifint(diff_lb, diff_ub, (2, h // 5))
    ow = unifint(diff_lb, diff_ub, (2, w // 5))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    gi = [[bgc] * w for _ in range(h)]
    obj = {random.choice(sorted(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)

    def nbrs(p):
        return {(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}
    for _ in range(ncells - 1):
        cand = set()
        for p in obj:
            cand |= nbrs(p)
        obj.add(random.choice(sorted((bounds - obj) & cand)))
    mi = min(i for i, _ in obj)
    mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    lst = sorted(obj)
    fp1 = random.choice(lst)
    fp2 = random.choice([p for p in lst if p != fp1])
    cobj = {p: objc for p in obj}
    cobj[fp1] = ac1
    cobj[fp2] = ac2
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    if maxhscf < 1 or maxwscf < 1:
        raise ValueError("grid too small")
    hscf = unifint(diff_lb, diff_ub, (1, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (1, maxwscf))
    loci = random.randint(0, 2)
    locj = random.randint(0, 2)
    for (i, j), v in cobj.items():
        gi[i + loci][j + locj] = v
    inh = hscf * oh
    inw = wscf * ow
    sqh = unifint(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = unifint(diff_lb, diff_ub, (inw + 2, w))
    sqloci = random.randint(loci + oh, h - sqh)
    sqlocj = random.randint(0, w - sqw)
    crns = [(sqloci, sqlocj), (sqloci, sqlocj + sqw - 1),
            (sqloci + sqh - 1, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)]
    for (i, j) in crns:
        gi[i][j] = cornc
    gloci = random.randint(sqloci + 1, sqloci + sqh - 1 - inh)
    glocj = random.randint(sqlocj + 1, sqlocj + sqw - 1 - inw)
    go = [row[:] for row in gi]
    for i in range(oh):
        for j in range(ow):
            v = cobj.get((i, j), bgc)
            for a in range(hscf):
                for b in range(wscf):
                    go[gloci + i * hscf + a][glocj + j * wscf + b] = v
    go = [row[sqlocj:sqlocj + sqw] for row in go[sqloci:sqloci + sqh]]
    for (i, j), v in cobj.items():
        if v in (ac1, ac2):
            for a in range(hscf):
                for b in range(wscf):
                    gi[gloci + i * hscf + a][glocj + j * wscf + b] = v
    if random.choice((True, False)) and len(cobj) > 3:
        i, j = random.choice(sorted(p for p, v in cobj.items() if v == objc))
        for a in range(hscf):
            for b in range(wscf):
                gi[gloci + i * hscf + a][glocj + j * wscf + b] = objc
    gi = np.array(gi)
    go = np.array(go)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for k in random.sample(range(8), random.choice((1, 2))):
        gi = mfs[k](gi)
        go = mfs[k](go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    bycol = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                bycol.setdefault(v, []).append((r, c))

    # frame: a colour forming exactly the 4 corners of a box that encloses the marker blocks
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts})
        cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if set(pts) != {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}:
            continue
        a0, a1, b0, b1 = rs[0], rs[1], cs[0], cs[1]
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        inner = {int(I[r, c]) for r in range(a0 + 1, a1) for c in range(b0 + 1, b1)} - {bgc, col}
        if len(inner) < 2:
            continue
        area = (a1 - a0 + 1) * (b1 - b0 + 1)
        if best is None or area > best[0]:
            best = (area, col, a0, a1, b0, b1)
    _, cornc, r0, r1, c0, c1 = best

    # key object: every non-background cell outside the frame box
    key = {}
    for col, pts in bycol.items():
        for (r, c) in pts:
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                key[(r, c)] = col
    kr0 = min(r for r, _ in key)
    kc0 = min(c for _, c in key)

    # markers inside the frame -> anchor colours, scale factors, placement origin
    inside = {(r, c): int(I[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
              if I[r, c] != bgc}
    kcnt = Counter(key.values())
    icnt = Counter(inside.values())
    ratios = [icnt[c] / kcnt[c] for c in icnt if kcnt.get(c, 0) > 0]
    R = Counter(ratios).most_common(1)[0][0]
    anchors = {c for c in icnt if kcnt.get(c, 0) > 0 and icnt[c] / kcnt[c] == R}
    ia = [p for p, v in inside.items() if v in anchors]
    ka = [p for p, v in key.items() if v in anchors]
    ia_r0 = min(r for r, _ in ia); ia_r1 = max(r for r, _ in ia)
    ia_c0 = min(c for _, c in ia); ia_c1 = max(c for _, c in ia)
    ka_r0 = min(r for r, _ in ka); ka_r1 = max(r for r, _ in ka)
    ka_c0 = min(c for _, c in ka); ka_c1 = max(c for _, c in ka)
    hs = (ia_r1 - ia_r0 + 1) // (ka_r1 - ka_r0 + 1)
    ws = (ia_c1 - ia_c0 + 1) // (ka_c1 - ka_c0 + 1)
    org_r = ia_r0 - (ka_r0 - kr0) * hs
    org_c = ia_c0 - (ka_c0 - kc0) * ws
    keyloc = {(r - kr0, c - kc0): v for (r, c), v in key.items()}

    G = I.copy()  # working grid as it stands (built only from I and our own ops)
    ops, sels = [], []
    # Paint the key object scaled up: one full rectangle per maximal same-colour
    # rectangle of each key component (each selection is a full rectangle).
    seen = set()
    for cell in sorted(keyloc):
        if cell in seen:
            continue
        col = keyloc[cell]
        comp = {cell}
        stack = [cell]
        seen.add(cell)
        while stack:
            p = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + d[0], p[1] + d[1])
                if q in keyloc and q not in seen and keyloc[q] == col:
                    seen.add(q)
                    comp.add(q)
                    stack.append(q)
        uncovered = set(comp)
        while uncovered:
            i0, j0 = min(uncovered)
            j1 = j0
            while (i0, j1 + 1) in comp:
                j1 += 1
            i1 = i0
            while all((i1 + 1, j) in comp for j in range(j0, j1 + 1)):
                i1 += 1
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    uncovered.discard((i, j))
            br, bc = org_r + i0 * hs, org_c + j0 * ws
            bh, bw = (i1 - i0 + 1) * hs, (j1 - j0 + 1) * ws
            if np.all(G[br:br + bh, bc:bc + bw] == col):
                continue  # already present (marker block) -> painting would be a no-op
            G[br:br + bh, bc:bc + bw] = col
            ops.append(col)
            sels.append([br, bc, bh - 1, bw - 1])  # full rectangle of scaled blocks

    # crop the canvas to the frame (full rectangle)
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    ops.append(34); sels.append([0, 0, r1 - r0, c1 - c0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8a004b2b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8a004b2b"
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
                                f"for task 8a004b2b"
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
                    f"Failed to build a complete episode for task 8a004b2b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8a004b2b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
