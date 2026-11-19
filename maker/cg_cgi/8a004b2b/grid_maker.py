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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cornc=None, ac1=None,
             ac2=None, objc=None, **kw) -> dict:
    if None in (bgc, cornc, ac1, ac2, objc):
        bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    H = max(10, min(30, max_h)); W = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, H))
    w = _unifint(diff_lb, diff_ub, (10, W))
    oh = _unifint(diff_lb, diff_ub, (2, h // 5))
    ow = _unifint(diff_lb, diff_ub, (2, w // 5))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    gi = [[bgc] * w for _ in range(h)]
    obj = {random.choice(sorted(bounds))}
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)

    def nbrs(p):
        return {(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}
    for _ in range(ncells - 1):
        cand = set()
        for p in obj:
            cand |= nbrs(p)
        cand = sorted((bounds - obj) & cand)
        obj.add(random.choice(cand))
    mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
    obj = {(r - mr, c - mc) for r, c in obj}
    oh = max(r for r, _ in obj) + 1; ow = max(c for _, c in obj) + 1
    objl = sorted(obj)
    fp1 = random.choice(objl)
    fp2 = random.choice([p for p in objl if p != fp1])
    cobj = {p: objc for p in obj}
    cobj[fp1] = ac1; cobj[fp2] = ac2
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    hscf = _unifint(diff_lb, diff_ub, (1, maxhscf))
    wscf = _unifint(diff_lb, diff_ub, (1, maxwscf))
    loci = random.randint(0, 2); locj = random.randint(0, 2)
    for (r, c), v in cobj.items():
        gi[r + loci][c + locj] = v
    inh = hscf * oh; inw = wscf * ow
    sqh = _unifint(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = _unifint(diff_lb, diff_ub, (inw + 2, w))
    sqloci = random.randint(loci + oh, h - sqh)
    sqlocj = random.randint(0, w - sqw)
    for r in (sqloci, sqloci + sqh - 1):
        for c in (sqlocj, sqlocj + sqw - 1):
            gi[r][c] = cornc
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
    for p in (fp1, fp2):
        for a in range(hscf):
            for b in range(wscf):
                gi[gloci + p[0] * hscf + a][glocj + p[1] * wscf + b] = cobj[p]
    if random.choice((True, False)) and len(obj) > 3:
        idx = random.choice([p for p in objl if cobj[p] == objc])
        for a in range(hscf):
            for b in range(wscf):
                gi[gloci + idx[0] * hscf + a][glocj + idx[1] * wscf + b] = objc
    gi = np.array(gi); go = np.array(go)
    fns = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(fns, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the small key outside the frame is one
    # stamp. Enlarge it by the scale that the marker blocks inside the frame show,
    # line it up with those markers, then cut the frame out.
    # Where this differs from the maker: the unit is each key cell's block, painted
    # in the key's raster order. The maker instead groups same-colour components
    # into rectangles. Scale/offset detection and the final crop are the same in both.
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

    # frame: a colour sitting exactly on the four corners of a box that holds markers
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts}); cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if set(pts) != {(a, b) for a in rs for b in cs}:
            continue
        if rs[1] - rs[0] < 2 or cs[1] - cs[0] < 2:
            continue
        inner = {int(I[a, b]) for a in range(rs[0] + 1, rs[1]) for b in range(cs[0] + 1, cs[1])}
        inner -= {bgc, col}
        if len(inner) < 2:
            continue
        area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
        if best is None or area > best[0]:
            best = (area, rs[0], rs[1], cs[0], cs[1])
    _, r0, r1, c0, c1 = best

    # the key: every non-background cell outside the frame
    key = {p: v for v, pts in bycol.items() for p in pts
           if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1)}
    kr0 = min(r for r, _ in key); kc0 = min(c for _, c in key)
    inside = {(r, c): int(I[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
              if I[r, c] != bgc}

    # anchor colours: those whose (inside count / key count) equals the common block area
    kcnt = Counter(key.values()); icnt = Counter(inside.values())
    ratios = [icnt[c] / kcnt[c] for c in icnt if kcnt.get(c, 0)]
    R = Counter(ratios).most_common(1)[0][0]
    anchors = {c for c in icnt if kcnt.get(c, 0) and icnt[c] / kcnt[c] == R}
    ia = [p for p, v in inside.items() if v in anchors]
    ka = [p for p, v in key.items() if v in anchors]
    hs = (max(r for r, _ in ia) - min(r for r, _ in ia) + 1) // \
         (max(r for r, _ in ka) - min(r for r, _ in ka) + 1)
    ws = (max(c for _, c in ia) - min(c for _, c in ia) + 1) // \
         (max(c for _, c in ka) - min(c for _, c in ka) + 1)
    org_r = min(r for r, _ in ia) - (min(r for r, _ in ka) - kr0) * hs
    org_c = min(c for _, c in ia) - (min(c for _, c in ka) - kc0) * ws

    # Stamp the scaled key into the frame one key cell at a time, in the key's
    # raster order: paint that cell's hs x ws block (a full rectangle) in its colour.
    # Skip a block only if the current grid already shows all of it
    # (anchor markers, a pre-filled hint block).
    G = I.copy()
    ops, sels = [], []
    for (r, c) in sorted(key):
        col = key[(r, c)]
        br = org_r + (r - kr0) * hs; bc = org_c + (c - kc0) * ws
        if np.all(G[br:br + hs, bc:bc + ws] == col):
            continue
        ops.append(col); sels.append([br, bc, hs - 1, ws - 1])   # whole block rectangle
        G[br:br + hs, bc:bc + ws] = col

    # crop the canvas to the frame, corners included (a full rectangle)
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
