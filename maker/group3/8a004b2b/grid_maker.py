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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(round((b - a) * max(0.0, lb)))
    hi = a + int(round((b - a) * min(1.0, ub)))
    lo = max(a, min(lo, b)); hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cornc, ac1, ac2, objc) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = _unifint(diff_lb, diff_ub, (2, h // 5))
    ow = _unifint(diff_lb, diff_ub, (2, w // 5))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    gi = [[bgc] * w for _ in range(h)]
    obj = {random.choice(sorted(bounds))}
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for _ in range(ncells - 1):
        nb = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        nb.add((i + di, j + dj))
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1; ow = max(j for _, j in obj) + 1
    objl = sorted(obj)
    fp1 = random.choice(objl)
    fp2 = random.choice([p for p in objl if p != fp1])
    key = {p: objc for p in obj}
    key[fp1] = ac1; key[fp2] = ac2
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    hscf = _unifint(diff_lb, diff_ub, (1, maxhscf))
    wscf = _unifint(diff_lb, diff_ub, (1, maxwscf))
    loci = random.randint(0, 2); locj = random.randint(0, 2)
    for (i, j), v in key.items():
        gi[i + loci][j + locj] = v
    inh = hscf * oh; inw = wscf * ow
    sqh = _unifint(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = _unifint(diff_lb, diff_ub, (inw + 2, w))
    sqloci = random.randint(loci + oh, h - sqh)
    sqlocj = random.randint(0, w - sqw)
    for (a, b) in ((sqloci, sqlocj), (sqloci, sqlocj + sqw - 1),
                   (sqloci + sqh - 1, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)):
        gi[a][b] = cornc
    gloci = random.randint(sqloci + 1, sqloci + sqh - 1 - inh)
    glocj = random.randint(sqlocj + 1, sqlocj + sqw - 1 - inw)
    go = [row[:] for row in gi]
    for i in range(oh):
        for j in range(ow):
            v = gi[i + loci][j + locj]
            for a in range(hscf):
                for b in range(wscf):
                    go[gloci + i * hscf + a][glocj + j * wscf + b] = v
                    if v in (ac1, ac2):
                        gi[gloci + i * hscf + a][glocj + j * wscf + b] = v
    if random.choice((True, False)) and len(key) > 3:
        objcells = [p for p, v in key.items() if v == objc]
        idxi, idxj = random.choice(objcells)
        for a in range(hscf):
            for b in range(wscf):
                gi[gloci + idxi * hscf + a][glocj + idxj * wscf + b] = objc
    go = [row[sqlocj:sqlocj + sqw] for row in go[sqloci:sqloci + sqh]]
    gi = np.array(gi); go = np.array(go)
    fns = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    swaps = {1, 2, 5, 7}
    allowed = [k for k in range(8) if k not in swaps or (h <= max_w and w <= max_h)]
    for k in random.sample(allowed, min(len(allowed), random.choice((1, 2)))):
        gi = fns[k](gi); go = fns[k](go)
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

    # frame = the four lone corner pixels of one colour enclosing markers
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts}); cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if set(pts) != {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}:
            continue
        if rs[1] - rs[0] < 2 or cs[1] - cs[0] < 2:
            continue
        inner = {int(I[r, c]) for r in range(rs[0] + 1, rs[1]) for c in range(cs[0] + 1, cs[1])
                 if I[r, c] != bgc and I[r, c] != col}
        if len(inner) < 2:
            continue
        area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
        if best is None or area > best[0]:
            best = (area, rs[0], rs[1], cs[0], cs[1])
    _, r0, r1, c0, c1 = best

    # key = everything outside the frame
    key = {}
    for col, pts in bycol.items():
        for (r, c) in pts:
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                key[(r, c)] = col
    kr0 = min(r for r, _ in key); kc0 = min(c for _, c in key)

    inside = {(r, c): int(I[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
              if I[r, c] != bgc}
    kcnt = Counter(key.values()); icnt = Counter(inside.values())
    ratios = [icnt[c] / kcnt[c] for c in icnt if kcnt.get(c, 0) > 0]
    R = Counter(ratios).most_common(1)[0][0]
    anchors = {c for c in icnt if kcnt.get(c, 0) > 0 and icnt[c] / kcnt[c] == R}
    ia = [p for p, v in inside.items() if v in anchors]
    ka = [p for p, v in key.items() if v in anchors]
    hs = (max(r for r, _ in ia) - min(r for r, _ in ia) + 1) // (max(r for r, _ in ka) - min(r for r, _ in ka) + 1)
    ws = (max(c for _, c in ia) - min(c for _, c in ia) + 1) // (max(c for _, c in ka) - min(c for _, c in ka) + 1)
    org_r = min(r for r, _ in ia) - (min(r for r, _ in ka) - kr0) * hs
    org_c = min(c for _, c in ia) - (min(c for _, c in ka) - kc0) * ws

    ops, sels = [], []
    # 1. Crop to the frame first (key layout already read from I).
    #    The frame is a full rectangle including background, so a bbox is exact.
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    G = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sub = I[r0:r1 + 1, c0:c1 + 1]
    G[sub != 0] = sub[sub != 0]

    # 2. One colour at a time: paint the whole upscaled footprint of that key
    #    colour (cropped coordinates), skipping cells that already hold it now.
    for col in sorted(set(key.values()), key=lambda c: (-kcnt[c], c)):
        cells = []
        for (r, c), v in key.items():
            if v != col:
                continue
            br = org_r + (r - kr0) * hs - r0
            bc = org_c + (c - kc0) * ws - c0
            for a in range(hs):
                for b in range(ws):
                    if G[br + a, bc + b] != col:
                        cells.append((br + a, bc + b))
        if not cells:
            continue
        cells.sort()
        ops.append(col); sels.append(sel_of(cells))
        for p in cells:
            G[p] = col

    ho, wo = G.shape
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
