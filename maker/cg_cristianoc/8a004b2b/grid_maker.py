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


def _unif(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d + random.uniform(-0.5, 0.5)))))


def sample_colors(num_examples=None) -> dict:
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, cornc=4, ac1=1, ac2=2, objc=3) -> dict:
    m = min(max_h, max_w, 30)          # transposing transforms may swap h/w
    h = _unif(diff_lb, diff_ub, (10, m))
    w = _unif(diff_lb, diff_ub, (10, m))
    oh = _unif(diff_lb, diff_ub, (2, h // 5))
    ow = _unif(diff_lb, diff_ub, (2, w // 5))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    gi = np.full((h, w), bgc, dtype=int)
    obj = {random.choice(sorted(bounds))}
    ncellsd = _unif(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for _ in range(ncells - 1):
        nb = {(i + di, j + dj) for i, j in obj for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    cells = sorted(obj)
    fp1 = random.choice(cells)
    fp2 = random.choice([c for c in cells if c != fp1])
    colobj = {p: objc for p in obj}
    colobj[fp1] = ac1
    colobj[fp2] = ac2
    hscf = _unif(diff_lb, diff_ub, (1, (h - oh - 4) // oh))
    wscf = _unif(diff_lb, diff_ub, (1, (w - ow - 4) // ow))
    loci = random.randint(0, 2); locj = random.randint(0, 2)
    for (i, j), v in colobj.items():
        gi[i + loci, j + locj] = v
    inh, inw = hscf * oh, wscf * ow
    sqh = _unif(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = _unif(diff_lb, diff_ub, (inw + 2, w))
    sqloci = random.randint(loci + oh, h - sqh)
    sqlocj = random.randint(0, w - sqw)
    r0, r1, c0, c1 = sqloci, sqloci + sqh - 1, sqlocj, sqlocj + sqw - 1
    for r in (r0, r1):
        for c in (c0, c1):
            gi[r, c] = cornc
    mini = gi[loci:loci + oh, locj:locj + ow]
    goo = np.repeat(np.repeat(mini, hscf, axis=0), wscf, axis=1)
    gh, gw = goo.shape
    gloci = random.randint(r0 + 1, r1 - gh)
    glocj = random.randint(c0 + 1, c1 - gw)
    go = gi.copy()
    go[gloci:gloci + gh, glocj:glocj + gw] = goo
    go = go[r0:r1 + 1, c0:c1 + 1]
    for (i, j), v in colobj.items():
        if v in (ac1, ac2):
            gi[gloci + i * hscf:gloci + (i + 1) * hscf, glocj + j * wscf:glocj + (j + 1) * wscf] = v
    if random.choice((True, False)) and len(obj) > 3:
        i, j = random.choice(sorted(p for p, v in colobj.items() if v == objc))
        gi[gloci + i * hscf:gloci + (i + 1) * hscf, glocj + j * wscf:glocj + (j + 1) * wscf] = objc
    fns = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, -1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for k in random.sample(range(8), random.choice((1, 2))):
        gi = fns[k](gi)
        go = fns[k](go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour the generator paints first (majority colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    bycol = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                bycol.setdefault(v, []).append((r, c))

    # frame = four same-coloured corner pixels enclosing the marker blocks
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts}); cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2 or rs[1] - rs[0] < 2 or cs[1] - cs[0] < 2:
            continue
        if set(pts) != {(a, b) for a in rs for b in cs}:
            continue
        inner = {int(I[a, b]) for a in range(rs[0] + 1, rs[1])
                 for b in range(cs[0] + 1, cs[1])} - {bgc, col}
        if len(inner) < 2:
            continue
        area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
        if best is None or area > best[0]:
            best = (area, rs[0], rs[1], cs[0], cs[1])
    _, r0, r1, c0, c1 = best

    # the miniature legend = everything outside the frame, treated as ONE unit
    key = {p: col for col, pts in bycol.items() for p in pts
           if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1)}
    inside = {(r, c): int(I[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
              if I[r, c] != bgc}

    # anchor colours: the ones whose inside count / legend count agree (= scale area)
    kcnt = Counter(key.values()); icnt = Counter(inside.values())
    ratios = [icnt[c] / kcnt[c] for c in icnt if kcnt.get(c, 0) > 0]
    R = Counter(ratios).most_common(1)[0][0]
    anchors = {c for c in icnt if kcnt.get(c, 0) > 0 and icnt[c] / kcnt[c] == R}
    ia = [p for p, v in inside.items() if v in anchors]
    ka = [p for p, v in key.items() if v in anchors]
    ka_r0 = min(r for r, _ in ka); ka_c0 = min(c for _, c in ka)
    ia_r0 = min(r for r, _ in ia); ia_c0 = min(c for _, c in ia)
    hs = (max(r for r, _ in ia) - ia_r0 + 1) // (max(r for r, _ in ka) - ka_r0 + 1)
    ws = (max(c for _, c in ia) - ia_c0 + 1) // (max(c for _, c in ka) - ka_c0 + 1)

    # legend cell (r, c) -> its hs x ws block in the upscaled copy aligned on the anchors
    def block(r, c):
        br = ia_r0 + (r - ka_r0) * hs
        bc = ia_c0 + (c - ka_c0) * ws
        return [(a, b) for a in range(br, br + hs) for b in range(bc, bc + ws)]

    ops, sels = [], []
    # Paint the upscaled legend one colour layer at a time: every block of a legend
    # colour in one Color op (blocks already showing that colour in I are skipped).
    order = []
    for p in sorted(key):
        if key[p] not in order:
            order.append(key[p])
    for col in order:
        cells = []
        for p in sorted(q for q, v in key.items() if v == col):
            blk = block(*p)
            if all(I[a, b] == col for a, b in blk):
                continue
            cells.extend(blk)
        if cells:
            ops.append(col); sels.append(sel_of(cells))

    # crop to the frame: the full rectangle spanned by the four corner pixels
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
