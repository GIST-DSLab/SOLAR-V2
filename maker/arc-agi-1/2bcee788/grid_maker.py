"""
ARC Task: 2bcee788 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [
    {"direction": "R"},
    {"direction": "L"},
    {"direction": "U"},
    {"direction": "D"},
]


def sample_colors(num_examples=None) -> dict:
    import random as _r
    cols = [c for c in range(1, 10) if c != 3]          # 3 is the output background; 0 breaks Copy/Paste
    bgc, sepc, objc = _r.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(_r.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        _r.shuffle(examples)
    else:
        examples = [dict(v) for v in _r.sample(VARIANTS, n_ex)]
    plan = examples + [dict(_r.choice(examples))]
    return {"bgc": bgc, "sepc": sepc, "objc": objc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sepc=None, objc=None, direction=None) -> dict:
    dirmap = {
        'identity': lambda v: v,
        'dmirror':  lambda v: (v[1], v[0]),
        'cmirror':  lambda v: (-v[1], -v[0]),
        'vmirror':  lambda v: (v[0], -v[1]),
        'hmirror':  lambda v: (-v[0], v[1]),
        'rot90':    lambda v: (v[1], -v[0]),
        'rot180':   lambda v: (-v[0], -v[1]),
        'rot270':   lambda v: (-v[1], v[0]),
    }
    mfs = ((identity, 'identity'), (dmirror, 'dmirror'), (cmirror, 'cmirror'),
           (vmirror, 'vmirror'), (hmirror, 'hmirror'), (rot90, 'rot90'),
           (rot180, 'rot180'), (rot270, 'rot270'))
    dvecs = {'R': (0, 1), 'L': (0, -1), 'D': (1, 0), 'U': (-1, 0)}
    if direction is None:
        direction = choice(('R', 'L', 'U', 'D'))
    target = dvecs[direction]
    while True:
        nmfs = choice((1, 2))
        fns = sample(mfs, nmfs)
        v = (0, 1)
        for _, nm in fns:
            v = dirmap[nm](v)
        if v == target:
            break
    # L/R keep the frame; U/D come from transposing transforms -> dims swap at the end
    if direction in ('L', 'R'):
        hlim, wlim = max_h, max_w
    else:
        hlim, wlim = max_w, max_h

    h = unifint(diff_lb, diff_ub, (2, min(20, hlim - 1)))
    w = unifint(diff_lb, diff_ub, (2, min(10, (wlim - 1) // 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(diff_lb, diff_ub, (h + 1, hlim))
    fullw = unifint(diff_lb, diff_ub, (2 * w + 1, wlim))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - h)
    locj = randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, gimini)
    go = paint(fullg, gomini)
    for fn, _ in fns:
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter, deque

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # exactly three colors: background > object > marker line (strict, by construction)
    cnt = Counter(I.flatten().tolist())
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    bgc = order[0][0]
    objc = order[1][0]
    sepc = order[2][0]

    obj = np.argwhere(I == objc)
    r0, c0 = int(obj[:, 0].min()), int(obj[:, 1].min())
    r1, c1 = int(obj[:, 0].max()), int(obj[:, 1].max())
    h, w = r1 - r0 + 1, c1 - c0 + 1

    sep = np.argwhere(I == sepc)
    srs = sorted(set(int(p) for p in sep[:, 0]))
    scs = sorted(set(int(p) for p in sep[:, 1]))

    # the marker line sits flush against one side of the object: that side is the mirror axis
    if len(scs) == 1 and scs[0] == c1 + 1:
        dr0, dc0, flip = r0, c1 + 1, 26            # mirror to the right  -> FlipH
    elif len(scs) == 1 and scs[0] == c0 - 1:
        dr0, dc0, flip = r0, c0 - w, 26            # mirror to the left   -> FlipH
    elif len(srs) == 1 and srs[0] == r1 + 1:
        dr0, dc0, flip = r1 + 1, c0, 27            # mirror downwards     -> FlipV
    else:
        dr0, dc0, flip = r0 - h, c0, 27            # mirror upwards       -> FlipV

    ops, sels = [], []

    # 1. every background region becomes 3 (object and marker untouched)
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = [(r, c)]
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == bgc:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            cells.append((ny, nx))
                comps.append(cells)
    comps.sort(key=len, reverse=True)              # open background first, then the enclosed pockets
    for cells in comps:
        sr, sc = cells[0]
        ops.append(13)
        sels.append([int(sr), int(sc), 0, 0])

    # 2. stamp the object (now sitting on a 3 background) across the marked side and mirror it
    ops.append(29)
    sels.append([r0, c0, h - 1, w - 1])
    ops.append(30)
    sels.append([dr0, dc0, 0, 0])
    ops.append(flip)
    sels.append([dr0, dc0, h - 1, w - 1])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
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
                                f"for task 2bcee788"
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
                    f"Failed to build a complete episode for task 2bcee788 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bcee788-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
