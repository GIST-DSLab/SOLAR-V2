"""
ARC Task: 57aa92db (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = max(a, min(b, round(a + (b - a) * diff_lb)))
    hi = max(a, min(b, round(a + (b - a) * diff_ub)))
    return random.randint(min(lo, hi), max(lo, hi))


def _nbrs(cells):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    out.add((r + dr, c + dc))
    return out


def sample_colors(num_examples=None) -> dict:
    bgc, fixc, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fixc=1, mainc=2) -> dict:
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = random.randint(2, 5)
    ow = random.randint(2, 5)
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    obj = {random.choice(sorted(bounds))}
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for _ in range(ncells - 1):
        obj.add(random.choice(sorted((bounds - obj) & _nbrs(obj))))
    r0 = min(r for r, c in obj); c0 = min(c for r, c in obj)
    obj = {(r - r0, c - c0) for r, c in obj}
    oh = max(r for r, c in obj) + 1
    ow = max(c for r, c in obj) + 1
    fixp = random.choice(sorted(obj))
    remcols = [c for c in range(10) if c not in (bgc, fixc, mainc)]
    gi = np.full((h, w), bgc, dtype=int)
    objc = {p: (fixc if p == fixp else mainc) for p in obj}
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = {(r + loci, c + locj): v for (r, c), v in objc.items()}
    for (r, c), v in plcd.items():
        gi[r, c] = v
    go = gi.copy()
    inds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc} - _nbrs(plcd.keys())
    nocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(obj)))))
    tr = succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        ups = {}
        for (r, c), v in objc.items():
            for a in range(fac):
                for b in range(fac):
                    ups[(r * fac + a, c * fac + b)] = v
        hh, ww = oh * fac, ow * fac
        cands = sorted(ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww)
        if not cands:
            continue
        li, lj = random.choice(cands)
        pl = {(r + li, c + lj): v for (r, c), v in ups.items()}
        if set(pl.keys()).issubset(inds):
            succ += 1
            newc = random.choice(remcols)
            fix2 = {p for p, v in pl.items() if v == fixc}
            inds = inds - _nbrs(pl.keys())
            for r, c in fix2:
                gi[r, c] = fixc; go[r, c] = fixc
            rem = {p for p, v in pl.items() if v != fixc}
            ntorem = _unifint(diff_lb, diff_ub, (0, max(0, len(rem) - 1)))
            ntokeep = len(rem) - ntorem
            fr0 = min(r for r, c in fix2); fr1 = max(r for r, c in fix2)
            fc0 = min(c for r, c in fix2); fc1 = max(c for r, c in fix2)
            outbox = {(r, c) for r in range(fr0 - 1, fr1 + 2) for c in range(fc0 - 1, fc1 + 2)
                      if r in (fr0 - 1, fr1 + 1) or c in (fc0 - 1, fc1 + 1)}
            tokeep = {random.choice(sorted(rem & outbox))}
            for _ in range(ntokeep - 1):
                opts = (rem & _nbrs(tokeep | fix2)) - tokeep
                tokeep.add(random.choice(sorted(opts)))
            for r, c in tokeep:
                gi[r, c] = newc
            for r, c in rem:
                go[r, c] = newc
    if succ == 0:
        raise ValueError("no copies placed")
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Reference reading: the indicator colour is the one present in every object; the
    template is the object with the largest size once its indicator is scaled to 1x1.
    For each other object, the template is rescaled by (its indicator size / template
    indicator size), recoloured to that object's colour and laid on top of it so the
    indicators coincide -> one Color(newc) on exactly the cells the rescaled template
    body lands on (a footprint computed from I alone)."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # 8-connected multicolour objects
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                seen[r, c] = True
                q = deque([(r, c)]); cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True; q.append((ny, nx))
                objs.append(cells)
    if len(objs) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # indicator colour: present in every object
    colsets = [{int(I[y, x]) for y, x in cells} for cells in objs]
    common = set.intersection(*colsets)
    if common:
        ind = sorted(common)[0]
    else:
        ind = Counter(c for s in colsets for c in s).most_common(1)[0][0]

    def isz(cells):
        return sum(1 for y, x in cells if I[y, x] == ind)

    # template: largest size after scaling indicator down to 1x1 (tie -> smallest indicator)
    def key(i):
        n = max(1, isz(objs[i]))
        return (len(objs[i]) // n, -n)
    ti = max(range(len(objs)), key=key)
    template = objs[ti]
    tind = [(y, x) for y, x in template if I[y, x] == ind]
    tr0 = min(y for y, x in template); tc0 = min(x for y, x in template)
    tbody = [(y - tr0, x - tc0) for y, x in template if I[y, x] != ind]
    tir = min(y for y, x in tind) - tr0
    tic = min(x for y, x in tind) - tc0
    tiw = max(x for y, x in tind) - min(x for y, x in tind) + 1

    G = I.copy()
    ops, sels = [], []
    for i, cells in enumerate(objs):
        if i == ti:
            continue
        icells = [(y, x) for y, x in cells if I[y, x] == ind]
        body = [int(I[y, x]) for y, x in cells if I[y, x] != ind]
        if not icells or not body:
            continue
        newc = Counter(body).most_common(1)[0][0]
        iw = max(x for y, x in icells) - min(x for y, x in icells) + 1
        sc = max(1, iw // tiw)                              # scale from indicator sizes
        orr = min(y for y, x in icells) - tir * sc          # align indicators
        occ = min(x for y, x in icells) - tic * sc
        foot = sorted({(orr + y * sc + a, occ + x * sc + b)
                       for y, x in tbody for a in range(sc) for b in range(sc)
                       if 0 <= orr + y * sc + a < hi and 0 <= occ + x * sc + b < wi})
        if not foot or all(G[r, c] == newc for r, c in foot):
            continue                                        # copy already complete: no-op
        # blit the rescaled, recoloured template body onto the partial copy
        ops.append(newc); sels.append(sel_of(foot))
        for r, c in foot:
            G[r, c] = newc

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
                        f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
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
                                f"for task 57aa92db"
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
                    f"Failed to build a complete episode for task 57aa92db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"57aa92db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
