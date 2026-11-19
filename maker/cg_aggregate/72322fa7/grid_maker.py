"""
ARC Task: 72322fa7 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; the key colour pairs are
    # read off each grid's own 2-colour keys, so they may vary per instance.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    def neighbors(loc):
        r, c = loc
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    def mapply_nb(cells):
        out = set()
        for x in cells:
            out |= neighbors(x)
        return out

    def normalize(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return {(r - mr, c - mc) for r, c in cells}

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, min(30, max_h)))
    w = unifint(diff_lb, diff_ub, (10, min(30, max_w)))
    remcols = [c for c in cols if c != bgc]
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
    ccols = random.sample(remcols, 2 * nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(r, c) for r in range(h) for c in range(w)}
    objs = []
    for ca, cb in cpairs:
        oh = unifint(diff_lb, diff_ub, (1, 4))
        ow = unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(r, c) for r in range(oh) for c in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(2, oh * ow)
        for _ in range(ncells - 1):
            cand = sorted((bounds - obj) & mapply_nb(obj))
            obj.add(random.choice(cand))
        objn = normalize(obj)
        objt = sorted(objn)
        apart = set(random.sample(objt, random.randint(1, len(objt) - 1)))
        bpart = set(objt) - apart
        cobj = {(ca, x) for x in apart} | {(cb, x) for x in bpart}
        cands = [ij for ij in inds if all((ij[0] + r, ij[1] + c) in inds for r, c in objn)]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcdi = set()
        for col, (r, c) in cobj:
            gi[loc[0] + r][loc[1] + c] = col
            plcdi.add((loc[0] + r, loc[1] + c))
        inds = (inds - plcdi) - mapply_nb(plcdi)
        objs.append(cobj)
    if not objs:
        raise ValueError("no objects")
    avgs = sum(len(o) for o in objs) / len(objs)
    ub = max(1, int((h * w) // (avgs * 2)))
    noccs = unifint(diff_lb, diff_ub, (1, ub))
    succ, tr, maxtr = 0, 0, 5 * noccs
    go = [row[:] for row in gi]
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = random.choice(objs)
        pal = sorted({col for col, _ in obj})
        oh = max(r for _, (r, c) in obj) + 1
        ow = max(c for _, (r, c) in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = {(col, (loc[0] + r, loc[1] + c)) for col, (r, c) in obj}
        plcdi = {x for _, x in plcd}
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply_nb(plcdi)
            col = random.choice(pal)
            for cc, (r, c) in plcd:
                go[r][c] = cc
                if cc == col:
                    gi[r][c] = cc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading: each 2-colour key is ONE template. Wherever one of its single-colour
    parts appears alone in I, that is an incomplete copy of the template, and the
    whole key is stamped onto it: CopyI the key's rectangle once, then Paste it at
    each incomplete copy's origin. A colour-by-colour fill is used only when a
    transparent Paste of the whole rectangle cannot be done safely (a key colour is
    0, the rectangle holds foreign cells, or it would run off the grid).
    Everything is measured from I; O is used only for the Submit extent.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    if examples:
        cnt = Counter()
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                comps.append(cells)

    keys = [cl for cl in comps if len({int(I[y, x]) for y, x in cl}) == 2]

    ops, sels = [], []
    grid = I.copy()

    for cells in keys:
        cellset = set(cells)
        kr = min(y for y, _ in cells)
        kc = min(x for _, x in cells)
        kh = max(y for y, _ in cells) - kr + 1
        kw = max(x for _, x in cells) - kc + 1
        kcols = sorted({int(I[y, x]) for y, x in cells})
        key_rel = {(y - kr, x - kc): int(I[y, x]) for y, x in cells}
        holes_rel = [(y, x) for y in range(kh) for x in range(kw) if (y, x) not in key_rel]
        # can the key's rectangle be copied cleanly from I?
        src_clean = (0 not in kcols) and all(I[kr + y, kc + x] == bgc for y, x in holes_rel)

        # collect incomplete copies: (origin of key rectangle) for each lone part
        origins = []
        for ca in kcols:
            part = [(y, x) for y, x in cells if int(I[y, x]) == ca]
            ar = min(y for y, _ in part)
            ac = min(x for _, x in part)
            prel = [(y - ar, x - ac) for y, x in part]
            ph = max(y for y, _ in prel)
            pw = max(x for _, x in prel)
            for i in range(h - ph):
                for j in range(w - pw):
                    if all(I[i + y, j + x] == ca for y, x in prel):
                        o = (i - (ar - kr), j - (ac - kc))
                        if o not in origins:
                            origins.append(o)

        copied = False
        for (orr, occ) in origins:
            tgt = {(orr + y, occ + x): col for (y, x), col in key_rel.items()}
            in_grid = {p: col for p, col in tgt.items() if 0 <= p[0] < h and 0 <= p[1] < w}
            need = [p for p, col in in_grid.items() if grid[p] != in_grid[p]]
            if not need:
                continue
            fits = (orr >= 0 and occ >= 0 and orr + kh <= h and occ + kw <= w)
            paste_ok = src_clean and fits and all(
                grid[orr + y, occ + x] == bgc for y, x in holes_rel)
            if paste_ok:
                if not copied:
                    # copy the key's whole rectangle (holes are background, so the
                    # paste leaves the target's background unchanged)
                    ops.append(28); sels.append([kr, kc, kh - 1, kw - 1])
                    copied = True
                ops.append(30); sels.append([orr, occ, 0, 0])
                for p, col in in_grid.items():
                    grid[p] = col
            else:
                for col in kcols:
                    cs = [p for p in need if in_grid[p] == col]
                    if cs:
                        ops.append(int(col)); sels.append(sel_of(cs))
                        for p in cs:
                            grid[p] = col

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
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
                                f"for task 72322fa7"
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
                    f"Failed to build a complete episode for task 72322fa7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72322fa7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
