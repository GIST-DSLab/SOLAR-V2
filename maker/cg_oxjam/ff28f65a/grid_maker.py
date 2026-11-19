"""
ARC Task: ff28f65a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = max(3, min(unifint(diff_lb, diff_ub, (3, max(3, max_h))), max_h))
    w = max(3, min(unifint(diff_lb, diff_ub, (3, max(3, max_w))), max_w))
    nred = random.randint(1, 5)
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def place(color):
        nonlocal inds
        oh = random.randint(1, h // 2 + 1)
        ow = random.randint(1, w // 2 + 1)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            return False
        loci, locj = random.choice(sorted(cands))
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if not bd.issubset(inds):
            return False
        nb = set()
        for (i, j) in bd:
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb.add((i + di, j + dj))
        inds = (inds - bd) - nb
        for (i, j) in bd:
            gi[i][j] = color
        return True

    succ, tr = 0, 0
    while tr < 5 * nred and succ < nred:
        tr += 1
        if place(2):
            succ += 1
    nblue = succ
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    remcols = [c for c in cols if c != bgc]
    succ, tr = 0, 1
    while tr < 5 * namt and succ < namt:
        tr += 1
        if place(random.choice(remcols)):
            succ += 1
    go = [[bgc] * 3 for _ in range(3)]
    for k in range(nblue):
        r, c = mpr[k + 1]
        go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: one blue pip per red rectangle, laid every other cell along a
    # 1x9 strip (strip cells 0,2,4,6,8), and the strip is then folded into a
    # 3x3 board -- its middle third slides down to row 1, its last third to row 2.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the board colour of the demonstration outputs (non-pip colour)
    bgc = None
    if examples:
        cnt = Counter()
        for _, eo in examples:
            cnt.update(int(x) for x in np.asarray(eo).flatten() if int(x) != 1)
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None:
        cnt = Counter(int(x) for x in I.flatten() if int(x) != 2)
        bgc = cnt.most_common(1)[0][0] if cnt else 0

    # count red rectangles (4-connected components of colour 2)
    seen = np.zeros((hi, wi), dtype=bool)
    nred = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == 2 and not seen[r, c]:
                nred += 1
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 2:
                            seen[ny, nx] = True
                            q.append((ny, nx))
    nred = min(nred, 5)

    # ---- plan as op "units" (a fold chain is one unit) ----------------------
    # strip workspace: a 3x9 canvas; prefer an all-background 3x9 window of I
    win = None
    for r in range(hi - 2):
        for c in range(wi - 8):
            if bool((I[r:r + 3, c:c + 9] == bgc).all()):
                win = (r, c)
                break
        if win is not None:
            break

    units = []  # (name, [(op, sel), ...]) ; sel either bbox list or cell list
    if wi >= 9:
        if win is not None:
            units.append(("crop9", [(33, [win[0], win[1], 2, 8])]))
        else:
            units.append(("crop9", [(33, [0, 0, 2, 8])]))
            units.append(("fill", [(bgc, [0, 0, 2, 8])]))   # lay the blank strip
        for k in range(nred):                                 # one pip per red object
            units.append(("pip", [(1, [0, 2 * k, 0, 0])]))
        for seg in (1, 2):                                    # fold the strip's thirds
            if 2 * (nred - 1) < 3 * seg:                      # this third holds no pip
                continue
            cells = [(0, 3 * seg + j) for j in range(3)]
            chain = [(21, cells)] + [(21, [])] * (seg - 1) + [(23, [])] * (3 * seg)
            units.append(("fold", chain))
        units.append(("crop3", [(33, [0, 0, 2, 2])]))
    else:
        # input narrower than the 9-cell strip: the folded positions of strip
        # cells 0,2,4,6,8 are placed straight onto a 3x3 board
        fold_pos = [(i // 3, i % 3) for i in range(0, 9, 2)]
        win3 = None
        for r in range(hi - 2):
            for c in range(wi - 2):
                if bool((I[r:r + 3, c:c + 3] == bgc).all()):
                    win3 = (r, c)
                    break
            if win3 is not None:
                break
        if win3 is not None:
            units.append(("crop3", [(33, [win3[0], win3[1], 2, 2])]))
        else:
            units.append(("crop3", [(33, [0, 0, 2, 2])]))
            units.append(("fill", [(bgc, [0, 0, 2, 2])]))
        for k in range(nred):
            pr, pc = fold_pos[k]
            units.append(("pip", [(1, [pr, pc, 0, 0])]))

    # ---- tiny ARCLE-faithful simulator used to drop ops with no effect -------
    def run(us):
        g = np.zeros((30, 30), dtype=int)
        g[:hi, :wi] = I
        gh, gw = hi, wi
        obj = None
        for _, steps in us:
            for op, sel in steps:
                if op == 33:
                    r, c, h, w = sel
                    patch = g[r:r + h + 1, c:c + w + 1].copy()
                    g = np.zeros((30, 30), dtype=int)
                    g[:h + 1, :w + 1] = patch
                    gh, gw = h + 1, w + 1
                    obj = None
                elif op <= 9:
                    r, c, h, w = sel
                    g[r:r + h + 1, c:c + w + 1] = op   # ARCLE paints the mask as given
                    obj = None
                else:
                    if sel:
                        bg = g.copy()
                        vals = [(rr, cc, g[rr, cc]) for rr, cc in sel]
                        for rr, cc in sel:
                            bg[rr, cc] = 0
                        obj = [bg, vals, 0, 0]
                    obj[2] += 1 if op == 21 else 0
                    obj[3] -= 1 if op == 23 else 0
                    g = obj[0].copy()
                    for rr, cc, v in obj[1]:
                        nr, nc = rr + obj[2], cc + obj[3]
                        if v > 0 and 0 <= nr < gh and 0 <= nc < gw:
                            g[nr, nc] = v
        return g[:gh, :gw].copy()

    target = run(units)
    changed = True
    while changed:
        changed = False
        for i, (name, _) in enumerate(units):
            if name in ("crop9", "fill", "crop3"):
                trial = units[:i] + units[i + 1:]
                res = run(trial)
                if res.shape == target.shape and np.array_equal(res, target):
                    units = trial
                    changed = True
                    break

    ops, sels = [], []
    for _, steps in units:
        for op, sel in steps:
            ops.append(op)
            sels.append(sel_of(sel) if isinstance(sel, list) and (len(sel) == 0 or isinstance(sel[0], tuple)) else sel)
    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
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
                                f"for task ff28f65a"
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
                    f"Failed to build a complete episode for task ff28f65a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff28f65a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
