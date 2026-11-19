"""
ARC Task: 5168d44c (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

_DIRS = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}
_DVEC = _DIRS


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dotcol = random.choice(rem)
    rem = [c for c in rem if c != dotcol]
    boxcol = random.choice(rem)
    variants = [{"direc": "down"}, {"direc": "right"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotcol=None, boxcol=None,
             direc=None, **kw) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)
    if bgc is None:
        bgc = random.randrange(10)
    if dotcol is None:
        dotcol = random.choice([c for c in range(10) if c != bgc])
    if boxcol is None:
        boxcol = random.choice([c for c in range(10) if c not in (bgc, dotcol)])
    if direc is None:
        direc = random.choice(["down", "right"])
    di, dj = _DIRS[direc]
    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    doth = unifint(diff_lb, diff_ub, (1, h // 3))
    dotw = unifint(diff_lb, diff_ub, (1, w // 3))
    borderh = unifint(diff_lb, diff_ub, (1, h // 4))
    borderw = unifint(diff_lb, diff_ub, (1, w // 4))
    dotloci = random.randint(0, h - doth - 1 if direc == "right" else h - doth - borderh - 1)
    dotlocj = random.randint(0, w - dotw - 1 if direc == "down" else w - dotw - borderw - 1)
    gi = np.full((h, w), bgc, dtype=int)
    oi, oj = di * (doth + borderh), dj * (dotw + borderw)
    dots = set()
    for k in range(-15, 16):
        for r in range(dotloci + k * oi, dotloci + k * oi + doth):
            for c in range(dotlocj + k * oj, dotlocj + k * oj + dotw):
                if 0 <= r < h and 0 <= c < w:
                    dots.add((r, c))
    for r, c in dots:
        gi[r, c] = dotcol
    starter = {(r, c) for r in range(dotloci, dotloci + doth) for c in range(dotlocj, dotlocj + dotw)}
    proto = {(r, c) for r in range(dotloci - borderh, dotloci + doth + borderh)
             for c in range(dotlocj - borderw, dotlocj + dotw + borderw)}
    bx = proto - starter
    go = gi.copy()
    for r, c in bx:
        if 0 <= r < h and 0 <= c < w:
            gi[r, c] = boxcol
    for r, c in bx:
        r2, c2 = r + oi, c + oj
        if 0 <= r2 < h and 0 <= c2 < w:
            go[r2, c2] = boxcol
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def comps(mask):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    cur = []
                    while stack:
                        rr, cc = stack.pop()
                        cur.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    out.append(cur)
        return out

    colors = sorted(set(int(v) for v in I.flatten().tolist()))
    hyp = None
    # Read the structure from I: periodic dots (dotcol); the anchor dot is the one
    # wrapped by a frame of a single other colour (boxcol).
    for dcol in colors:
        dmask = (I == dcol)
        dcells = set((int(r), int(c)) for r, c in np.argwhere(dmask).tolist())
        for comp in comps(dmask):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            ar0, ar1, ac0, ac1 = min(rs), max(rs), min(cs), max(cs)
            ah, aw = ar1 - ar0 + 1, ac1 - ac0 + 1
            if len(comp) != ah * aw:
                continue
            ring = [(r, c) for r in range(ar0 - 1, ar1 + 2) for c in range(ac0 - 1, ac1 + 2)
                    if 0 <= r < h and 0 <= c < w and not (ar0 <= r <= ar1 and ac0 <= c <= ac1)]
            if not ring:
                continue
            rcols = set(int(I[r, c]) for r, c in ring)
            if len(rcols) != 1:
                continue
            bcol = rcols.pop()
            if bcol == dcol:
                continue
            bcells = set((int(r), int(c)) for r, c in np.argwhere(I == bcol).tolist())
            br0 = min(r for r, _ in bcells); br1 = max(r for r, _ in bcells)
            bc0 = min(c for _, c in bcells); bc1 = max(c for _, c in bcells)
            if br0 > 0:
                bh = ar0 - br0
            elif br1 < h - 1:
                bh = br1 - ar1
            else:
                continue
            if bc0 > 0:
                bw = ac0 - bc0
            elif bc1 < w - 1:
                bw = bc1 - ac1
            else:
                continue
            if bh < 1 or bw < 1:
                continue
            R0, R1, C0, C1 = ar0 - bh, ar1 + bh, ac0 - bw, ac1 + bw
            recon = {(r, c)
                     for r in range(max(R0, 0), min(R1, h - 1) + 1)
                     for c in range(max(C0, 0), min(C1, w - 1) + 1)} - set(comp)
            if recon != bcells:
                continue
            found = None
            for name in ("unity", "down", "right"):
                di, dj = _DVEC[name]
                orr, occ = di * (ah + bh), dj * (aw + bw)
                rec = set()
                for k in range(-40, 41):
                    r0k, c0k = ar0 + k * orr, ac0 + k * occ
                    for r in range(r0k, r0k + ah):
                        for c in range(c0k, c0k + aw):
                            if 0 <= r < h and 0 <= c < w:
                                rec.add((r, c))
                if rec == dcells:
                    found = (orr, occ)
                    break
            if found is None:
                continue
            hyp = dict(dotcol=dcol, boxcol=bcol, rect=(R0, R1, C0, C1),
                       off=found, bcells=bcells)
            break
        if hyp is not None:
            break

    if hyp is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    boxcol, dotcol = int(hyp['boxcol']), int(hyp['dotcol'])
    rest = [c for c in colors if c not in (boxcol, dotcol)]
    if rest:
        bgc = int(rest[0])
    else:
        bgc = 0
        if examples:
            for ei, _ in examples:
                ec = set(np.asarray(ei).flatten().tolist()) - {boxcol, dotcol}
                if len(ec) == 1:
                    bgc = int(ec.pop()); break
    orr, occ = hyp['off']
    R0, R1, C0, C1 = hyp['rect']
    old_frame = set(hyp['bcells'])

    # Redraw the frame around the NEXT dot as one paint stroke: the frame's
    # rectangle shifted by the dot period, minus the dot cells already inside it.
    rect = {(r, c)
            for r in range(max(R0 + orr, 0), min(R1 + orr, h - 1) + 1)
            for c in range(max(C0 + occ, 0), min(C1 + occ, w - 1) + 1)}
    new_frame = {(r, c) for (r, c) in rect if I[r, c] != dotcol}
    paint_new = sorted(p for p in new_frame if I[p] != boxcol)
    if paint_new:
        ops.append(boxcol); sels.append(sel_of(paint_new))
    # Then wipe whatever part of the old frame the new frame does not reuse.
    wipe = sorted(old_frame - new_frame)
    if wipe:
        ops.append(bgc); sels.append(sel_of(wipe))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5168d44c"
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
                                f"for task 5168d44c"
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
                    f"Failed to build a complete episode for task 5168d44c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5168d44c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
