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

DIRS = {"down": (1, 0), "right": (0, 1), "unity": (1, 1)}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    dotcol = random.choice(rem)
    rem = [c for c in rem if c != dotcol]
    boxcol = random.choice(rem)
    names = list(DIRS)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(names):
        ex = [{"direc": d} for d in names]
        ex += [{"direc": random.choice(names)} for _ in range(n_ex - len(names))]
        random.shuffle(ex)
    else:
        ex = [{"direc": d} for d in random.sample(names, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotcol": dotcol, "boxcol": boxcol, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotcol, boxcol, direc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round(lb * (b - a)); hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))
    if direc is None:
        direc = random.choice(list(DIRS))
    di, dj = DIRS[direc]
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
    starter = {(dotloci + r, dotlocj + c) for r in range(doth) for c in range(dotw)}
    for k in range(-15, 16):
        for (r, c) in starter:
            rr, cc = r + k * oi, c + k * oj
            if 0 <= rr < h and 0 <= cc < w:
                gi[rr, cc] = dotcol
    bx = {(r, c) for r in range(dotloci - borderh, dotloci + doth + borderh)
          for c in range(dotlocj - borderw, dotlocj + dotw + borderw)} - starter
    go = gi.copy()
    for (r, c) in bx:
        if 0 <= r < h and 0 <= c < w:
            gi[r, c] = boxcol
        if 0 <= r + oi < h and 0 <= c + oj < w:
            go[r + oi, c + oj] = boxcol
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the frame and the dot it encloses
    # are ONE window on the periodic dot track; the window advances by one
    # period, so the whole framed block (frame + its dot) is slid along the
    # track and lands exactly over the next dot. The dot it leaves behind is
    # then put back, and the rest of its old footprint returns to background.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    def comps(mask):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]; seen[r, c] = True; cur = []
                    while stack:
                        rr, cc = stack.pop(); cur.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True; stack.append((nr, nc))
                    out.append(cur)
        return out

    colors = sorted(set(int(v) for v in I.flatten().tolist()))
    hyp = None
    for dcol in colors:
        dmask = (I == dcol)
        dcells = set((int(r), int(c)) for r, c in np.argwhere(dmask).tolist())
        for comp in comps(dmask):
            rs = [r for r, _ in comp]; cs = [c for _, c in comp]
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
            recon = {(r, c) for r in range(max(R0, 0), min(R1, h - 1) + 1)
                     for c in range(max(C0, 0), min(C1, w - 1) + 1)} - set(comp)
            if recon != bcells:
                continue
            found = None
            for name in ("unity", "down", "right"):
                di, dj = DIRS[name]
                orr, occ = di * (ah + bh), dj * (aw + bw)
                rec = set()
                for k in range(-40, 41):
                    for r in range(ar0 + k * orr, ar0 + k * orr + ah):
                        for c in range(ac0 + k * occ, ac0 + k * occ + aw):
                            if 0 <= r < h and 0 <= c < w:
                                rec.add((r, c))
                if rec == dcells:
                    found = (orr, occ); break
            if found is None:
                continue
            hyp = dict(dotcol=dcol, boxcol=bcol, anchor=set(comp),
                       rect=(R0, R1, C0, C1), off=found, bcells=bcells)
            break
        if hyp is not None:
            break

    if hyp is None:
        raise ValueError("no frame/dot-track structure found in I")

    boxcol, dotcol = hyp["boxcol"], hyp["dotcol"]
    rest = [c for c in colors if c not in (boxcol, dotcol)]
    bgc = rest[0] if rest else 0
    if examples:   # every demonstration is drawn on the same background
        for ex_i, _ in examples:
            E = np.asarray(ex_i, dtype=int)
            cand = [c for c in set(E.flatten().tolist()) if c not in (boxcol, dotcol)]
            if len(cand) == 1:
                bgc = int(cand[0]); break

    anchor = hyp["anchor"]
    frame = hyp["bcells"]
    orr, occ = hyp["off"]
    R0, R1, C0, C1 = hyp["rect"]
    block = frame | anchor                    # the framed block: full rectangle (clipped)

    state = I.copy()
    carry = boxcol
    if boxcol == 0:
        # Move only carries non-zero cells; give the 0-frame a stand-in colour
        carry = next(c for c in range(1, 10) if c not in colors)
        ops.append(carry); sels.append(sel_of(sorted(frame)))
        for (r, c) in frame:
            state[r, c] = carry

    snap = state.copy()
    grabbed = {(r, c): int(state[r, c]) for (r, c) in block if state[r, c] != 0}
    for (r, c) in grabbed:
        snap[r, c] = 0

    def render(obj):
        g = snap.copy()
        for (r, c), v in obj.items():
            g[r, c] = v
        return g

    obj = dict(grabbed)
    first = True
    di = 1 if orr else 0
    dj = 1 if occ else 0
    # slide the framed block one period along the track (down then right):
    # one grab, then empty selections keep the same block grabbed
    for (sr, sc, cnt, mop) in ((di, 0, orr, 21), (0, dj, occ, 22)):
        for _ in range(cnt):
            nobj = {(r + sr, c + sc): v for (r, c), v in obj.items()
                    if 0 <= r + sr < h and 0 <= c + sc < w}
            ns = render(nobj)
            if np.array_equal(ns, state):
                break
            ops.append(mop); sels.append(sel_of(sorted(block)) if first else sel_of([]))
            first = False
            obj, state = nobj, ns

    # the frame as it stands around the next dot (incl. parts off-canvas in I)
    full_frame = {(r, c) for r in range(R0, R1 + 1) for c in range(C0, C1 + 1)} - \
                 {(r, c) for (r, c) in anchor}
    dest_frame = {(r + orr, c + occ) for (r, c) in full_frame
                  if 0 <= r + orr < h and 0 <= c + occ < w}
    if boxcol == 0:
        ops.append(0); sels.append(sel_of(sorted(dest_frame)))
        for (r, c) in dest_frame:
            state[r, c] = 0
    else:
        scrolled_in = sorted(p for p in dest_frame if state[p] != boxcol)
        if scrolled_in:
            ops.append(boxcol); sels.append(sel_of(scrolled_in))
            for p in scrolled_in:
                state[p] = boxcol

    # the dot the block left behind is part of the track: put it back
    if dotcol != 0:
        ops.append(dotcol); sels.append(sel_of(sorted(anchor)))
        for p in anchor:
            state[p] = dotcol
    # rest of the block's old footprint (old frame, not under the new block) -> background
    dest_block = dest_frame | {(r + orr, c + occ) for (r, c) in anchor}
    vac = sorted(p for p in frame - dest_block if state[p] != bgc)
    if vac:
        ops.append(bgc); sels.append(sel_of(vac))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # full-grid bbox: submit
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
