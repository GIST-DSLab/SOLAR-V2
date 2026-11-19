"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transposed": False}, {"transposed": True}]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(list(range(10)), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])
    hlim = min(30, int(max_w if transposed else max_h))
    wlim = min(30, int(max_h if transposed else max_w))
    if hlim < 10 or wlim < 4:
        raise ValueError("grid limit too small")
    h = unifint(diff_lb, diff_ub, (10, hlim))
    w = unifint(diff_lb, diff_ub, (4, wlim))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = random.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(random.sample(totuple(cands), ncells))
        cells = insert(random.choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transposed:
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (golf): the pattern inside the two box walls is a stack of strips
    running PARALLEL to the walls.  Each strip is carried, unchanged, straight
    across its nearer wall; the strip that sat nearest the wall lands one gap
    cell outside it (two cells from the wall) and the rest stack outward in
    reverse order, so the half-pattern reappears turned over, outside.  Each
    strip is one Move chain (grab the strip, then empty-selection steps).
    Everything is measured from I only.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(I[0, 0])                  # canvas row 0 outside the box is never painted in I
    palette = sorted(set(I.flatten().tolist()) - {bgc})
    ops, sels = [], []
    if len(palette) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]
    bb = {}
    for c in palette:
        rs, cs = np.where(I == c)
        bb[c] = (int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max()))
    boxc = max(palette, key=lambda c: (bb[c][1] - bb[c][0] + 1) * (bb[c][3] - bb[c][2] + 1))
    objc = [c for c in palette if c != boxc][0]
    r0, r1, c0, c1 = bb[boxc]
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))

    # work in a frame where the walls are rows; map back when emitting
    if horizontal:
        F = I.copy()
        L1, L2 = r0, r1
        UP, DOWN, FLIP = 20, 21, 27
        to_g = lambda cells: [(a, b) for a, b in cells]
    else:
        F = I.T.copy()
        L1, L2 = c0, c1
        UP, DOWN, FLIP = 23, 22, 26
        to_g = lambda cells: [(b, a) for a, b in cells]
    FH, FW = F.shape
    mid = (L1 + L2) / 2.0
    cells = [(int(a), int(b)) for a, b in zip(*np.where(F == objc))]

    def apply_paint(G, cl, col):
        for a, b in cl:
            G[a, b] = col

    def sim_move(G, cl, d, steps):
        """Simulate an ARCLE grab-then-slide; return list of grids after each step."""
        snap = G.copy()
        vals = [(a, b, int(G[a, b])) for a, b in cl]
        for a, b in cl:
            snap[a, b] = 0
        out = []
        for k in range(1, steps + 1):
            g = snap.copy()
            for a, b, v in vals:
                na = a + d * k
                if 0 <= na < FH and v > 0:
                    g[na, b] = v
            out.append(g)
        return out

    def emit_move(G, cl, d, steps):
        """Emit + apply a slide (with bgc repair). Returns False if any step is a no-op."""
        if steps == 0:
            return True
        grids = sim_move(G, cl, d, steps)
        prev = G
        for g in grids:
            if np.array_equal(prev, g):
                return False
            prev = g
        local_ops, local_sels = [], []
        op = UP if d < 0 else DOWN
        local_ops.append(op); local_sels.append(sel_of(to_g(cl)))
        for _ in range(steps - 1):
            local_ops.append(op); local_sels.append(sel_of([]))
        newg = grids[-1]
        dest = {(a + d * steps, b) for a, b in cl}
        hole = sorted(set(cl) - dest)
        if bgc != 0 and hole:
            newg = newg.copy()
            apply_paint(newg, hole, bgc)
            local_ops.append(bgc); local_sels.append(sel_of(to_g(hole)))
        G[:, :] = newg
        ops.extend(local_ops); sels.extend(local_sels)
        return True

    for side in ("near_L1", "near_L2"):
        if side == "near_L1":
            half = [(a, b) for a, b in cells if a < mid]
            if not half:
                continue
            edge = min(a for a, _ in half)              # strip nearest the wall
            target = lambda a: (L1 - 2) - (a - edge)
        else:
            half = [(a, b) for a, b in cells if a > mid]
            if not half:
                continue
            edge = max(a for a, _ in half)
            target = lambda a: (L2 + 2) + (edge - a)
        strips = {}
        for a, b in half:
            strips.setdefault(a, []).append((a, b))
        rows_near_first = sorted(strips, key=lambda a: abs(a - edge))

        if objc == 0:
            # a 0-coloured strip is invisible to ARCLE's object Move -> repaint each strip
            for a in rows_near_first:
                cl = sorted(strips[a])
                dst = [(target(a), b) for _, b in cl]
                ops.append(bgc); sels.append(sel_of(to_g(cl))); apply_paint(F, cl, bgc)
                ops.append(0); sels.append(sel_of(to_g(dst))); apply_paint(F, dst, 0)
            continue

        done = False
        for order in (rows_near_first, rows_near_first[::-1]):
            Gtry = F.copy()
            save_ops, save_sels = list(ops), list(sels)
            ok = True
            for a in order:
                t = target(a)
                d = -1 if t < a else 1
                if not emit_move(F, sorted(strips[a]), d, abs(t - a)):
                    ok = False
                    break
            if ok:
                done = True
                break
            F[:, :] = Gtry
            ops[:], sels[:] = save_ops, save_sels
        if done:
            continue

        # Fallback (identical strips would make a slide step invisible):
        # carry the whole half across as one object, then turn it over in place.
        Gtry = F.copy()
        save_ops, save_sels = list(ops), list(sels)
        lo, hi_ = min(a for a, _ in half), max(a for a, _ in half)
        if side == "near_L1":
            d, steps = -1, hi_ - (L1 - 2)
        else:
            d, steps = 1, (L2 + 2) - lo
        ok = emit_move(F, sorted(half), d, steps)
        if ok:
            moved = sorted((a + d * steps, b) for a, b in half)
            mlo, mhi = moved[0][0], max(a for a, _ in moved)
            flipped = {(mlo + mhi - a, b) for a, b in moved}
            g = F.copy()
            for a, b in moved:
                g[a, b] = 0 if (a, b) not in flipped else g[a, b]
            for a, b in moved:
                g[mlo + mhi - a, b] = F[a, b]
            if np.array_equal(g, F):
                ok = False
            else:
                ops.append(FLIP); sels.append(sel_of(to_g(moved)))
                F[:, :] = g
                hole = sorted(set(moved) - flipped)
                if bgc != 0 and hole:
                    ops.append(bgc); sels.append(sel_of(to_g(hole)))
                    apply_paint(F, hole, bgc)
        if not ok:
            F[:, :] = Gtry
            ops[:], sels[:] = save_ops, save_sels
            for a in rows_near_first:
                cl = sorted(strips[a])
                dst = [(target(a), b) for _, b in cl]
                ops.append(bgc); sels.append(sel_of(to_g(cl))); apply_paint(F, cl, bgc)
                ops.append(objc); sels.append(sel_of(to_g(dst))); apply_paint(F, dst, objc)

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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
