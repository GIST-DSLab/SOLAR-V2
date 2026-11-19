"""
ARC Task: 0e206a2e (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, acol, bcol, ccol, Dcol = random.sample(cols, 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol, **kwargs) -> dict:
    # Reject draws where a coincidental marker triplet makes the rule ambiguous
    # (the generator's output disagrees with the declarative rule).
    from verifiers import verify_0e206a2e
    res = None
    for _ in range(30):
        res = _generate_raw(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol)
        if verify_0e206a2e(res["input"]) == res["output"]:
            return res
    raise RuntimeError("could not draw an unambiguous instance")


def _generate_raw(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, min(h, w) // 5))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1):
            shuffle(shp)
            a, b, c = shp[:3]
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {"input": gi, "output": go}


from arcle.actions.object import gen_move as _gen_move, gen_rotate as _gen_rotate, \
    gen_flip as _gen_flip, gen_copy as _gen_copy, gen_paste as _gen_paste

_SIM_OPS = {20: _gen_move(0), 21: _gen_move(1), 22: _gen_move(2), 23: _gen_move(3),
            24: _gen_rotate(1), 25: _gen_rotate(3), 26: _gen_flip("H"), 27: _gen_flip("V"),
            28: _gen_copy("I"), 29: _gen_copy("O"), 30: _gen_paste()}
_ARR_OPS = {26: np.fliplr, 27: np.flipud,
            24: lambda a: np.rot90(a, 1), 25: lambda a: np.rot90(a, 3)}


def derive_operations(I, O, examples=None):
    """
    Reading: the 4-colour objects are TEMPLATES (body + three single-cell markers);
    elsewhere bare marker triples are sprinkled. Each template is flipped/rotated and
    relocated so that its markers land on a matching triple. A template with several
    matching triples leaves a copy (CopyO + Paste of its current orientation) at each
    earlier triple and is itself MOVED onto the last one, vacating its source.
    Everything is measured from I; O is never read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    MH = MW = 30

    # ---- ARCLE state simulation (only to know where objects sit) -------------
    def z():
        return np.zeros((MH, MW), dtype=np.uint8)
    st = {"grid": z(), "grid_dim": (h, w), "input": z(), "input_dim": (h, w),
          "clip": z(), "clip_dim": (0, 0), "selected": z(),
          "object_states": {"active": 0, "object": z(), "object_sel": z(),
                            "background": z(), "object_pos": (0, 0),
                            "object_dim": (0, 0), "rotation_parity": 0}}
    st["grid"][:h, :w] = I
    st["input"][:h, :w] = I
    ops, sels = [], []

    def step(op, sel):
        if isinstance(sel, dict):
            mask = np.zeros((MH, MW), dtype=bool)
            for r, c in sel["cells"]:
                mask[r, c] = True
        else:
            r, c, hh, ww = sel
            mask = np.zeros((MH, MW), dtype=bool)
            mask[r:r + hh + 1, c:c + ww + 1] = True
        if op < 10:
            st["grid"][mask] = op
            st["object_states"]["active"] = 0
        else:
            if op in (28, 29, 30):
                st["object_states"]["active"] = 0
            _SIM_OPS[op](st, {"selection": mask, "operation": op})
        ops.append(int(op))
        sels.append(sel)

    def G():
        return st["grid"][:h, :w]

    def obj_view():
        od = st["object_states"]
        oh, ow = od["object_dim"]
        return od["object_pos"], np.array(od["object"][:oh, :ow], dtype=int), \
            np.array(od["object_sel"][:oh, :ow], dtype=bool)

    # ---- components & templates ---------------------------------------------
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
                comps.append(sorted(cells))
    templates = []
    for cells in comps:
        cols = [int(I[r, c]) for r, c in cells]
        if len(set(cols)) != 4:
            continue
        body = Counter(cols).most_common(1)[0][0]
        markers = [int(I[r, c]) for r, c in cells if int(I[r, c]) != body]
        if len(markers) == 3:
            templates.append((cells, body))

    masked = I.copy()
    for cells, _ in templates:
        for r, c in cells:
            masked[r, c] = bgc

    TF = [lambda a: a, lambda a: a.T, lambda a: np.rot90(a, 2).T, np.flipud, np.fliplr,
          lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 3)]

    all_pl = []
    for cells, body in templates:
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells); c1 = max(c for _, c in cells)
        A = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
        for r, c in cells:
            A[r - r0, c - c0] = I[r, c]
        # ---- matching triples, per orientation (found on the template-free view)
        placements, seenv = [], set()
        for f in TF:
            B = np.ascontiguousarray(f(A))
            key = (B.shape, B.tobytes())
            if key in seenv:
                continue
            seenv.add(key)
            mk = [(int(B[i, j]), i, j) for i in range(B.shape[0]) for j in range(B.shape[1])
                  if B[i, j] >= 0 and B[i, j] != body]
            for oi in range(-B.shape[0], h):
                for oj in range(-B.shape[1], w):
                    if all(0 <= oi + i < h and 0 <= oj + j < w and masked[oi + i, oj + j] == v
                           for v, i, j in mk):
                        placements.append((B, oi, oj))
        all_pl += placements
        if not placements:
            step(bgc, sel_of(cells))           # unused template: just erased
            continue

        def norm(a):
            nz = np.argwhere(a > 0)
            if len(nz) == 0:
                return a[:0, :0]
            return a[nz[:, 0].min():nz[:, 0].max() + 1, nz[:, 1].min():nz[:, 1].max() + 1]

        def plan_to(buf, B):
            tgt = norm(np.where(B > 0, B, 0))
            frontier = [(buf, [])]
            for _ in range(4):
                nxt = []
                for a, seq in frontier:
                    na = norm(a)
                    if na.shape == tgt.shape and np.array_equal(na, tgt):
                        return seq
                    for op, fn in _ARR_OPS.items():
                        nxt.append((fn(a), seq + [op]))
                frontier = nxt
            return None

        def fixups(B, oi, oj):
            need = {}
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    v = int(B[i, j])
                    r, c = oi + i, oj + j
                    if v >= 0 and 0 <= r < h and 0 <= c < w and int(G()[r, c]) != v:
                        need.setdefault(v, []).append((r, c))
            for v in sorted(need):
                step(v, sel_of(need[v]))

        grab = list(cells)          # first grab: the whole template object
        grabbed = set(cells)
        paint_mode = False
        remaining = list(placements)
        done = []
        while remaining:
            # next target: the orientation reachable with the fewest flips/rotations
            if grab is not None:
                gs = set(grab)
                gr0 = min(r for r, _ in grab); gc0 = min(c for _, c in grab)
                buf = np.array([[int(G()[r, c]) if (r, c) in gs else 0
                                 for c in range(gc0, max(c for _, c in grab) + 1)]
                                for r in range(gr0, max(r for r, _ in grab) + 1)])
            else:
                _, buf, _ = obj_view()
            if not paint_mode:
                cands = [(len(s), k) for k, p in enumerate(remaining)
                         for s in [plan_to(buf, p[0])] if s is not None]
                if not cands:
                    paint_mode = True
            if paint_mode:
                for B, oi, oj in remaining:
                    fixups(B, oi, oj)
                break
            _, k = min(cands)
            B, oi, oj = remaining.pop(k)
            last = not remaining
            if not last:
                done.append((B, oi, oj))
            seq = plan_to(buf, B)
            for op in seq:
                step(op, sel_of(grab) if grab is not None else sel_of([]))
                grab = None
            (px, py), obuf, osel = obj_view() if grab is None else ((gr0, gc0), buf, buf > 0)
            # marker-based offset from the object's current place to the target
            mv = next(v for v in np.unique(B) if v > 0 and v != body)
            bi, bj = map(int, np.argwhere(B == mv)[0])
            oa, ob = map(int, np.argwhere(obuf == mv)[0])
            dr, dc = (oi + bi) - (px + oa), (oj + bj) - (py + ob)
            disp = [(px + a, py + b) for a, b in np.argwhere(obuf > 0)
                    if 0 <= px + a < h and 0 <= py + b < w]
            if not last:
                # leave a copy of the object, in its current orientation, on the triple
                if any(0 <= r + dr < h and 0 <= c + dc < w and int(G()[r + dr, c + dc]) != int(G()[r, c])
                       for r, c in disp):
                    step(29, sel_of(disp))
                    pr = min(r for r, _ in disp) + dr
                    pc = min(c for _, c in disp) + dc
                    step(30, [pr, pc, 0, 0])
                fixups(B, oi, oj)
                grab = disp            # pick the template up again
                grabbed |= set(disp)
            else:
                # the template itself slides onto the last triple
                sel = sel_of(grab) if grab is not None else sel_of([])
                for d, n in ((21 if dr > 0 else 20, abs(dr)), (22 if dc > 0 else 23, abs(dc))):
                    for _ in range(n):
                        step(d, sel)
                        sel = sel_of([])
                if grab is not None and (dr or dc):
                    grab = None
                tgt = {(oi + i, oj + j) for i, j in np.argwhere(B >= 0)}
                hole = sorted(p for p in grabbed if int(G()[p]) == 0 and p not in tgt)
                if bgc != 0 and hole:
                    step(bgc, sel_of(hole))
                # touch up every placement of this template (also the earlier copies,
                # in case the template swept over one while turning in place)
                for pB, pi, pj in done + [(B, oi, oj)]:
                    fixups(pB, pi, pj)
        if paint_mode:
            # template could not be turned further (clipped at the edge): clear what is
            # left of it, restoring whatever it was covering
            bg = st["object_states"]["background"]
            want = {}
            for p in grabbed:
                if int(G()[p]) == 0 and bgc != 0:
                    want[p] = bgc
            if grab is None:
                (px, py), obuf, _ = obj_view()
                for a, b in np.argwhere(obuf > 0):
                    p = (px + a, py + b)
                    if 0 <= p[0] < h and 0 <= p[1] < w:
                        u = int(bg[p]) if int(bg[p]) != 0 else bgc
                        want[p] = u
            else:
                for p in grab:
                    want[p] = bgc
            for pB, pi, pj in placements:
                for i2, j2 in np.argwhere(pB >= 0):
                    want.pop((pi + i2, pj + j2), None)
            byc = {}
            for p, u in want.items():
                if int(G()[p]) != u:
                    byc.setdefault(u, []).append(p)
            for u in sorted(byc):
                step(u, sel_of(byc[u]))
            for pB, pi, pj in placements:
                fixups(pB, pi, pj)

    # final check over every placement: a template turning in place may have swept a
    # neighbouring placement drawn earlier
    for pB, pi, pj in all_pl:
        need = {}
        for i, j in np.argwhere(pB >= 0):
            r, c, v = pi + i, pj + j, int(pB[i, j])
            if 0 <= r < h and 0 <= c < w and int(G()[r, c]) != v:
                need.setdefault(v, []).append((r, c))
        for v in sorted(need):
            step(v, sel_of(need[v]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0e206a2e"
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
                                f"for task 0e206a2e"
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
                    f"Failed to build a complete episode for task 0e206a2e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0e206a2e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
