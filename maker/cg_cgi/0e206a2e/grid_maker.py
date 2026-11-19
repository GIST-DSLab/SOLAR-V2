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
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, acol, bcol, ccol, Dcol = random.sample(cols, 5)
    return {"bgc": bgc, "acol": acol, "bcol": bcol, "ccol": ccol, "Dcol": Dcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol, **kwargs) -> dict:
    # keep only draws where the declarative rule reproduces the generator's output
    # (a coincidental marker triple would otherwise make the instance ambiguous)
    from verifiers import verify_0e206a2e
    for _ in range(30):
        res = _generate_raw(diff_lb, diff_ub, max_h, max_w, bgc, acol, bcol, ccol, Dcol)
        if verify_0e206a2e(res["input"]) == res["output"]:
            return res
    raise ValueError("could not draw an unambiguous instance")


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

_OBJ_OPS = {20: _gen_move(0), 21: _gen_move(1), 22: _gen_move(2), 23: _gen_move(3),
            24: _gen_rotate(1), 25: _gen_rotate(3), 26: _gen_flip("H"), 27: _gen_flip("V")}
_CLIP_OPS = {28: _gen_copy("I"), 29: _gen_copy("O"), 30: _gen_paste()}


def derive_operations(I, O, examples=None):
    """
    Reading (golf decomposition): the 4-colour objects are TEMPLATES (a body
    colour plus three single-cell markers); elsewhere bare marker triples are
    sprinkled. Each template is turned (Flip/Rotate) into the orientation whose
    markers fit a sprinkled triple and is MOVED onto it. If a template fits several
    triples, a copy of its turned self is left on each earlier triple (CopyO+Paste)
    and the template itself slides onto the last one. A template fitting nothing is
    removed. Everything is measured from I; O is never read.
    """
    import copy as _copy
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    M = 30

    def z():
        return np.zeros((M, M), dtype=np.uint8)
    st = {"grid": z(), "grid_dim": (h, w), "input": z(), "input_dim": (h, w),
          "clip": z(), "clip_dim": (0, 0), "selected": z(),
          "object_states": {"active": 0, "object": z(), "object_sel": z(),
                            "background": z(), "object_pos": (0, 0),
                            "object_dim": (0, 0), "rotation_parity": 0}}
    st["grid"][:h, :w] = I
    st["input"][:h, :w] = I
    ops, sels = [], []

    def step(op, sel):
        mask = np.zeros((M, M), dtype=bool)
        for r, c in sel["cells"] if isinstance(sel, dict) else []:
            mask[r, c] = True
        if not isinstance(sel, dict):
            r, c, hh, ww = sel
            mask[r:r + hh + 1, c:c + ww + 1] = True
        st["selected"] = np.zeros((M, M), dtype=np.uint8)
        if op < 10:
            st["grid"][mask] = op
            st["object_states"]["active"] = 0
        elif op in _CLIP_OPS:
            st["object_states"]["active"] = 0
            _CLIP_OPS[op](st, {"selection": mask, "operation": op})
        else:
            _OBJ_OPS[op](st, {"selection": mask, "operation": op})
        ops.append(int(op))
        sels.append(sel)

    def G():
        return st["grid"][:h, :w]

    # ---- 1. 4-connected non-background components ---------------------------
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

    # ---- 2. templates: components holding exactly four colours ---------------
    templates = []
    for cells in comps:
        cols = [int(I[r, c]) for r, c in cells]
        if len(set(cols)) != 4:
            continue
        body = Counter(cols).most_common(1)[0][0]
        if sum(1 for v in cols if v != body) == 3:
            templates.append((cells, body))

    loose = I.copy()
    for cells, _ in templates:
        for r, c in cells:
            loose[r, c] = bgc

    TF = [lambda a: a, lambda a: a.T, lambda a: np.rot90(a, 2).T, np.flipud, np.fliplr,
          lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 3)]

    def same(vals, msk, B):
        return (vals.shape == B.shape and np.array_equal(msk, B >= 0)
                and np.array_equal(np.where(B > 0, vals, 0), np.where(B > 0, B, 0)))

    MOVES = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}
    DEPTH, RADIUS = 14, 6
    lifted = set()

    def visible(pos, msk):
        return [(pos[0] + i, pos[1] + j) for i, j in np.argwhere(msk)
                if 0 <= pos[0] + i < h and 0 <= pos[1] + j < w]

    def clean(state, lift):
        od = state["object_states"]
        if not od["active"]:
            return True
        pr, pc = od["object_pos"]
        oh, ow = od["object_dim"]
        m = np.array(od["object_sel"][:oh, :ow], dtype=bool)
        for i, j in np.argwhere(m):
            r, c = pr + i, pc + j
            if not (0 <= r < h and 0 <= c < w):
                return False
            if (r, c) not in lift and int(od["background"][r, c]) != bgc:
                return False
        return True

    def run_on(state, op, sel):
        global_st = dict(st)
        st.clear(); st.update(state)
        n = len(ops)
        step(op, sel)
        del ops[n:]; del sels[n:]
        new = dict(st)
        st.clear(); st.update(global_st)
        return new

    def plan(pose, held, B, need_clean_end, avoid=frozenset()):
        root = _copy.deepcopy(st)
        start = (root, pose, held, [], frozenset(lifted))
        frontier, seenk = [start], set()
        for depth in range(DEPTH + 1):
            nxt = []
            for state, (pos, vals, msk), hd, path, lift in frontier:
                if same(vals, msk, B) and (not need_clean_end or (
                        clean(state, lift) and not (set(visible(pos, msk)) & avoid))):
                    return state, (pos, vals, msk), hd, path, lift
                key = (pos, vals.tobytes(), vals.shape, msk.tobytes(), hd)
                if key in seenk:
                    continue
                seenk.add(key)
                if depth == DEPTH:
                    continue
                cells = visible(pos, msk)
                grid0 = state["grid"][:h, :w]
                for op in (26, 27, 25, 24, 20, 21, 22, 23):
                    if op in MOVES:
                        if abs(pos[0] + MOVES[op][0] - pose[0][0]) + \
                                abs(pos[1] + MOVES[op][1] - pose[0][1]) > RADIUS:
                            continue
                        sel = sel_of([]) if hd else sel_of(cells)
                    else:
                        if hd and not clean(state, lift):
                            continue
                        sel = sel_of(cells)
                    ns = run_on(_copy.deepcopy(state), op, sel)
                    if np.array_equal(ns["grid"][:h, :w], grid0):
                        continue
                    od = ns["object_states"]
                    oh, ow = od["object_dim"]
                    npose = (tuple(od["object_pos"]),
                             np.array(od["object"][:oh, :ow], dtype=int),
                             np.array(od["object_sel"][:oh, :ow], dtype=bool))
                    nl = lift if (sel["cells"] == []) else lift | set(cells)
                    nxt.append((ns, npose, True, path + [(op, sel)], frozenset(nl)))
            frontier = nxt
        return None

    for cells, body in templates:
        r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
        A = np.full((max(r for r, _ in cells) - r0 + 1, max(c for _, c in cells) - c0 + 1),
                    -1, dtype=int)
        for r, c in cells:
            A[r - r0, c - c0] = I[r, c]

        # ---- 3. where does this template fit? (any of the 8 orientations) -----
        places, keys = [], set()
        for f in TF:
            B = np.ascontiguousarray(f(A))
            mk = [(int(B[i, j]), i, j) for i, j in np.argwhere(B >= 0) if B[i, j] != body]
            for oi in range(0, h - B.shape[0] + 1):
                for oj in range(0, w - B.shape[1] + 1):
                    if all(loose[oi + i, oj + j] == v for v, i, j in mk):
                        key = frozenset((oi + i, oj + j, int(B[i, j]))
                                        for i, j in np.argwhere(B >= 0))
                        if key not in keys:
                            keys.add(key)
                            places.append((B, oi, oj))
        if not places:
            step(bgc, sel_of(cells))        # a template that fits nowhere is removed
            continue

        places.sort(key=lambda p: abs(p[1] - r0) + abs(p[2] - c0))
        pose = ((r0, c0), np.where(A >= 0, A, 0), A >= 0)
        held = False
        for k, (B, oi, oj) in enumerate(places):
            last = k == len(places) - 1
            dest = [(oi + i, oj + j) for i, j in np.argwhere(B >= 0)]
            res = plan(pose, held, B, not last, frozenset(dest))
            if res is None:
                raise ValueError("template cannot be turned without losing content")
            nst, pose, held, path, lift = res
            st.clear(); st.update(nst)
            for op, sel in path:
                ops.append(op); sels.append(sel)
            lifted |= set(lift)
            pos = pose[0]
            zero_cells = [(oi + i, oj + j) for i, j in np.argwhere(B == 0)]
            if not last:
                # ---- leave a copy of the turned template on this triple -------
                saved, n_ops, before = _copy.deepcopy(st), len(ops), G().copy()
                step(29, sel_of(visible(pos, pose[2])))
                step(30, [oi, oj, 0, 0])
                if np.array_equal(before, G()):
                    st.clear(); st.update(saved)
                    del ops[n_ops:]; del sels[n_ops:]
                else:
                    held = False
                lifted -= set(dest)
                zc = [p for p in zero_cells if int(G()[p]) != 0]
                if zc:                        # colour 0 is not carried by a copy
                    step(0, sel_of(zc))
                    held = False
                continue
            # ---- the template itself slides onto the last triple --------------
            dr, dc = oi - pos[0], oj - pos[1]
            for op, n in ((21 if dr > 0 else 20, abs(dr)), (22 if dc > 0 else 23, abs(dc))):
                for _ in range(n):
                    if held:
                        step(op, sel_of([]))
                    else:
                        cur = visible(pos, pose[2])
                        lifted |= set(cur)
                        step(op, sel_of(cur))
                        held = True
            zc = [p for p in zero_cells if int(G()[p]) != 0]
            if zc:                            # colour 0 is not carried by a Move
                step(0, sel_of(zc))
            hole = sorted(p for p in lifted if int(G()[p]) == 0 and p not in set(dest))
            if bgc != 0 and hole:
                step(bgc, sel_of(hole))
        lifted = set()

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
