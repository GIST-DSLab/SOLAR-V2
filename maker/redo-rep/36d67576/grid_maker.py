"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


# ---------------------------------------------------------------- sample_colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # colour 0 means "empty" to ARCLE's clipboard, so keep it out of the shapes
    fgcols = [c for c in cols if c != bgc and c != 0]
    mainc, markerc = random.sample(fgcols, 2)
    remcols = [c for c in fgcols if c not in (mainc, markerc)]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "mainc": mainc, "markerc": markerc, "ccols": ccols}


# --------------------------------------------------------------- object helpers
def _neighbors(ij):
    i, j = ij
    return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)}


def _norm_cells(cells):
    mi = min(r for r, _ in cells)
    mj = min(c for _, c in cells)
    return {(r - mi, c - mj) for r, c in cells}


def _norm_obj(obj):
    mi = min(r for r, _ in obj)
    mj = min(c for _, c in obj)
    return {(r - mi, c - mj): v for (r, c), v in obj.items()}


def _obj_shape(obj):
    hh = max(r for r, _ in obj) - min(r for r, _ in obj) + 1
    ww = max(c for _, c in obj) - min(c for _, c in obj) + 1
    return hh, ww


def _t_identity(obj):
    return dict(obj)


def _t_hmirror(obj):          # up <-> down
    hh, _ = _obj_shape(obj)
    return {(hh - 1 - r, c): v for (r, c), v in obj.items()}


def _t_vmirror(obj):          # left <-> right
    _, ww = _obj_shape(obj)
    return {(r, ww - 1 - c): v for (r, c), v in obj.items()}


def _t_dmirror(obj):          # transpose
    return {(c, r): v for (r, c), v in obj.items()}


def _t_cmirror(obj):          # anti transpose
    return _t_vmirror(_t_hmirror(_t_dmirror(obj)))


def _t_rotcw(obj):
    hh, _ = _obj_shape(obj)
    return {(c, hh - 1 - r): v for (r, c), v in obj.items()}


# ------------------------------------------------------------------- generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, mainc, markerc, ccols) -> dict:
    hmax = max(10, min(int(max_h), 30))
    wmax = max(10, min(int(max_w), 30))
    mirrors = [_t_identity, _t_dmirror, _t_cmirror, _t_hmirror, _t_vmirror]
    while True:
        h = _unifint(diff_lb, diff_ub, (10, hmax))
        w = _unifint(diff_lb, diff_ub, (10, wmax))
        oh = _unifint(diff_lb, diff_ub, (2, 5))
        ow = _unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = _unifint(diff_lb, diff_ub, (4, len(bounds)))
        cells = {random.choice(sorted(bounds))}
        for _ in range(ncells - 1):
            nbrs = set()
            for ij in cells:
                nbrs |= _neighbors(ij)
            cands = sorted((bounds - cells) & nbrs)
            if not cands:
                break
            cells.add(random.choice(cands))
        cells = _norm_cells(cells)
        ncells = len(cells)
        if ncells < 4:
            continue
        ntocompc = _unifint(diff_lb, diff_ub, (1, max(1, ncells - 3)))
        markercell = random.choice(sorted(cells))
        remobj = sorted(cells - {markercell})
        ntocompc = max(1, min(ntocompc, len(remobj) - 1))
        tocompc = set(random.sample(remobj, ntocompc))
        obj = {}
        for ij in cells - {markercell} - tocompc:
            obj[ij] = mainc
        obj[markercell] = markerc
        for ij in tocompc:
            obj[ij] = random.choice(ccols)
        objn = _norm_obj(obj)
        smobjn = _norm_obj({ij: v for ij, v in objn.items() if v in (mainc, markerc)})
        fake = False
        for mf in mirrors[1:]:
            if _norm_obj(mf(smobjn)) == smobjn and _norm_obj(mf(objn)) != objn:
                fake = True
                break
        if fake:                       # partial shape would not fix the orientation
            continue
        ooh, oow = _obj_shape(objn)
        if ooh > h or oow > w:
            continue

        gi = [[bgc] * w for _ in range(h)]
        loci = random.randint(0, h - ooh)
        locj = random.randint(0, w - oow)
        plcd = {(r + loci, c + locj): v for (r, c), v in objn.items()}
        for (r, c), v in plcd.items():
            gi[r][c] = v
        blocked = set(plcd)
        for ij in list(plcd):
            blocked |= _neighbors(ij)
        inds = {(r, c) for r in range(h) for c in range(w)} - blocked
        go = [row[:] for row in gi]
        noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(objn)))))
        succ, tr, maxtr = 0, 0, noccs * 5
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = random.choice(mirrors)
            mf2 = random.choice(mirrors)
            outobj = _norm_obj(mf1(_norm_obj(mf2(objn))))
            aoh, aow = _obj_shape(outobj)
            cands = sorted(ij for ij in inds if ij[0] <= h - aoh and ij[1] <= w - aow)
            if not cands:
                continue
            loc = random.choice(cands)
            outobjp = {(r + loc[0], c + loc[1]): v for (r, c), v in outobj.items()}
            if set(outobjp).issubset(inds):
                succ += 1
                newblocked = set(outobjp)
                for ij in list(outobjp):
                    newblocked |= _neighbors(ij)
                inds -= newblocked
                for (r, c), v in outobjp.items():
                    go[r][c] = v
                    if v in (mainc, markerc):
                        gi[r][c] = v
        if succ == 0:
            continue
        return {"input": gi, "output": go}


# --------------------------------------------------------- ARCLE mini-simulator
def _sel_cells(sel, shape):
    hh, ww = shape
    if isinstance(sel, dict):
        cells = [(int(r), int(c)) for r, c in sel["cells"]]
    else:
        r, c, dh, dw = sel
        cells = [(i, j) for i in range(r, r + dh + 1) for j in range(c, c + dw + 1)]
    return [(r, c) for r, c in cells if 0 <= r < hh and 0 <= c < ww]


def _rot_pos(rmin, rmax, cmin, cmax, k):
    """Where ARCLE drops a rotated selection (ports arcle.actions.object.gen_rotate)."""
    oh = rmax - rmin + 1
    ow = cmax - cmin + 1
    cx = (rmax + rmin) * 0.5
    cy = (cmax + cmin) * 0.5
    if oh % 2 == ow % 2:
        return (int(np.floor(cx - cy + cmin)), int(np.floor(cy - cx + rmin)))
    parity = k % 2
    sig = (k + 2) % 4 - 2
    mod = 1 - parity
    mx = min(cx + sig * (cy - cmin), cx + sig * (cy - cmax)) + mod
    my = min(cy - sig * (cx - rmin), cy - sig * (cx - rmax)) + mod
    return (int(np.floor(mx)), int(np.floor(my)))


def _sim_step(state, op, sel):
    g = state["grid"]
    hh, ww = g.shape
    cells = _sel_cells(sel, (hh, ww))
    if not cells:
        return
    op = int(op)
    if op < 10:
        for r, c in cells:
            g[r, c] = op
        return
    rmin = min(r for r, _ in cells)
    rmax = max(r for r, _ in cells)
    cmin = min(c for _, c in cells)
    cmax = max(c for _, c in cells)
    if op in (28, 29):
        src = (state["input"] if op == 28 else g)[rmin:rmax + 1, cmin:cmax + 1]
        mask = np.zeros(src.shape, dtype=bool)
        for r, c in cells:
            mask[r - rmin, c - cmin] = True
        state["clip"] = np.where(mask & (src > 0), src, 0)
        return
    if op == 30:
        clip = state["clip"]
        if clip is None or clip.size == 0:
            return
        ch, cw = clip.shape
        for i in range(ch):
            for j in range(cw):
                r, c = rmin + i, cmin + j
                if clip[i, j] > 0 and 0 <= r < hh and 0 <= c < ww:
                    g[r, c] = clip[i, j]
        return
    if op in (24, 25, 26, 27):
        oh = rmax - rmin + 1
        ow = cmax - cmin + 1
        obj = np.zeros((oh, ow), dtype=int)
        for r, c in cells:
            obj[r - rmin, c - cmin] = g[r, c]
        bg = g.copy()
        for r, c in cells:
            bg[r, c] = 0
        if op in (26, 27):
            obj = np.fliplr(obj) if op == 26 else np.flipud(obj)
            pos = (rmin, cmin)
        else:
            k = 1 if op == 24 else 3
            pos = _rot_pos(rmin, rmax, cmin, cmax, k)
            obj = np.rot90(obj, k=k)
        g[:, :] = bg
        ph, pw = obj.shape
        x, y = pos
        for i in range(ph):
            for j in range(pw):
                r, c = x + i, y + j
                if obj[i, j] > 0 and 0 <= r < hh and 0 <= c < ww:
                    g[r, c] = obj[i, j]
        return


def _components(grid, bgc):
    hh, ww = grid.shape
    seen = np.zeros((hh, ww), dtype=bool)
    comps = []
    for r in range(hh):
        for c in range(ww):
            if seen[r, c] or grid[r, c] == bgc:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = {}
            while stack:
                rr, cc = stack.pop()
                comp[(rr, cc)] = int(grid[rr, cc])
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hh and 0 <= nc < ww and not seen[nr, nc] and grid[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


# ------------------------------------------------------------ derive_operations
def derive_operations(I, O):
    """Every incomplete copy is completed by stamping the template and turning /
    mirroring that stamp into the copy's orientation: CopyI + Paste + Rotate/Flip."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    full_sel = [0, 0, O.shape[0] - 1, O.shape[1] - 1]
    if np.array_equal(I, O):
        return [34], [full_sel]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps_I = _components(I, bgc)
    comps_O = _components(O, bgc)
    if not comps_I:
        return [34], [full_sel]

    # the complete template: the biggest connected shape in the input
    tmpl = max(comps_I, key=len)
    tr = min(r for r, _ in tmpl)
    tc = min(c for _, c in tmpl)
    th = max(r for r, _ in tmpl) - tr + 1
    tw = max(c for _, c in tmpl) - tc + 1
    tmpl_clean = all(
        (r, c) in tmpl
        for r in range(tr, tr + th) for c in range(tc, tc + tw)
        if I[r, c] != bgc
    )
    N = _norm_obj({(r - tr, c - tc): v for (r, c), v in tmpl.items()})
    rn = _norm_obj(_t_rotcw(N))
    variants = [                                   # cheapest reorientation first
        ([], N),
        ([26], _norm_obj(_t_vmirror(N))),
        ([27], _norm_obj(_t_hmirror(N))),
        ([26, 27], _norm_obj(_t_hmirror(_t_vmirror(N)))),
        ([25], rn),
        ([25, 26], _norm_obj(_t_vmirror(rn))),
        ([25, 27], _norm_obj(_t_hmirror(rn))),
        ([25, 26, 27], _norm_obj(_t_hmirror(_t_vmirror(rn)))),
    ]

    dests = []
    for comp in comps_O:
        if all(I[r, c] == v for (r, c), v in comp.items()):
            continue                               # already complete (the template)
        dr = min(r for r, _ in comp)
        dc = min(c for _, c in comp)
        dh = max(r for r, _ in comp) - dr + 1
        dw = max(c for _, c in comp) - dc + 1
        nrm = {(r - dr, c - dc): v for (r, c), v in comp.items()}
        extra = None
        for ex, tobj in variants:
            if tobj == nrm:
                extra = ex
                break
        dests.append({"r": dr, "c": dc, "h": dh, "w": dw, "cells": comp, "extra": extra})
    dests.sort(key=lambda d: (d["r"], d["c"]))

    state = {"grid": I.copy(), "input": I.copy(), "clip": None}

    def trial(seq_ops, seq_sels):
        st = {"grid": state["grid"].copy(), "input": state["input"], "clip": state["clip"]}
        for o, s in zip(seq_ops, seq_sels):
            _sim_step(st, o, s)
        return st["grid"]

    def commit(seq_ops, seq_sels):
        for o, s in zip(seq_ops, seq_sels):
            _sim_step(state, o, s)
        ops.extend(seq_ops)
        sels.extend(seq_sels)

    copied = False
    for d in dests:
        dr, dc, dh, dw = d["r"], d["c"], d["h"], d["w"]
        expected = state["grid"].copy()
        for (r, c), v in d["cells"].items():
            expected[r, c] = v
        done = False
        if d["extra"] is not None and tmpl_clean:
            extra = d["extra"]
            rot = 25 in extra
            flips = [o for o in extra if o in (26, 27)]
            placements = []
            if not rot:
                placements.append((dr, dc, dr, dc, th, tw))
            else:
                # turn the stamped rectangle itself
                for pr in range(0, H - th + 1):
                    for pc in range(0, W - tw + 1):
                        if _rot_pos(pr, pr + th - 1, pc, pc + tw - 1, 3) == (dr, dc):
                            placements.append((pr, pc, pr, pc, th, tw))
                # or turn a square patch that holds the stamp
                s = max(th, tw)
                for sr in range(max(0, dr + dh - s), min(dr, H - s) + 1):
                    for sc in range(max(0, dc + dw - s), min(dc, W - s) + 1):
                        a = s - th - (dc - sc)
                        b = dr - sr
                        if 0 <= a <= s - th and 0 <= b <= s - tw:
                            placements.append((sr + a, sc + b, sr, sc, s, s))
            for (pr, pc, sr, sc, sh, sw) in placements:
                if pr < 0 or pc < 0 or pr + th > H or pc + tw > W:
                    continue
                if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
                    continue
                cand_ops, cand_sels = [], []
                if not copied:
                    cand_ops.append(28)                        # grab the template
                    cand_sels.append([tr, tc, th - 1, tw - 1])
                region = [(r, c) for r in range(sr, sr + sh) for c in range(sc, sc + sw)]
                cand_ops.append(int(bgc))                      # clear the landing area
                cand_sels.append(sel_of(region))
                cand_ops.append(30)                            # stamp the template
                cand_sels.append([pr, pc, 0, 0])
                if rot:
                    cand_ops.append(25)                        # turn it
                    cand_sels.append([sr, sc, sh - 1, sw - 1])
                for f in flips:                                # mirror it
                    cand_ops.append(f)
                    cand_sels.append([dr, dc, dh - 1, dw - 1])
                vacated = [rc for rc in region if rc not in d["cells"]]
                if vacated:
                    cand_ops.append(int(bgc))                  # restore what it left behind
                    cand_sels.append(sel_of(vacated))
                if not np.array_equal(trial(cand_ops, cand_sels), expected):
                    continue                                   # region not free — try next
                keep = [True] * len(cand_ops)
                for i in range(len(cand_ops) - 1, -1, -1):
                    if cand_ops[i] == 28:
                        continue
                    t_ops = [o for j, o in enumerate(cand_ops) if keep[j] and j != i]
                    t_sels = [s for j, s in enumerate(cand_sels) if keep[j] and j != i]
                    if np.array_equal(trial(t_ops, t_sels), expected):
                        keep[i] = False
                fin_ops = [o for j, o in enumerate(cand_ops) if keep[j]]
                fin_sels = [s for j, s in enumerate(cand_sels) if keep[j]]
                if 30 not in fin_ops:
                    continue
                commit(fin_ops, fin_sels)
                copied = True
                done = True
                break
        if not done:
            # no room to stamp here: draw this copy's missing cells, colour by colour
            bycol = {}
            for (r, c), v in sorted(d["cells"].items()):
                if state["grid"][r, c] != v:
                    bycol.setdefault(int(v), []).append((r, c))
            seq_ops, seq_sels = [], []
            for col, cs in sorted(bycol.items()):
                seq_ops.append(col)
                seq_sels.append(sel_of(sorted(set(cs))))
            commit(seq_ops, seq_sels)

    rest = {}
    for r in range(H):
        for c in range(W):
            if state["grid"][r, c] != O[r, c]:
                rest.setdefault(int(O[r, c]), []).append((r, c))
    for col, cs in sorted(rest.items()):
        ops.append(col)
        sels.append(sel_of(cs))
        _sim_step(state, col, sel_of(cs))

    # drop any step that a later step makes pointless
    def replay(seq_ops, seq_sels, start_state, start):
        st = {"grid": start_state["grid"].copy(), "input": I,
              "clip": None if start_state["clip"] is None else start_state["clip"].copy()}
        for o, s in zip(seq_ops[start:], seq_sels[start:]):
            _sim_step(st, o, s)
        return st["grid"]

    def snapshots(seq_ops, seq_sels):
        st = {"grid": I.copy(), "input": I, "clip": None}
        snaps = [{"grid": st["grid"].copy(), "clip": None}]
        for o, s in zip(seq_ops, seq_sels):
            _sim_step(st, o, s)
            snaps.append({"grid": st["grid"].copy(),
                          "clip": None if st["clip"] is None else st["clip"].copy()})
        return snaps

    snaps = snapshots(ops, sels)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops):
            red_ops = ops[:i] + ops[i + 1:]
            red_sels = sels[:i] + sels[i + 1:]
            if np.array_equal(replay(red_ops, red_sels, snaps[i], i), O):
                ops, sels = red_ops, red_sels
                snaps = snapshots(ops, sels)
                changed = True
            else:
                i += 1

    ops.append(34)
    sels.append(full_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
