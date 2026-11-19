"""
ARC Task: 234bbc79 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of
from utils import *  # noqa: F401,F403  (unifint)
from dsl import *    # noqa: F401,F403  (width)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc = random.sample(cols, 2)
    return {"bgc": bgc, "dotc": dotc}


# ----------------------------------------------------------------------------- analysis
def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(sorted(cells))
    return comps


def _bg_colour(I):
    # the gaps between pieces are whole columns of background
    cnt = Counter(I.reshape(-1).tolist())
    uni = {int(I[0, c]) for c in range(I.shape[1]) if len(set(I[:, c].tolist())) == 1}
    if uni:
        return max(uni, key=lambda v: cnt[v])
    return cnt.most_common(1)[0][0]


def _marker_colour(I, bgc, comps, examples):
    pal = set(I.reshape(-1).tolist()) - {bgc}
    if examples:
        # the marker colour is the one every demonstration input has and no output keeps
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            s = set(ei.reshape(-1).tolist()) - set(eo.reshape(-1).tolist())
            cand = s if cand is None else cand & s
        cand = (cand or set()) & pal
        if len(cand) == 1:
            return cand.pop()
    cnt = Counter(I.reshape(-1).tolist())
    ok = [col for col in sorted(pal)
          if all(sum(1 for p in cells if I[p] == col) in (1, 2) for cells in comps)]
    if not ok:
        return None
    return min(ok, key=lambda col: (cnt[col], col))


def _analyse(I, examples=None):
    I = np.asarray(I, dtype=int)
    bgc = _bg_colour(I)
    comps = _components(I, bgc)
    if len(comps) < 2:
        return None
    comps.sort(key=lambda cells: min(c for _, c in cells))
    dotc = _marker_colour(I, bgc, comps, examples)
    if dotc is None:
        return None
    n = len(comps)
    pieces = []
    for k, cells in enumerate(comps):
        body = [int(I[p]) for p in cells if I[p] != dotc]
        if not body:
            return None
        bcol = Counter(body).most_common(1)[0][0]
        dots = [p for p in cells if I[p] == dotc]
        lm = rm = None
        if k > 0:
            lm = min(dots, key=lambda p: p[1])
        if k < n - 1:
            rm = max(dots, key=lambda p: p[1])
        pieces.append({"cells": list(cells), "col": bcol, "lm": lm, "rm": rm})
    # width of the reassembled snake = number of non-empty columns of I
    W = sum(1 for c in range(I.shape[1]) if any(I[:, c] != bgc))
    return {"bgc": bgc, "dotc": dotc, "pieces": pieces, "W": W}


def _reassemble(I, an):
    """Python-side model of the rule (used by generate() to reject bad draws)."""
    I = np.asarray(I, dtype=int)
    h = I.shape[0]
    P = an["pieces"]
    out = {}
    D = 0
    prev_rm = None
    prev_right = None
    for k, pc in enumerate(P):
        if k > 0:
            D = prev_rm[0] - pc["lm"][0]
        left = min(c for _, c in pc["cells"])
        dc = 0 if k == 0 else (prev_right + 1 - left)
        for (r, c) in pc["cells"]:
            v = int(I[r, c])
            if v == an["dotc"]:
                v = pc["col"]
            nr, ncl = r + D, c + dc
            if not (0 <= nr < h) or (nr, ncl) in out:
                return None
            out[(nr, ncl)] = v
        prev_right = max(c for _, c in pc["cells"]) + dc
        if pc["rm"] is not None:
            prev_rm = (pc["rm"][0] + D, pc["rm"][1] + dc)
    W = max(c for _, c in out) + 1
    g = np.full((h, W), an["bgc"], dtype=int)
    for (r, c), v in out.items():
        g[r, c] = v
    return g


# ----------------------------------------------------------------------------- generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None or dotc is None:
        bgc, dotc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, dotc)]
    Hmax = max(5, min(30, max_h))
    Wtop = min(30, max_w)
    for _attempt in range(2000):
        h = unifint(diff_lb, diff_ub, (5, Hmax))
        w = unifint(diff_lb, diff_ub, (6, max(6, min(20, Wtop - 1))))
        ncols = unifint(diff_lb, diff_ub, (1, 8))
        ccols = random.sample(remcols, ncols)
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        occ = {(spi, 0)}
        while True:
            pi, pj = snek[-1]
            if pj == w - 1 and random.choice((True, False, False)):
                break
            options = []
            if pi < h - 1 and (pi + 1, pj) not in occ:
                options.append((pi + 1, pj))
            if pi > 0 and (pi - 1, pj) not in occ:
                options.append((pi - 1, pj))
            if pj < w - 1:
                options.append((pi, pj + 1))
            if not options:
                break
            loc = random.choice(options)
            snek.append(loc)
            occ.add(loc)
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            if (len(cobj) > 2 and width(frozenset(cobj)) > 1
                    and snek[idx - 1] == (cel[0], cel[1] - 1)):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if not objs:
            continue
        objs[-1] += cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, nobjs))
        for _ in range(nobjs - ntokeep):
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        n = len(objs)
        segcols = [random.choice(ccols) for _ in objs]
        inobjs = []
        for idx, obj in enumerate(objs):
            cells = {}
            for j, p in enumerate(obj):
                if j == 0 and idx > 0:
                    cells[p] = dotc
                elif j == len(obj) - 1 and idx < n - 1:
                    cells[p] = dotc
                else:
                    cells[p] = segcols[idx]
            inobjs.append(cells)
        sw = max(c for _, c in snek) + 1
        fullw = unifint(diff_lb, diff_ub, (w, max(w, Wtop)))
        if fullw > Wtop:
            continue
        spacings = [1] * (n - 1)
        for _ in range(fullw - w - n - 1):
            spacings[random.randint(0, n - 2)] += 1
        need = sw + sum(spacings)
        if need > fullw:
            continue
        gi = [[bgc] * fullw for _ in range(h)]
        ofs = 0
        for i, cells in enumerate(inobjs):
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            mr, mc = min(rs), min(cs)
            hh, ww = max(rs) - mr + 1, max(cs) - mc + 1
            if i == 0:
                ulc = (mr, mc)
                lsp = 0
            else:
                lsp = spacings[i - 1]
                ulc = (random.randint(0, h - hh), ofs + lsp)
            ofs += ww + lsp
            for (r, c), v in cells.items():
                gi[ulc[0] + r - mr][ulc[1] + c - mc] = v
        go = [[bgc] * sw for _ in range(h)]
        for i, obj in enumerate(objs):
            for (r, c) in obj:
                go[r][c] = segcols[i]
        gi_np = np.array(gi, dtype=int)
        go_np = np.array(go, dtype=int)
        an = _analyse(gi_np, None)
        if an is None or an["bgc"] != bgc or an["dotc"] != dotc or len(an["pieces"]) != n:
            continue
        rec = _reassemble(gi_np, an)
        if rec is None or rec.shape != go_np.shape or not np.array_equal(rec, go_np):
            continue
        return {"input": tuple(tuple(int(v) for v in row) for row in gi),
                "output": tuple(tuple(int(v) for v in row) for row in go)}
    raise ValueError("234bbc79: generation failed")


# ----------------------------------------------------------------------------- trajectory
def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: walk the columns left to right; every run of
    non-empty columns is one piece, every empty column is dropped.  Each piece in turn is
    (1) raised/lowered so its left marker sits level with the previous piece's right marker,
    (2) joined on: the empty columns in front of it are deleted, i.e. it and everything to
        its right slides left as one block, and
    (3) the two markers that now touch take the colour of their own piece.
    Finally only the non-empty columns are kept."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    an = _analyse(I, examples)
    if an is None:
        return [34], [[0, 0, hi - 1, wi - 1]]
    bgc, W, P = an["bgc"], an["W"], an["pieces"]
    n = len(P)
    pos = [{p: p for p in pc["cells"]} for pc in P]

    # ARCLE's grab ignores 0-valued cells: give every 0 cell of a piece that will move a
    # stand-in colour for the journey
    zeros = [p for k in range(1, n) for p in P[k]["cells"] if int(I[p]) == 0]
    if zeros:
        present = set(I.reshape(-1).tolist())
        free = [c for c in range(1, 10) if c not in present]
        temp = free[0] if free else next(c for c in range(1, 10) if c != bgc)
        ops.append(temp); sels.append(sel_of(zeros))

    def slide(keys, dr, dc):
        src = [pos[m][p] for m, p in keys]
        seq = []
        if dr:
            seq += [(21 if dr > 0 else 20, (1 if dr > 0 else -1, 0))] * abs(dr)
        if dc:
            seq += [(22 if dc > 0 else 23, (0, 1 if dc > 0 else -1))] * abs(dc)
        cur = list(src)
        for i, (op, (sr, sc)) in enumerate(seq):
            ops.append(op)
            sels.append(sel_of(cur) if i == 0 else sel_of([]))   # grab once, then continue
            cur = [(r + sr, c + sc) for r, c in cur]
        for (m, p), q in zip(keys, cur):
            pos[m][p] = q
        # the grab left 0 on the footprint it vacated; lay background there
        # (only inside the columns that will be kept)
        hole = sorted(q for q in set(src) - set(cur) if q[1] < W)
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    for k in range(1, n):
        # (1) level piece k's left marker with piece k-1's right marker
        dr = pos[k - 1][P[k - 1]["rm"]][0] - pos[k][P[k]["lm"]][0]
        if dr:
            slide([(k, p) for p in P[k]["cells"]], dr, 0)
        # (2) delete the empty columns in front of piece k: the rest of the snake
        #     (pieces k..n-1) slides left as one block
        right_prev = max(c for _, c in pos[k - 1].values())
        g = min(c for _, c in pos[k].values()) - right_prev - 1
        if g > 0:
            slide([(m, p) for m in range(k, n) for p in P[m]["cells"]], 0, -g)
        # (3) the two touching markers become body cells of their own pieces
        a_pos = pos[k - 1][P[k - 1]["rm"]]
        b_pos = pos[k][P[k]["lm"]]
        if P[k - 1]["col"] == P[k]["col"]:
            ops.append(P[k]["col"]); sels.append(sel_of([a_pos, b_pos]))
        else:
            ops.append(P[k - 1]["col"]); sels.append(sel_of([a_pos]))
            ops.append(P[k]["col"]); sels.append(sel_of([b_pos]))
        # piece k is now in its final place: its 0-coloured body cells get their 0 back
        mk = {P[k]["lm"], P[k]["rm"]}
        back = [pos[k][p] for p in P[k]["cells"] if int(I[p]) == 0 and p not in mk]
        if back:
            ops.append(0); sels.append(sel_of(back))

    # keep only the non-empty columns (full rectangle crop)
    if W < wi:
        ops.append(33); sels.append([0, 0, hi - 1, W - 1])
    ops.append(34); sels.append([0, 0, hi - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 234bbc79"
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
                                f"for task 234bbc79"
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
                    f"Failed to build a complete episode for task 234bbc79 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"234bbc79-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
