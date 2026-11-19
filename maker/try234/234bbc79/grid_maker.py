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
import numpy as np
from collections import Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in sorted(set(cells))]}


def sample_colors(num_examples=None) -> dict:
    bgc, dotc = random.sample(list(range(10)), 2)
    return {"bgc": bgc, "dotc": dotc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = max(a, min(b, int(a + (b - a) * diff_lb)))
    hi = max(a, min(b, int(a + (b - a) * diff_ub)))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _comps(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    out = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append(sorted(cells))
    return out


def _chain(I, examples=None):
    """Read the scattered snake pieces from I (plus the marker colour convention
    from the demonstrations).  Returns pieces sorted left->right with their head
    (left) / tail (right) marker cells and body colour."""
    I = np.asarray(I, dtype=int)
    cnt = Counter(I.reshape(-1).tolist())
    bgc = cnt.most_common(1)[0][0]
    comps = _comps(I, bgc)
    if len(comps) < 2:
        return None
    comps.sort(key=lambda cs: min(c for _, c in cs))
    present = set(cnt) - {bgc}
    dotc = None
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            s = set(ei.reshape(-1).tolist()) - set(eo.reshape(-1).tolist())
            cand = s if cand is None else cand & s
        cand = (cand or set()) & present
        if len(cand) == 1:
            dotc = cand.pop()
    if dotc is None:
        cands = [col for col in present
                 if all(sum(1 for p in cs if I[p] == col) in (1, 2) for cs in comps)]
        if not cands:
            return None
        dotc = min(cands, key=lambda col: (cnt[col], col))
    n = len(comps)
    pieces = []
    for k, cs in enumerate(comps):
        dots = sorted([p for p in cs if I[p] == dotc], key=lambda p: (p[1], p[0]))
        body = [p for p in cs if I[p] != dotc]
        if not body:
            return None
        need = 1 if k in (0, n - 1) else 2
        if len(dots) != need:
            return None
        head = dots[0] if k > 0 else None
        tail = dots[-1] if k < n - 1 else None
        pieces.append({"cells": cs, "head": head, "tail": tail,
                       "col": int(Counter(int(I[p]) for p in body).most_common(1)[0][0])})
    # horizontal gap before each piece, vertical offset of each piece
    hs, vs = [0], [0]
    for k in range(1, n):
        t, hd = pieces[k - 1]["tail"], pieces[k]["head"]
        gap = hd[1] - t[1] - 1
        if gap < 1:
            return None
        hs.append(gap)
        vs.append(0)
    vcum = 0
    for k in range(1, n):
        t, hd = pieces[k - 1]["tail"], pieces[k]["head"]
        vcum_prev = vcum
        vs[k] = (t[0] + vcum_prev) - hd[0]
        vcum = vs[k]
    W = max(c for _, c in pieces[-1]["cells"]) - sum(hs) + 1
    return {"bgc": int(bgc), "dotc": int(dotc), "pieces": pieces,
            "gaps": hs, "drs": vs, "W": int(W)}


def _render(I, ch):
    """Python-side reconstruction (used only by generate() to check an instance)."""
    I = np.asarray(I, dtype=int)
    h = I.shape[0]
    out = np.full((h, ch["W"]), ch["bgc"], dtype=int)
    shift = 0
    for k, pc in enumerate(ch["pieces"]):
        shift += ch["gaps"][k]
        dr = ch["drs"][k]
        for (r, c) in pc["cells"]:
            nr, nc = r + dr, c - shift
            if not (0 <= nr < h and 0 <= nc < ch["W"]):
                return None
            out[nr, nc] = pc["col"] if I[r, c] == ch["dotc"] else I[r, c]
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, **kw) -> dict:
    if bgc is None or dotc is None:
        bgc, dotc = random.sample(list(range(10)), 2)
    cols = [c for c in range(10) if c not in (bgc, dotc)]
    hmax = max(5, min(30, int(max_h)))
    wtop = min(30, int(max_w))
    for _ in range(5000):
        h = _unifint(diff_lb, diff_ub, (5, hmax))
        w = _unifint(diff_lb, diff_ub, (6, max(6, min(20, wtop - 1))))
        ncols = _unifint(diff_lb, diff_ub, (1, 8))
        ccols = random.sample(cols, ncols)
        spi = random.randint(0, h - 1)
        snek = [(spi, 0)]
        occ = {(spi, 0)}
        while True:
            pi, pj = snek[-1]
            if pj == w - 1 and random.choice((True, False, False)):
                break
            opts = []
            if pi < h - 1 and (pi + 1, pj) not in occ:
                opts.append((pi + 1, pj))
            if pi > 0 and (pi - 1, pj) not in occ:
                opts.append((pi - 1, pj))
            if pj < w - 1:
                opts.append((pi, pj + 1))
            if not opts:
                break
            loc = random.choice(opts)
            snek.append(loc)
            occ.add(loc)
        objs, cobj = [], []
        for idx, cel in enumerate(snek):
            cw = (max(c for _, c in cobj) - min(c for _, c in cobj)) if cobj else 0
            if len(cobj) > 2 and cw > 0 and snek[idx - 1] == (cel[0], cel[1] - 1):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        if not objs:
            continue
        objs[-1] = objs[-1] + cobj
        if len(objs) < 2:
            continue
        ntokeep = _unifint(diff_lb, diff_ub, (2, len(objs)))
        for _k in range(len(objs) - ntokeep):
            idx = random.randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx + 1]] + objs[idx + 2:]
        n = len(objs)
        sw = max(c for _, c in snek) + 1
        minw = sw + n - 1
        if minw > wtop:
            continue
        fullw = _unifint(diff_lb, diff_ub, (max(w, minw), wtop))
        spacings = [1] * (n - 1)
        for _k in range(fullw - minw):
            i = random.randint(0, n - 1)
            if i < n - 1:
                spacings[i] += 1
        segcols = [random.choice(ccols) for _ in objs]
        gi = [[bgc] * fullw for _ in range(h)]
        go = [[bgc] * sw for _ in range(h)]
        nxt = 0
        bad = False
        for i, ob in enumerate(objs):
            cells = {}
            for j, p in enumerate(ob):
                go[p[0]][p[1]] = segcols[i]
                if (j == 0 and i > 0) or (j == len(ob) - 1 and i < n - 1):
                    cells[p] = dotc
                else:
                    cells[p] = segcols[i]
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            mr, mc = min(rs), min(cs)
            hh, ww = max(rs) - mr + 1, max(cs) - mc + 1
            if i == 0:
                ur, uc = mr, mc
            else:
                ur = random.randint(0, h - hh)
                uc = nxt + spacings[i - 1]
            if uc + ww > fullw:
                bad = True
                break
            nxt = uc + ww
            for (pr, pc), v in cells.items():
                gi[ur + pr - mr][uc + pc - mc] = v
        if bad:
            continue
        giA, goA = np.array(gi, dtype=int), np.array(go, dtype=int)
        cnt = Counter(giA.reshape(-1).tolist()).most_common(2)
        if cnt[0][0] != bgc or (len(cnt) > 1 and cnt[1][1] == cnt[0][1]):
            continue
        ch = _chain(giA)
        if ch is None or ch["dotc"] != dotc:
            continue
        rec = _render(giA, ch)
        if rec is None or rec.shape != goA.shape or not np.array_equal(rec, goA):
            continue
        return {"input": gi, "output": go}
    raise ValueError("234bbc79: generation failed")


def derive_operations(I, O, examples=None):
    """Re-assemble the snake axis by axis.

    Phase 1 (columns): walking the gaps left to right, grab EVERYTHING to the
    right of the gap -- the whole rest of the chain as one object -- and slide
    it left until the gap is closed.
    Phase 2 (rows): with the pieces now packed side by side (disjoint column
    bands), slide each piece up/down until its left marker sits beside the
    previous piece's right marker, then weld the two markers into body cells.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    ch = _chain(I, examples)
    if ch is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    bgc, W = ch["bgc"], ch["W"]
    pieces = [dict(p) for p in ch["pieces"]]
    n = len(pieces)
    present = set(I.reshape(-1).tolist())

    # ARCLE object grabs drop colour-0 cells: give literal-0 content of the
    # pieces that will move a stand-in colour first.
    zeros = sorted(p for pc in pieces[1:] for p in pc["cells"] if I[p] == 0)
    temp = None
    if zeros:
        temp = next(c for c in range(1, 10) if c not in present)
        ops.append(temp); sels.append(sel_of(zeros))

    def slide(cells, dr, dc):
        cur = list(cells)
        steps = [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr) + \
                [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
        for i, (op, (sr, sc)) in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(cur) if i == 0 else sel_of([]))
            cur = [(r + sr, c + sc) for r, c in cur]
        hole = sorted(p for p in set(cells) - set(cur) if p[1] < W)
        if steps and bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

    def shift_piece(pc, dr, dc):
        pc["cells"] = [(r + dr, c + dc) for r, c in pc["cells"]]
        for key in ("head", "tail"):
            if pc[key] is not None:
                pc[key] = (pc[key][0] + dr, pc[key][1] + dc)

    # Phase 1: close every gap by pulling the rest of the chain leftwards
    for k in range(1, n):
        gap = ch["gaps"][k]
        group = [p for pc in pieces[k:] for p in pc["cells"]]
        slide(group, 0, -gap)
        for pc in pieces[k:]:
            shift_piece(pc, 0, -gap)

    # Phase 2: line each piece up vertically, then weld the joint
    for k in range(1, n):
        prev, pc = pieces[k - 1], pieces[k]
        dr = prev["tail"][0] - pc["head"][0]
        slide(pc["cells"], dr, 0)
        shift_piece(pc, dr, 0)
        if prev["col"] == pc["col"]:
            ops.append(pc["col"]); sels.append(sel_of([prev["tail"], pc["head"]]))
        else:
            ops.append(prev["col"]); sels.append(sel_of([prev["tail"]]))
            ops.append(pc["col"]); sels.append(sel_of([pc["head"]]))

    if temp is not None:
        # put the literal-0 body cells back (markers were already welded)
        back = sorted(p for k, pc in enumerate(pieces) if k > 0 and pc["col"] == 0
                      for p in pc["cells"] if p != pc["head"])
        if back:
            ops.append(0); sels.append(sel_of(back))

    if W != wi:
        ops.append(33); sels.append([0, 0, hi - 1, W - 1])   # keep full left band of width W
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
