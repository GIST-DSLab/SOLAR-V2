"""
ARC Task: 97a05b5b (RE-ARC) — LLM-generated grid_maker
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
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def _nbrs(cells):
    out = set()
    for (i, j) in cells:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    out.add((i + di, j + dj))
    return out


def _p_hm(p):
    a = min(i for i, _ in p); b = max(i for i, _ in p)
    return {(a + b - i, j) for i, j in p}


def _p_vm(p):
    a = min(j for _, j in p); b = max(j for _, j in p)
    return {(i, a + b - j) for i, j in p}


def _p_dm(p):
    a = min(i for i, _ in p); b = min(j for _, j in p)
    return {(j - b + a, i - a + b) for i, j in p}


def _p_cm(p):
    return _p_vm(_p_dm(_p_vm(p)))


def _p_norm(p):
    a = min(i for i, _ in p); b = min(j for _, j in p)
    return frozenset((i - a, j - b) for i, j in p)


def _g_tf(name, g):
    if name == "id":
        return g
    if name == "dm":
        return g.T
    if name == "vm":
        return g[:, ::-1]
    if name == "hm":
        return g[::-1, :]
    return g[::-1, ::-1].T  # cmirror


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, **kw) -> dict:
    def unifint(lo, hi):
        lo = min(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    for _attempt in range(30):
        h = unifint(min(15, max_h), max_h)
        w = unifint(min(15, max_w), max_w)
        sgh = random.randint(h // 3, h // 3 * 2)
        sgw = random.randint(w // 3, w // 3 * 2)
        remcols = [c for c in range(10) if c not in (bgc, sqc)]
        gi = np.full((h, w), bgc, dtype=int)
        oh = random.randint(2, max(2, sgh // 2))
        ow = random.randint(2, max(2, sgw // 2))
        nobjs = unifint(1, 8)
        objs = []
        cands = {(i, j) for i in range(oh) for j in range(ow)}
        forbidden = set()
        tr = 0
        while len(objs) != nobjs and tr < 4 * nobjs:
            tr += 1
            obj = {random.choice(sorted(cands))}
            ncells = random.randint(1, oh * ow - 1)
            for _ in range(ncells - 1):
                opts = sorted((cands - obj) & _nbrs(obj))
                if not opts:
                    break
                obj.add(random.choice(opts))
            obj |= random.choice((_p_dm, _p_cm, _p_vm, _p_hm))(obj)
            hh = max(i for i, _ in obj) - min(i for i, _ in obj) + 1
            ww = max(j for _, j in obj) - min(j for _, j in obj) + 1
            if len(obj) == hh * ww:
                continue
            objn = _p_norm(obj)
            if objn not in forbidden and objn not in objs:
                objs.append(objn)
            fs = (lambda p: set(p), _p_dm, _p_cm, _p_hm, _p_vm)
            for f1 in fs:
                for f2 in fs:
                    forbidden.add(_p_norm(f1(f2(set(objn)))))
        loci = random.randint(0, h - sgh)
        locj = random.randint(0, w - sgw)
        gi[loci:loci + sgh, locj:locj + sgw] = sqc
        go = np.full((sgh, sgw), sqc, dtype=int)
        goinds = {(i, j) for i in range(sgh) for j in range(sgw)}
        rect = {(i + loci, j + locj) for i, j in goinds}
        giinds = {(i, j) for i in range(h) for j in range(w)} - rect
        giinds -= _nbrs(rect)
        succ = 0
        tr = 0
        while succ < nobjs and tr < 5 * nobjs and objs and remcols:
            tr += 1
            obj = random.choice(objs)
            col = random.choice(remcols)
            oh_ = max(i for i, _ in obj) + 1
            ow_ = max(j for _, j in obj) + 1
            subgi = np.full((oh_, ow_), col, dtype=int)
            for (i, j) in obj:
                subgi[i, j] = sqc
            if len(set(subgi.flatten().tolist())) == 1:
                continue
            f1 = random.choice(("id", "dm", "vm", "cm", "hm"))
            f2 = random.choice(("id", "dm", "vm", "cm", "hm"))
            subgo = _g_tf(f1, _g_tf(f2, subgi))
            ohi, owi = subgi.shape
            oho, owo = subgo.shape
            gocands = sorted(ij for ij in goinds if ij[0] <= sgh - oho and ij[1] <= sgw - owo)
            if not gocands:
                continue
            gl = random.choice(gocands)
            goplcdi = {(gl[0] + a, gl[1] + b) for a in range(oho) for b in range(owo)}
            if not goplcdi.issubset(goinds):
                continue
            gicands = sorted(ij for ij in giinds if ij[0] <= h - ohi and ij[1] <= owi)
            if not gicands:
                continue
            il = random.choice(gicands)
            giplcdi = {(il[0] + a, il[1] + b) for a in range(ohi) for b in range(owi)}
            if not giplcdi.issubset(giinds):
                continue
            succ += 1
            remcols.remove(col)
            objs.remove(obj)
            goinds -= goplcdi
            giinds = (giinds - giplcdi) - _nbrs(giplcdi)
            gi[il[0]:il[0] + ohi, il[1]:il[1] + owi] = subgi
            for a in range(oho):
                for b in range(owo):
                    if subgo[a, b] == sqc:
                        gi[loci + gl[0] + a, locj + gl[1] + b] = bgc
            go[gl[0]:gl[0] + oho, gl[1]:gl[1] + owo] = subgo
        if succ > 0:
            break
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, bgc):
    hi, wi = I.shape
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]; seen[r, c] = True; comp = []
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] != bgc:
                            seen[x, y] = True; stack.append((x, y))
            comps.append(comp)
    return comps


def _dihedral(g):
    out, seen = [], set()
    for k in range(4):
        for t in (np.rot90(g, k), np.rot90(g, k)[:, ::-1]):
            key = (t.shape, t.tobytes())
            if key not in seen:
                seen.add(key); out.append(np.array(t))
    return out


def _analyze(I, bgc, sqc=None):
    comps = _components(I, bgc)
    keys, rest = [], []
    for comp in comps:
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        cols = {int(I[p]) for p in comp}
        if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(cols) == 2:
            keys.append(I[r0:r1 + 1, c0:c1 + 1].copy())
        else:
            rest.extend(comp)
    if not rest or not keys:
        return None
    rcols = {int(I[p]) for p in rest}
    if len(rcols) != 1:
        return None
    s = rcols.pop()
    if sqc is not None and s != sqc:
        return None
    R0 = min(p[0] for p in rest); R1 = max(p[0] for p in rest)
    C0 = min(p[1] for p in rest); C1 = max(p[1] for p in rest)
    rect = I[R0:R1 + 1, C0:C1 + 1]
    if not set(np.unique(rect).tolist()) <= {s, bgc}:
        return None
    rh, rw = rect.shape
    holes = {(r, c) for r in range(rh) for c in range(rw) if rect[r, c] == bgc}
    # each key = (its colour, its rectangle-coloured mask); collect every orientation
    # of that mask that lands exactly on a hole in the rectangle
    kinfo = []
    for k in keys:
        if s not in np.unique(k):
            return None
        col = [int(v) for v in np.unique(k) if v != s][0]
        cands = []
        for t in _dihedral(k):
            th, tw = t.shape
            m = (t == s)
            for rr in range(rh - th + 1):
                for cc in range(rw - tw + 1):
                    win = rect[rr:rr + th, cc:cc + tw]
                    if np.all((win == bgc) == m):
                        hc = frozenset((rr + a, cc + b) for a in range(th) for b in range(tw) if m[a, b])
                        cands.append((rr, cc, t, col, hc))
        if not cands:
            return None
        kinfo.append(cands)
    order = sorted(range(len(kinfo)), key=lambda i: len(kinfo[i]))
    chosen = [None] * len(kinfo)

    def boxes_overlap(a, b):
        return not (a[0] + a[2].shape[0] <= b[0] or b[0] + b[2].shape[0] <= a[0]
                    or a[1] + a[2].shape[1] <= b[1] or b[1] + b[2].shape[1] <= a[1])

    def rec(idx, used):
        if idx == len(order):
            return used == holes
        for cand in kinfo[order[idx]]:
            if cand[4] & used:
                continue
            if any(chosen[j] is not None and boxes_overlap(cand, chosen[j]) for j in range(len(chosen))):
                continue
            chosen[order[idx]] = cand
            if rec(idx + 1, used | cand[4]):
                return True
            chosen[order[idx]] = None
        return False

    if not rec(0, frozenset()):
        return None
    placements = [(c[0], c[1], c[2], c[3]) for c in chosen]
    return R0, C0, R1, C1, s, placements


def _roles_from_examples(examples):
    if not examples:
        return None, None
    bg_votes, sq_cnt = Counter(), Counter()
    for EI, EO in examples:
        EI = np.asarray(EI); EO = np.asarray(EO)
        for c in set(EI.flatten().tolist()) - set(EO.flatten().tolist()):
            bg_votes[c] += 1
        sq_cnt.update(EO.flatten().tolist())
    bgc = bg_votes.most_common(1)[0][0] if bg_votes else None
    sqc = sq_cnt.most_common(1)[0][0] if sq_cnt else None
    return bgc, sqc


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Roles read from the demonstrations: background = the colour that never
    # survives into an output; rectangle colour = the colour the outputs are made of.
    ex_bg, ex_sq = _roles_from_examples(examples)
    info = None
    if ex_bg is not None:
        info = _analyze(I, ex_bg, ex_sq) or _analyze(I, ex_bg)
    if info is None:
        for cand_bg, _ in Counter(I.flatten().tolist()).most_common():
            info = _analyze(I, int(cand_bg))
            if info is not None:
                break

    ops, sels = [], []
    if info is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    R0, C0, R1, C1, sqc, placements = info
    rh, rw = R1 - R0 + 1, C1 - C0 + 1

    # 1. Keys have been read (remembered as shape masks); discard everything outside
    #    the big rectangle first. Selection is exactly the full rectangle, holes included.
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])

    # 2. In the cropped rectangle, each hole is matched to the key whose
    #    rectangle-coloured mask has that shape (in some orientation): fill the hole
    #    with the rectangle colour, then paint the key's colour on the rest of its footprint.
    for (rr, cc, t, col) in sorted(placements, key=lambda p: (p[0], p[1])):
        th, tw = t.shape
        hole = [(rr + a, cc + b) for a in range(th) for b in range(tw) if t[a, b] == sqc]
        body = [(rr + a, cc + b) for a in range(th) for b in range(tw) if t[a, b] != sqc]
        ops.append(int(sqc)); sels.append(sel_of(hole))
        ops.append(int(col)); sels.append(sel_of(body))

    ops.append(34); sels.append([0, 0, rh - 1, rw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
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
                                f"for task 97a05b5b"
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
                    f"Failed to build a complete episode for task 97a05b5b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97a05b5b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
