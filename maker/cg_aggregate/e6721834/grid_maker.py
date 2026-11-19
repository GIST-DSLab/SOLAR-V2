"""
ARC Task: e6721834 (RE-ARC) — LLM-generated grid_maker
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

_MFS = ['identity', 'dmirror', 'cmirror', 'vmirror', 'hmirror', 'rot90', 'rot180', 'rot270']
_TRANSPOSING = {'dmirror', 'cmirror', 'rot90', 'rot270'}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return int(min(max(a, round(a + (b - a) * d)), b))


def _apply(fn, g):
    if fn == 'identity':
        return g
    if fn == 'dmirror':
        return g.T
    if fn == 'cmirror':
        return np.rot90(g, 2).T
    if fn == 'vmirror':
        return g[:, ::-1]
    if fn == 'hmirror':
        return g[::-1, :]
    if fn == 'rot90':
        return np.rot90(g, 3)
    if fn == 'rot180':
        return np.rot90(g, 2)
    return np.rot90(g, 1)


def _e67_split(I):
    # the two panels: count full uniform rows vs uniform columns (verifier's frontier rule)
    H, W = I.shape
    nh = sum(len(set(I[r].tolist())) == 1 for r in range(H))
    nv = sum(len(set(I[:, c].tolist())) == 1 for c in range(W))
    if nh > nv:
        halves = [(0, 0, H // 2, W), (H // 2, 0, H // 2, W)]
    else:
        halves = [(0, 0, H, W // 2), (0, W // 2, H, W // 2)]

    def ncol(p):
        r, c, h, w = p
        return len(np.unique(I[r:r + h, c:c + w]))
    halves.sort(key=ncol)
    return halves[0], halves[1]          # canvas (fewer colours), legend


def _e67_boxes(I):
    C, L = _e67_split(I)
    can = I[C[0]:C[0] + C[2], C[1]:C[1] + C[3]]
    leg = I[L[0]:L[0] + L[2], L[1]:L[1] + L[3]]
    bg1 = Counter(leg.ravel().tolist()).most_common(1)[0][0]
    lh, lw = leg.shape
    seen = np.zeros_like(leg, dtype=bool)
    comps = []
    for r in range(lh):
        for c in range(lw):
            if seen[r, c] or leg[r, c] == bg1:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < lh and 0 <= y < lw and not seen[x, y] and leg[x, y] != bg1:
                        seen[x, y] = True
                        stack.append((x, y))
            comps.append(cells)
    return C, L, can, leg, comps


def _e67_sq_counts(I):
    I = np.asarray(I, dtype=int)
    C, L, can, leg, comps = _e67_boxes(I)
    canc = set(can.ravel().tolist())
    cnt = Counter()
    for cells in comps:
        for a, b in cells:
            v = int(leg[a, b])
            if v not in canc:
                cnt[v] += 1
    return cnt


def _e67_plan(I, examples=None):
    I = np.asarray(I, dtype=int)
    C, L, can, leg, comps = _e67_boxes(I)
    # box fill colour: never appears on the canvas panel; pooled over the demos
    cnt = _e67_sq_counts(I)
    for ex in (examples or []):
        try:
            cnt += _e67_sq_counts(ex[0])
        except Exception:
            pass
    sqc = cnt.most_common(1)[0][0] if cnt else None
    bg2 = Counter(can.ravel().tolist()).most_common(1)[0][0]
    ch, cw = can.shape
    P = np.full((ch + 2, cw + 2), bg2, dtype=int)
    P[1:-1, 1:-1] = can
    stamps = []
    for cells in comps:
        rs = [a for a, _ in cells]
        cs = [b for _, b in cells]
        r0, c0 = min(rs), min(cs)
        rh, rw = max(rs) - r0 + 1, max(cs) - c0 + 1
        if rh * rw != len(cells):
            continue
        box = leg[r0:r0 + rh, c0:c0 + rw]
        nmarks = int((box != sqc).sum())
        if nmarks == 0 or nmarks == box.size:
            continue
        pat = np.full((rh + 2, rw + 2), bg2, dtype=int)
        pat[1:-1, 1:-1] = np.where(box == sqc, bg2, box)
        dest = None
        for i in range(ch - rh + 1):
            for j in range(cw - rw + 1):
                if np.array_equal(P[i:i + rh + 2, j:j + rw + 2], pat):
                    dest = (i, j)
                    break
            if dest:
                break
        if dest is None:
            continue
        stamps.append((dest[0], dest[1], L[0] + r0, L[1] + c0, rh, rw, nmarks))
    # richest pattern first (the box with the most marks is the least ambiguous match)
    stamps.sort(key=lambda s: (-s[6], s[2], s[3]))
    return C, sqc, stamps


def _e67_solve(I):
    I = np.asarray(I, dtype=int)
    C, sqc, stamps = _e67_plan(I)
    out = I[C[0]:C[0] + C[2], C[1]:C[1] + C[3]].copy()
    for dr, dc, sr, sc, rh, rw, _n in stamps:
        out[dr:dr + rh, dc:dc + rw] = I[sr:sr + rh, sc:sc + rw]
    return out


def sample_colors(num_examples=None) -> dict:
    bgc1, bgc2, sqc = random.sample(range(10), 3)
    return {"bgc1": bgc1, "bgc2": bgc2, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc1=None, bgc2=None, sqc=None, **kwargs) -> dict:
    if bgc1 is None or bgc2 is None or sqc is None:
        bgc1, bgc2, sqc = random.sample(range(10), 3)
    cols = list(range(10))
    remcols = [c for c in cols if c not in (bgc1, bgc2, sqc)]
    for _attempt in range(200):
        nmfs = random.choice((1, 2))
        fns = random.sample(_MFS, nmfs)
        transposed = sum(f in _TRANSPOSING for f in fns) % 2 == 1
        lim_long, lim_w = (max_w, max_h) if transposed else (max_h, max_w)
        h = _unifint(diff_lb, diff_ub, (6, max(6, min(15, lim_long // 2))))
        w = _unifint(diff_lb, diff_ub, (8, max(8, min(30, lim_w))))
        gi1 = np.full((h, w), bgc1, dtype=int)
        gi2 = np.full((h, w), bgc2, dtype=int)
        go = np.full((h, w), bgc2, dtype=int)
        noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
        allinds = {(i, j) for i in range(h) for j in range(w)}
        gi1inds = set(allinds)
        gi2inds = set(allinds)

        def nbh(cells):
            out = set()
            for a, b in cells:
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        out.add((a + da, b + db))
            return out
        tr = succ = 0
        maxtr = 5 * noccs
        seen = []
        while tr < maxtr and succ < noccs:
            tr += 1
            oh = random.randint(2, min(6, h // 2))
            ow = random.randint(2, min(6, w // 2))
            cands = [ij for ij in gi1inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loc = random.choice(cands)
            bounds = {(loc[0] + a, loc[1] + b) for a in range(oh) for b in range(ow)}
            ncells = _unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
            obj = set(random.sample(sorted(bounds), ncells))
            objc = random.choice(remcols)
            objn = frozenset((a - loc[0], b - loc[1]) for a, b in obj)
            if (objn, objc) in seen:
                continue
            seen.append((objn, objc))
            if bounds <= gi1inds:
                succ += 1
                gi1inds = gi1inds - nbh(bounds)
                for a, b in bounds:
                    gi1[a, b] = sqc
                for a, b in obj:
                    gi1[a, b] = objc
                cands2 = [ij for ij in gi2inds if ij[0] <= h - oh and ij[1] <= w - ow]
                if not cands2:
                    continue
                loc2 = random.choice(cands2)
                bounds2 = {(a - loc[0] + loc2[0], b - loc[1] + loc2[1]) for a, b in bounds}
                obj2 = {(a - loc[0] + loc2[0], b - loc[1] + loc2[1]) for a, b in obj}
                if bounds2 <= gi2inds:
                    gi2inds = gi2inds - nbh(bounds2)
                    for a, b in obj2:
                        gi2[a, b] = objc
                    for a, b in bounds2:
                        go[a, b] = sqc
                    for a, b in obj2:
                        go[a, b] = objc
        gi = np.vstack([gi1, gi2])
        for fn in fns:
            gi = _apply(fn, gi)
            go = _apply(fn, go)
        if gi.shape[0] > max_h or gi.shape[1] > max_w:
            continue
        try:
            if not np.array_equal(_e67_solve(gi), go):
                continue
        except Exception:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("could not generate a consistent e6721834 instance")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    ops, sels = [], []
    C, sqc, stamps = _e67_plan(I, examples)
    ch, cw = C[2], C[3]
    # Reading (golf entry): stamp every legend box onto its marks while both panels
    # are still on the board, the box with the most marks first, and only then
    # cut the board down to the canvas panel.
    for dr, dc, sr, sc, rh, rw, _n in stamps:
        R, Cc = C[0] + dr, C[1] + dc
        if sqc == 0:
            # Paste cannot write 0: lay the box fill down first on the exact
            # rectangle the box will occupy (full rectangle -> bbox).
            ops.append(0); sels.append([R, Cc, rh - 1, rw - 1])
        # the legend box is a full rectangle (fill + marks) -> bbox
        ops.append(28); sels.append([sr, sc, rh - 1, rw - 1])
        # paste over its marks on the canvas panel (marks are transparent-safe)
        ops.append(30); sels.append([R, Cc, 0, 0])
    # keep only the canvas panel (exact rectangle -> bbox)
    ops.append(33); sels.append([C[0], C[1], ch - 1, cw - 1])
    ops.append(34); sels.append([0, 0, ch - 1, cw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e6721834"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e6721834"
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
                                f"for task e6721834"
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
                    f"Failed to build a complete episode for task e6721834 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e6721834-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
