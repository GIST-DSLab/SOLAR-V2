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


def _e67_halves(I):
    # Two equal panels glued together. On the true split each panel's background
    # is absent from the other panel.
    H, W = I.shape
    cands = []
    if H % 2 == 0:
        cands.append([(0, 0, H // 2, W), (H // 2, 0, H // 2, W)])
    if W % 2 == 0:
        cands.append([(0, 0, H, W // 2), (0, W // 2, H, W // 2)])
    best = None
    for pair in cands:
        a, b = [I[r:r + h, c:c + w] for r, c, h, w in pair]
        ca, cb = set(a.ravel().tolist()), set(b.ravel().tolist())
        ma = Counter(a.ravel().tolist()).most_common(1)[0][0]
        mb = Counter(b.ravel().tolist()).most_common(1)[0][0]
        score = int(ma not in cb) + int(mb not in ca)
        if best is None or score > best[0]:
            best = (score, pair)
    pair = best[1]
    # canvas = panel with fewer colours (bg + marks)
    pair = sorted(pair, key=lambda p: len(np.unique(I[p[0]:p[0] + p[2], p[1]:p[1] + p[3]])))
    return pair[0], pair[1]


def _e67_panels(I):
    C, L = _e67_halves(I)
    can = I[C[0]:C[0] + C[2], C[1]:C[1] + C[3]]
    leg = I[L[0]:L[0] + L[2], L[1]:L[1] + L[3]]
    return C, L, can, leg


def _e67_roles(I, examples=None):
    # legend bg and box fill: colours present in every legend panel, absent from
    # every canvas panel; commonest = legend bg, next = box fill.
    grids = [np.asarray(I, dtype=int)] + [np.asarray(e[0], dtype=int) for e in (examples or [])]
    present, absent, tot = None, set(), Counter()
    for g in grids:
        try:
            _, _, can, leg = _e67_panels(g)
        except Exception:
            continue
        lc = set(leg.ravel().tolist())
        present = lc if present is None else present & lc
        absent |= set(can.ravel().tolist())
        tot += Counter(leg.ravel().tolist())
    S = [c for c in (present or set()) if c not in absent]
    S.sort(key=lambda c: -tot[c])
    if len(S) < 2:
        S = [c for c, _ in tot.most_common()]
    return S[0], S[1]


def _e67_plan(I, examples=None):
    I = np.asarray(I, dtype=int)
    C, L, can, leg = _e67_panels(I)
    lbg, sqc = _e67_roles(I, examples)
    cbg = Counter(can.ravel().tolist()).most_common(1)[0][0]
    lh, lw = leg.shape
    seen = np.zeros(leg.shape, dtype=bool)
    boxes = []
    for r in range(lh):
        for c in range(lw):
            if seen[r, c] or leg[r, c] == lbg:
                continue
            st = [(r, c)]; seen[r, c] = True; cells = []
            while st:
                a, b = st.pop(); cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < lh and 0 <= y < lw and not seen[x, y] and leg[x, y] != lbg:
                            seen[x, y] = True; st.append((x, y))
            r0 = min(a for a, _ in cells); c0 = min(b for _, b in cells)
            rh = max(a for a, _ in cells) - r0 + 1; rw = max(b for _, b in cells) - c0 + 1
            if rh * rw != len(cells):
                continue
            box = leg[r0:r0 + rh, c0:c0 + rw]
            nm = int((box != sqc).sum())
            if nm == 0 or nm == box.size:
                continue
            boxes.append((nm, r0, c0, rh, rw, box))
    # most marks first; each stamp updates the canvas so later (smaller) patterns
    # cannot match inside an already-stamped box.
    boxes.sort(key=lambda t: (-t[0], -t[3] * t[4], t[1], t[2]))
    ch, cw = can.shape
    cur = can.copy()
    stamps = []
    for nm, r0, c0, rh, rw, box in boxes:
        want = np.where(box == sqc, cbg, box)
        for i in range(ch - rh + 1):
            for j in range(cw - rw + 1):
                if not np.array_equal(cur[i:i + rh, j:j + rw], want):
                    continue
                ok = True
                for a in range(i - 1, i + rh + 1):
                    for b in range(j - 1, j + rw + 1):
                        if i <= a < i + rh and j <= b < j + rw:
                            continue
                        if 0 <= a < ch and 0 <= b < cw and cur[a, b] != cbg:
                            ok = False
                if ok:
                    cur[i:i + rh, j:j + rw] = box
                    stamps.append((i, j, L[0] + r0, L[1] + c0, rh, rw))
    return C, sqc, stamps, cur


def sample_colors(num_examples=None) -> dict:
    bgc1, bgc2, sqc = random.sample(range(10), 3)
    return {"bgc1": bgc1, "bgc2": bgc2, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc1=None, bgc2=None, sqc=None, **kwargs) -> dict:
    if bgc1 is None or bgc2 is None or sqc is None:
        bgc1, bgc2, sqc = random.sample(range(10), 3)
    remcols = [c for c in range(10) if c not in (bgc1, bgc2, sqc)]

    def nbh(cells):
        return {(a + da, b + db) for a, b in cells for da in (-1, 0, 1) for db in (-1, 0, 1)}
    for _attempt in range(200):
        fns = random.sample(_MFS, random.choice((1, 2)))
        transposed = sum(f in _TRANSPOSING for f in fns) % 2 == 1
        lim_long, lim_w = (max_w, max_h) if transposed else (max_h, max_w)
        h = _unifint(diff_lb, diff_ub, (6, max(6, min(15, lim_long // 2))))
        w = _unifint(diff_lb, diff_ub, (8, max(8, min(30, lim_w))))
        gi1 = np.full((h, w), bgc1, dtype=int)
        gi2 = np.full((h, w), bgc2, dtype=int)
        go = np.full((h, w), bgc2, dtype=int)
        noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
        gi1inds = {(i, j) for i in range(h) for j in range(w)}
        gi2inds = set(gi1inds)
        tr = succ = 0
        seen = []
        while tr < 5 * noccs and succ < noccs:
            tr += 1
            oh = random.randint(2, min(6, h // 2))
            ow = random.randint(2, min(6, w // 2))
            cands = sorted(ij for ij in gi1inds if ij[0] <= h - oh and ij[1] <= w - ow)
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
                cands2 = sorted(ij for ij in gi2inds if ij[0] <= h - oh and ij[1] <= w - ow)
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
        # keep only instances whose answer is readable from the input alone
        try:
            C, s_, _st, cur = _e67_plan(gi)
            if s_ != sqc or not np.array_equal(cur, go):
                continue
        except Exception:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("could not generate a consistent e6721834 instance")


def derive_operations(I, O=None, examples=None):
    # Same reading as the golf entry: each legend box (fill + marks) is one unit,
    # boxes with the most marks are placed first, and a box is placed where its
    # marks appear on the canvas surrounded by background.
    I = np.asarray(I, dtype=int)
    C, sqc, stamps, _cur = _e67_plan(I, examples)
    ch, cw = C[2], C[3]
    ops, sels = [], []
    # 1. cut the board down to the canvas panel (exact full rectangle -> bbox)
    ops.append(33); sels.append([C[0], C[1], ch - 1, cw - 1])
    # 2. per matched legend box, most marks first: copy the box (full rectangle)
    #    out of the input legend and paste it over its marks. Paste is
    #    0-transparent, so a 0 fill is laid over the box rectangle first.
    for dr, dc, sr, sc, rh, rw in stamps:
        if sqc == 0:
            ops.append(0); sels.append([dr, dc, rh - 1, rw - 1])
        ops.append(28); sels.append([sr, sc, rh - 1, rw - 1])
        ops.append(30); sels.append([dr, dc, 0, 0])
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
