"""
ARC Task: 746b3537 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Every colour is part of the content being collapsed; nothing plays a fixed role.
    # Orientation (row stripes vs column stripes) is planned so both are shown.
    n_ex = num_examples if num_examples else 3
    if n_ex >= 2:
        ex = [{"transpose": False}, {"transpose": True}]
        ex += [{"transpose": random.choice((False, True))} for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [{"transpose": random.choice((False, True))} for _ in range(n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, transpose=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    if transpose is None:
        transpose = random.choice((True, False))
    lim_len = max_w if transpose else max_h     # stripe-count axis of the input
    lim_wid = max_h if transpose else max_w
    h = unifint(diff_lb, diff_ub, (2, min(15, lim_len - 1)))
    w = unifint(diff_lb, diff_ub, (1, min(30, lim_wid)))
    cols, lastc = [], -1
    for _ in range(h):
        c = random.choice([x for x in range(10) if x != lastc])
        cols.append(c)
        lastc = c
    go = [[c] for c in cols]
    gi = [[c] * w for c in cols]
    numinserts = unifint(diff_lb, diff_ub, (1, min(30, lim_len) - h))
    for _ in range(numinserts):
        loc = random.randint(0, len(gi) - 1)
        gi = gi[:loc + 1] + gi[loc:]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


# ---------------------------------------------------------------------------
# Reading of the task: a line (row or column) that repeats the line before it
# is a duplicate.  Collapse consecutive duplicate rows, then consecutive
# duplicate columns.  A duplicate is removed by sliding everything after it
# back onto it (MoveU / MoveL of the rectangle of remaining lines); duplicates
# at the very end are simply cropped off.
# ---------------------------------------------------------------------------

def _mask(sel, shape=(30, 30)):
    m = np.zeros(shape, dtype=bool)
    if isinstance(sel, dict):
        for r, c in sel["cells"]:
            m[r, c] = True
    else:
        r, c, h, w = sel
        m[r:r + h + 1, c:c + w + 1] = True
    return m


def _simulate(I, ops, sels):
    """Minimal ARCLE model of the ops used here (Color0-9, Move, Crop)."""
    g = np.zeros((30, 30), dtype=int)
    gh, gw = I.shape
    g[:gh, :gw] = I
    obj = None
    for op, sel in zip(ops, sels):
        m = _mask(sel)
        if op <= 9:
            g[m] = op
            obj = None
        elif op in (20, 21, 22, 23):
            if m.any():
                rs, cs = np.where(m)
                x0, x1, y0, y1 = rs.min(), rs.max(), cs.min(), cs.max()
                part = m[x0:x1 + 1, y0:y1 + 1]
                patch = np.where(part, g[x0:x1 + 1, y0:y1 + 1], 0)
                bg = g.copy()
                bg[m] = 0
                obj = [patch, bg, (x0, y0)]
            if obj is None:
                continue
            dx, dy = {20: (-1, 0), 21: (1, 0), 22: (0, 1), 23: (0, -1)}[op]
            x, y = obj[2]
            x, y = x + dx, y + dy
            obj[2] = (x, y)
            g = obj[1].copy()
            p = obj[0]
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if p[i, j] > 0 and 0 <= x + i < gh and 0 <= y + j < gw:
                        g[x + i, y + j] = p[i, j]
        elif op == 33:
            obj = None
            rs, cs = np.where(m)
            x0, x1, y0, y1 = rs.min(), rs.max(), cs.min(), cs.max()
            sub = np.where(m[x0:x1 + 1, y0:y1 + 1], g[x0:x1 + 1, y0:y1 + 1], 0)
            g = np.zeros((30, 30), dtype=int)
            g[:x1 - x0 + 1, :y1 - y0 + 1] = sub
            gh, gw = x1 - x0 + 1, y1 - y0 + 1
    return g[:gh, :gw]


def _rule(I):
    """The answer as read from I alone (not from O)."""
    rows = [I[0]] + [I[i] for i in range(1, I.shape[0]) if not np.array_equal(I[i], I[i - 1])]
    R = np.array(rows)
    cols = [R[:, 0]] + [R[:, j] for j in range(1, R.shape[1])
                        if not np.array_equal(R[:, j], R[:, j - 1])]
    return np.array(cols).T


def _runs(lines):
    rs = []
    for v in lines:
        if rs and rs[-1][0] == v:
            rs[-1][1] += 1
        else:
            rs.append([v, 1])
    return rs


def _build(I, backward=True):
    """Slide-to-collapse plan.  Returns ops/sels (without Crop/Submit)."""
    G = I.copy()
    ops, sels = [], []
    ext = [I.shape[0], I.shape[1]]

    for axis in (0, 1):
        M = G if axis == 0 else G.T            # view: lines along M's first axis
        n_other = ext[1 - axis]
        runs = _runs([tuple(M[i, :n_other].tolist()) for i in range(ext[axis])])
        clen = ext[axis]
        mop = 20 if axis == 0 else 23

        def cells_of(lines_from, lines_to):
            out = []
            for i in range(lines_from, lines_to):
                for j in range(n_other):
                    out.append((i, j) if axis == 0 else (j, i))
            return out

        starts, p = [], 0
        for (_v, L) in runs:
            starts.append(p)
            p += L
        order = list(range(len(runs)))
        if backward:
            order = order[::-1]
        for idx in order:
            L = runs[idx][1]
            cur = starts[idx] if backward else idx
            if L == 1:
                continue
            s, k = cur + L, L - 1
            if s > clen - 1:
                clen = cur + 1                 # trailing duplicates: cropped off at the end
                continue
            block = M[s:clen, :n_other].copy()
            if not block[0].any():
                # next line is a 0 stripe, which Move cannot carry: wipe this run's
                # duplicate lines to 0 so they become (part of) that stripe
                ops.append(0); sels.append(sel_of(cells_of(cur + 1, s)))
                M[cur + 1:s, :n_other] = 0
                if not block.any():
                    clen = cur + 2
                    continue
            # the whole rectangle of lines after the run slides back k lines
            # (bbox: the full rectangle of remaining lines is the intended cell set)
            if axis == 0:
                first = [s, 0, clen - 1 - s, n_other - 1]
            else:
                first = [0, s, n_other - 1, clen - 1 - s]
            ops.append(mop); sels.append(first)
            for _ in range(k - 1):
                ops.append(mop); sels.append(sel_of([]))
            newM = M[:clen, :n_other].copy()
            newM[s:clen] = 0
            region = newM[s - k:clen - k]
            nz = block != 0
            region[nz] = block[nz]
            # the block's own 0 cells did not overwrite what lay under them
            fixc = [(s - k + i, j) for i in range(block.shape[0]) for j in range(n_other)
                    if block[i, j] == 0 and newM[s - k + i, j] != 0]
            if fixc:
                for a, b in fixc:
                    newM[a, b] = 0
                ops.append(0)
                sels.append(sel_of([(a, b) if axis == 0 else (b, a) for a, b in fixc]))
            M[:clen, :n_other] = newM
            clen -= k
        ext[axis] = clen
    return ops, sels, ext


def _states_ok(I, ops, sels, want):
    seen = {(I.shape, I.tobytes())}
    g = I
    for k in range(1, len(ops) + 1):
        g = _simulate(I, ops[:k], sels[:k])
        key = (g.shape, g.tobytes())
        if key in seen:
            return False
        seen.add(key)
    return np.array_equal(g, want)


def _robust(I, ops, sels, want):
    if not _states_ok(I, ops, sels, want):
        return False
    for k in range(len(ops)):
        if np.array_equal(_simulate(I, ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]), want):
            return False
    return True


def _finish(I, ops, sels, ext):
    ops = list(ops); sels = list(sels)
    ops.append(33); sels.append([0, 0, ext[0] - 1, ext[1] - 1])
    return ops, sels


def _prune(I, ops, sels, want):
    # a 0 repair is dropped when the slides already leave 0 there by themselves
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            if ops[k] != 0:
                continue
            o2, s2 = ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]
            if _states_ok(I, o2, s2, want):
                ops, sels = o2, s2
                changed = True
                break
    return ops, sels


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    want = _rule(I)
    present = set(int(v) for v in np.unique(I))
    spare = next((c for c in range(1, 10) if c not in present), None)

    candidates = []
    for backward in (True, False):
        o, s, ext = _build(I, backward)
        o, s = _finish(I, o, s, ext)
        candidates.append(_prune(I, o, s, want))
    if spare is not None and 0 in present:
        # tint the 0 stripes with an unused colour while lines slide (Move carries
        # only non-zero cells), turn them back to 0 after the crop
        J = np.where(I == 0, spare, I)
        zc = [(r, c) for r in range(I.shape[0]) for c in range(I.shape[1]) if I[r, c] == 0]
        wj = _rule(J)
        back = [(r, c) for r in range(wj.shape[0]) for c in range(wj.shape[1]) if wj[r, c] == spare]
        for backward in (True, False):
            o, s, ext = _build(J, backward)
            o, s = _finish(J, o, s, ext)
            candidates.append(([spare] + o + [0], [sel_of(zc)] + s + [sel_of(back)]))

    chosen = None
    for o, s in candidates:
        if _robust(I, o, s, want):
            chosen = (o, s)
            break
    if chosen is None:
        for o, s in candidates:
            if _states_ok(I, o, s, want):
                chosen = (o, s)
                break
    if chosen is None:
        chosen = candidates[0]
    ops, sels = list(chosen[0]), list(chosen[1])
    h, w = want.shape
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 746b3537"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 746b3537"
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
                                f"for task 746b3537"
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
                    f"Failed to build a complete episode for task 746b3537 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"746b3537-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
