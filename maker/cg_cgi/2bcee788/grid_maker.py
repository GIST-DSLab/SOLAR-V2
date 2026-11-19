"""
ARC Task: 2bcee788 (RE-ARC) — LLM-generated grid_maker
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

_MFS = ['identity', 'dmirror', 'cmirror', 'vmirror', 'hmirror', 'rot90', 'rot180', 'rot270']
_SWAP = {'dmirror', 'cmirror', 'rot90', 'rot270'}


def _apply(fn, g):
    if fn == 'identity': return g
    if fn == 'dmirror': return g.T.copy()
    if fn == 'cmirror': return np.rot90(g, 2).T.copy()
    if fn == 'vmirror': return np.fliplr(g).copy()
    if fn == 'hmirror': return np.flipud(g).copy()
    if fn == 'rot90': return np.rot90(g, -1).copy()
    if fn == 'rot180': return np.rot90(g, 2).copy()
    if fn == 'rot270': return np.rot90(g, 1).copy()
    raise ValueError(fn)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round((b - a) * d + random.uniform(-0.5, 0.5)))))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "sepc": sepc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sepc, objc) -> dict:
    max_h = min(max_h, 30); max_w = min(max_w, 30)
    nmfs = random.choice((1, 2))
    fns = random.sample(_MFS, nmfs)
    swap = sum(f in _SWAP for f in fns) % 2 == 1
    H, W = (max_w, max_h) if swap else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (2, min(20, H - 1)))
    w = _unifint(diff_lb, diff_ub, (2, min(10, (W - 1) // 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    spi = random.randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(inds) - {sp}

    def grow():
        nb = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        nb.add((i + di, j + dj))
        cand = sorted((reminds - shp) & nb)
        if cand:
            shp.add(random.choice(cand))
    for _ in range(numc):
        grow()
    while len({j for _, j in shp}) == 1:
        grow()
    c2 = np.full((h, w), bgc, dtype=int)
    for (i, j) in shp:
        c2[i, j] = objc
    c3 = np.full((h, w), bgc, dtype=int)
    for (i, j) in shp:
        if j == w - 1:
            c3[i, j] = sepc
    gimini = np.hstack([c2, np.fliplr(c3)])
    gomini = np.hstack([c2, np.fliplr(c2)])
    fullh = _unifint(diff_lb, diff_ub, (h + 1, H))
    fullw = _unifint(diff_lb, diff_ub, (2 * w + 1, W))
    gi = np.full((fullh, fullw), bgc, dtype=int)
    go = np.full((fullh, fullw), bgc, dtype=int)
    # RE-ARC paints the mini grids at the origin (loc is sampled but unused)
    gi[:h, :2 * w] = gimini
    go[:h, :2 * w] = gomini
    for fn in fns:
        gi = _apply(fn, gi)
        go = _apply(fn, go)
    go = np.where(go == bgc, 3, go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    # Roles from I alone: background owns the widest bounding box (a full row
    # and a full column always remain background); of the other two colours
    # the blob is the bigger one, the mirror marker (its edge echo) the smaller.
    best = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        key = (int((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)), int(len(rs)))
        if best is None or key > best[0]:
            best = (key, int(col))
    bgc = best[1]
    rest = [int(c) for c in np.unique(I) if int(c) != bgc]
    counts = {c: int((I == c).sum()) for c in rest}
    objc = max(rest, key=lambda c: counts[c])
    sepc = min(rest, key=lambda c: counts[c])

    obj_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    sep_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == sepc))]
    r0 = min(r for r, _ in obj_rc); r1 = max(r for r, _ in obj_rc)
    c0 = min(c for _, c in obj_rc); c1 = max(c for _, c in obj_rc)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    # Which edge of the blob the marker sits against = the mirror axis.
    if all(c == c1 + 1 and r0 <= r <= r1 for r, c in sep_rc):
        side = 'right'
    elif all(c == c0 - 1 and r0 <= r <= r1 for r, c in sep_rc):
        side = 'left'
    elif all(r == r1 + 1 and c0 <= c <= c1 for r, c in sep_rc):
        side = 'bottom'
    else:
        side = 'top'

    if side in ('right', 'left'):
        flip_op = 26
        dj = ow if side == 'right' else -ow
        dest_r, dest_c = r0, c0 + dj
        mirror_rc = [(r, c0 + c1 - c + dj) for r, c in obj_rc]
    else:
        flip_op = 27
        di = oh if side == 'bottom' else -oh
        dest_r, dest_c = r0 + di, c0
        mirror_rc = [(r0 + r1 - r + di, c) for r, c in obj_rc]

    # 1. every background cell becomes 3 (base layer; bgc is never 3)
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == bgc))]
    ops.append(3); sels.append(sel_of(bg_cells))

    if objc != 0:
        # 2. copy the blob's whole bounding rectangle (blob + its new 3-background)
        #    bbox intended: the full rectangle, background included.
        ops.append(29); sels.append([r0, c0, oh - 1, ow - 1])
        # 3. paste it right across the marker line (marker gets covered)
        ops.append(30); sels.append([dest_r, dest_c, 0, 0])
        # 4. reflect the pasted rectangle in place
        src = np.where(I[r0:r1 + 1, c0:c1 + 1] == bgc, 3, I[r0:r1 + 1, c0:c1 + 1])
        mirrored = np.fliplr(src) if flip_op == 26 else np.flipud(src)
        if not np.array_equal(src, mirrored):
            ops.append(flip_op); sels.append([dest_r, dest_c, oh - 1, ow - 1])  # whole rectangle
    else:
        # colour 0 is invisible to Copy/Paste: paint the reflected blob's cells directly
        ops.append(0); sels.append(sel_of(mirror_rc))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
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
                                f"for task 2bcee788"
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
                    f"Failed to build a complete episode for task 2bcee788 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bcee788-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
