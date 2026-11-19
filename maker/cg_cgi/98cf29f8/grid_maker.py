"""
ARC Task: 98cf29f8 (RE-ARC) — LLM-generated grid_maker
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
    bgc, objc, otherc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "otherc": otherc}


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, otherc=None) -> dict:
    if bgc is None or objc is None or otherc is None:
        bgc, objc, otherc = random.sample(range(10), 3)

    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + int(round(diff_lb * (hi - lo)))
        b = lo + int(round(diff_ub * (hi - lo)))
        return random.randint(min(a, b), max(a, b))

    cap = max(10, min(30, max_h, max_w))
    h = unifint(10, cap)
    w = unifint(10, cap)
    objh = unifint(2, h - 5)
    objw = unifint(2, w - 5)
    loci = random.randint(0, h - objh)
    locj = random.randint(0, w - objw)
    gi = [[bgc] * w for _ in range(h)]
    for i in range(loci, loci + objh):
        for j in range(locj, locj + objw):
            gi[i][j] = objc
    margs = (h - (loci + objh), w - (locj + objw), loci, locj)
    options = [k for k, m in enumerate(margs) if m > 2]
    pos = random.choice(options)
    for _ in range(pos):
        gi = _rot90(gi)
    h, w = len(gi), len(gi[0])
    ofc = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == objc]
    low = max(i for i, _ in ofc)
    rgt = max(j for _, j in ofc)
    lft = min(j for _, j in ofc)
    locis = random.randint(low + 2, h - 2)
    locie = random.randint(locis + 1, h - 1)
    locjs = random.randint(0, min(w - 2, rgt))
    locje = random.randint(max(locjs + 1, lft), w - 1)
    ub = min(rgt, locje)
    lb = max(lft, locjs)
    jloc = random.randint(lb, ub)
    ln = [(i, jloc) for i in range(low + 1, locis)]
    go = [row[:] for row in gi]
    for i in range(locis, locie + 1):
        for j in range(locjs, locje + 1):
            gi[i][j] = otherc
            go[i - len(ln)][j] = otherc
    for i, j in ln:
        gi[i][j] = otherc
    fns = [
        lambda g: [list(r) for r in g],
        lambda g: [list(r) for r in zip(*g)],
        lambda g: [list(r) for r in zip(*[row[::-1] for row in g[::-1]])],
        lambda g: [row[::-1] for row in g],
        lambda g: [list(r) for r in g[::-1]],
        lambda g: _rot90(g),
        lambda g: [row[::-1] for row in g[::-1]],
        lambda g: _rot90(_rot90(_rot90(g))),
    ]
    for fn in random.sample(fns, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def _find_stem(A):
    """On grid A, find a colour made of a full rectangle plus a 1-wide stem
    running along the row axis (stem rows hold exactly one cell of it)."""
    h, w = A.shape
    found = []
    for c in sorted(set(A.flatten().tolist())):
        cells = list(zip(*np.nonzero(A == c)))
        cnt = Counter(r for r, _ in cells)
        stem_rows = sorted(r for r, k in cnt.items() if k == 1)
        body_rows = sorted(r for r, k in cnt.items() if k > 1)
        if not stem_rows or len(body_rows) < 2:
            continue
        if stem_rows != list(range(stem_rows[0], stem_rows[-1] + 1)):
            continue
        if body_rows != list(range(body_rows[0], body_rows[-1] + 1)):
            continue
        cols = sorted({cc for r, cc in cells if r in set(body_rows)})
        c0, c1 = cols[0], cols[-1]
        if any(cnt[r] != c1 - c0 + 1 for r in body_rows):
            continue
        if any(A[r, cc] != c for r in body_rows for cc in range(c0, c1 + 1)):
            continue
        scol = {cc for r, cc in cells if r in set(stem_rows)}
        if len(scol) != 1:
            continue
        sc = scol.pop()
        if not (c0 <= sc <= c1):
            continue
        if body_rows[-1] + 1 == stem_rows[0]:
            d = 1          # block above the stem -> closes up downward
            tip = stem_rows[-1] + 1
        elif stem_rows[-1] + 1 == body_rows[0]:
            d = -1         # block below the stem -> closes up upward
            tip = stem_rows[0] - 1
        else:
            continue
        if not (0 <= tip < h) or A[tip, sc] == c:
            continue
        anchor = int(A[tip, sc])
        found.append(dict(block=int(c), anchor=anchor, body=body_rows, c0=c0, c1=c1,
                          stem=stem_rows, scol=sc, d=d))
    return found


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    submit_sel = [0, 0, hi - 1, wi - 1]
    ops, sels = [], []

    res, transposed = None, False
    for tr in (False, True):
        A = I.T if tr else I
        f = _find_stem(A)
        if f:
            res, transposed = f[0], tr
            break
    if res is None:
        return [34], [submit_sel]

    counts = Counter(I.flatten().tolist())
    others = [c for c in counts if c not in (res["block"], res["anchor"])]
    bg = max(others, key=lambda c: counts[c]) if others else 0

    A = I.T if transposed else I
    ha, wa = A.shape
    d = res["d"]
    n = len(res["stem"])
    fix = (lambda r, c: (c, r)) if transposed else (lambda r, c: (r, c))

    if res["block"] != 0:
        # Reading: the stem lines are deleted and the whole band of the grid on
        # the block's side of the stem closes up toward the anchor.
        if d == 1:
            band_rows = range(0, res["stem"][0])
        else:
            band_rows = range(res["stem"][-1] + 1, ha)
        band = [fix(r, c) for r in band_rows for c in range(wa)]  # full rectangle band
        if bg == 0:
            # background 0 is not carried by Move, so the band cannot wipe the
            # connector itself: remove the stem first.
            stem = [fix(r, res["scol"]) for r in res["stem"]]
            ops.append(0); sels.append(sel_of(stem))
        if transposed:
            move_op = 22 if d == 1 else 23
        else:
            move_op = 21 if d == 1 else 20
        ops.append(move_op); sels.append(sel_of(band))   # grab the band once
        for _ in range(n - 1):
            ops.append(move_op); sels.append(sel_of([]))  # keep sliding same band
        # lines vacated at the grid edge the band came from
        if d == 1:
            vac_rows = range(0, n)
        else:
            vac_rows = range(ha - n, ha)
        vac = [fix(r, c) for r in vac_rows for c in range(wa)]
        if bg != 0:
            ops.append(int(bg)); sels.append(sel_of(vac))
    else:
        # colour-0 block cannot be carried by ARCLE's Move: repaint it.
        body = res["body"]
        lo, hi_ = res["c0"], res["c1"]
        rect = [(r, c) for r in body for c in range(lo, hi_ + 1)]
        dst = [(r + d * n, c) for r, c in rect]
        stem = [(r, res["scol"]) for r in res["stem"]]
        # erase the whole colour-0 shape (block + connector) to background,
        # then lay the block down against the anchor
        ops.append(int(bg)); sels.append(sel_of([fix(r, c) for r, c in rect + stem]))
        ops.append(0); sels.append(sel_of([fix(r, c) for r, c in dst]))

    ops.append(34); sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 98cf29f8"
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
                                f"for task 98cf29f8"
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
                    f"Failed to build a complete episode for task 98cf29f8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"98cf29f8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
