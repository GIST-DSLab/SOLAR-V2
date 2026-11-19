"""
ARC Task: f8a8fe49 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the pattern is duplicated with CopyI/Paste, which cannot carry colour 0
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    hb, wb = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, hb))
    w = unifint(diff_lb, diff_ub, (4, wb))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(sample(totuple(cands), ncells))
        cells = insert(choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        # exact reflection through the box line (row objh+1), as the verifier reads it
        refl = frozenset((2 * (objh + 1) - i, j) for i, j in cells)
        canvo = fill(canvo, objc, refl)
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Two brackets (box lines) enclose a strip.  Split the strip's interior into
    two halves at its middle; each half is mirrored and placed just outside the
    box line it touches; the strip is left empty.
    Per half: CopyI the half -> Paste it just beyond the line -> Flip it in place.
    Then clear the interior to background.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    others = sorted({int(v) for v in I.flatten().tolist()} - {bgc})

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    def area(col):
        r0, r1, c0, c1 = bbox(col)
        return (r1 - r0 + 1) * (c1 - c0 + 1)

    boxc = max(others, key=area)
    r0, r1, c0, c1 = bbox(boxc)
    horizontal = bool(np.all(I[r0, c0:c1 + 1] == boxc))

    ops, sels = [], []
    grid = I.copy()

    def rect(r, c, hh, ww):
        return [(rr, cc) for rr in range(r, r + hh) for cc in range(c, c + ww)]

    def half_moves():
        # (src_r, src_c, hh, ww, dst_r, dst_c, flip_op)
        if horizontal:
            n = r1 - r0 - 1
            ww = c1 - c0 - 1
            L1 = n // 2
            L2 = n - L1
            a = min(L1, r0)                 # rows available above the upper line
            b = min(L2, H - 1 - r1)         # rows available below the lower line
            out = []
            if a > 0:
                out.append((r0 + 1, c0 + 1, a, ww, r0 - a, c0 + 1, 27))
            if b > 0:
                out.append((r1 - b, c0 + 1, b, ww, r1 + 1, c0 + 1, 27))
            return out
        n = c1 - c0 - 1
        hh = r1 - r0 - 1
        L1 = n // 2
        L2 = n - L1
        a = min(L1, c0)
        b = min(L2, W - 1 - c1)
        out = []
        if a > 0:
            out.append((r0 + 1, c0 + 1, hh, a, r0 + 1, c0 - a, 26))
        if b > 0:
            out.append((r0 + 1, c1 - b, hh, b, r0 + 1, c1 + 1, 26))
        return out

    for sr, sc, hh, ww, dr, dc, fop in half_moves():
        src = I[sr:sr + hh, sc:sc + ww]
        if np.all(src == bgc):
            continue
        # copy the half from the input (full rectangle), paste it just beyond its box line
        ops.append(28); sels.append(sel_of(rect(sr, sc, hh, ww)))
        ops.append(30); sels.append(sel_of([(dr, dc)]))
        blk = grid[dr:dr + hh, dc:dc + ww]
        blk[src != 0] = src[src != 0]
        # mirror the pasted half in place (full destination rectangle)
        cur = grid[dr:dr + hh, dc:dc + ww].copy()
        fl = np.flipud(cur) if fop == 27 else np.fliplr(cur)
        if not np.array_equal(fl, cur):
            ops.append(fop); sels.append(sel_of(rect(dr, dc, hh, ww)))
            grid[dr:dr + hh, dc:dc + ww] = fl

    # empty the strip between the box lines (full interior rectangle)
    ir, ic, ih, iw = r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1
    if np.any(grid[ir:ir + ih, ic:ic + iw] != bgc):
        ops.append(bgc); sels.append(sel_of(rect(ir, ic, ih, iw)))

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
                        f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8a8fe49"
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
                                f"for task f8a8fe49"
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
                    f"Failed to build a complete episode for task f8a8fe49 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8a8fe49-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
