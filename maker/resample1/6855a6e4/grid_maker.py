"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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
    # objc != 0: ARCLE's object buffer (Flip/Move) treats 0 as empty, so a
    # 0-coloured object could not be carried across the bracket
    objc = random.choice(range(1, 10))
    bgc, boxc = random.sample([c for c in range(10) if c != objc], 2)
    return {"bgc": bgc, "objc": objc, "boxc": boxc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, boxc=None, **kw) -> dict:
    from dsl import (uppermost, interval, canvas, connect, backdrop, fill, hmirror, shift,
                     totuple, paint, asobject, dmirror)
    from utils import unifint
    if bgc is None:
        bgc, objc, boxc = random.sample(range(10), 3)
    transpose = random.choice((True, False))
    H, W = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, H))))
    w = unifint(diff_lb, diff_ub, (4, max(4, min(30, W))))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loc = (random.randint(0, h - fullh), random.randint(0, w - fullw))
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
        ncells = random.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(random.sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        # the original shifted by a constant objh+3, which depends on the hidden canvas
        # offset; place the flipped object one gap inside the bracket (verifier rule)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3 - uppermost(cells), 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def _is_bracket_color(I, col):
    cells = list(zip(*np.where(I == col)))
    if not cells:
        return False
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    t, b, l, r = min(rs), max(rs), min(cs), max(cs)
    if any(not (rr in (t, b) or cc in (l, r)) for rr, cc in cells):
        return False
    n = len(cells)
    return n == 2 * (r - l + 1) + 4 or n == 2 * (b - t + 1) + 4


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cols = sorted(set(I.flatten().tolist()))
    boxc = None
    if examples:
        # the bracket is the one colour the demonstrations never change
        for c in cols:
            ok = True
            for ex in examples:
                EI = np.asarray(ex[0], dtype=int); EO = np.asarray(ex[1], dtype=int)
                if not (EI == c).any() or not np.array_equal(EI == c, EO == c):
                    ok = False; break
            if ok and _is_bracket_color(I, c):
                boxc = c; break
    if boxc is None:
        # structural fallback: drawn only on its bbox outline, two full lines + 4 ends
        boxc = next(c for c in cols if _is_bracket_color(I, c))
    rest = [c for c in cols if c != boxc]
    if examples:
        cnt = Counter()
        for ex in examples:
            E = np.asarray(ex[0], dtype=int)
            cnt[Counter(E.flatten().tolist()).most_common(1)[0][0]] += 1
        bgc = cnt.most_common(1)[0][0]
        if bgc not in rest:
            bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    objc = [c for c in rest if c != bgc][0]

    bx = list(zip(*np.where(I == boxc)))
    t = min(r for r, _ in bx); b = max(r for r, _ in bx)
    l = min(c for _, c in bx); rr = max(c for _, c in bx)
    # horizontal brackets: the top row of the bracket bbox is fully bracket-coloured
    horiz = all(I[t, c] == boxc for c in range(l, rr + 1)) and (rr - l + 1) * 2 + 4 == len(bx)
    G = I if horiz else I.T
    if horiz:
        T, B = t, b
        FLIP = 27; DN, UP = 21, 20
    else:
        T, B = l, rr
        FLIP = 26; DN, UP = 22, 23
    conv = (lambda cs: cs) if horiz else (lambda cs: [(c, r) for r, c in cs])

    ops, sels = [], []
    obj = [(int(r), int(c)) for r, c in zip(*np.where(G == objc))]
    groups = [([p for p in obj if p[0] < T], T, DN),
              ([p for p in obj if p[0] > B], B, UP)]
    for cells, axis, mop in groups:
        if not cells:
            continue
        a = min(r for r, _ in cells); z = max(r for r, _ in cells)
        flipped = sorted((a + z - r, c) for r, c in cells)
        # in-place flip, then slide so the far edge lands 2 cells past the bracket line
        steps = (axis + 2 - a) if mop == DN else (z - (axis - 2))
        d = steps if mop == DN else -steps
        final = {(r + d, c) for r, c in flipped}
        if objc == 0:  # 0 is invisible to the object buffer: erase and repaint
            ops.append(bgc); sels.append(sel_of(conv(cells)))
            ops.append(0); sels.append(sel_of(conv(sorted(final))))
            continue
        did_flip = sorted(cells) != flipped
        if did_flip:
            ops.append(FLIP); sels.append(sel_of(conv(cells)))   # grab + flip in place
            ops.append(mop); sels.append(sel_of([]))             # continue same object
        else:
            ops.append(mop); sels.append(sel_of(conv(cells)))    # grab
        for _ in range(steps - 1):
            ops.append(mop); sels.append(sel_of([]))
        hole = sorted(set(cells) - final)
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(conv(hole)))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
