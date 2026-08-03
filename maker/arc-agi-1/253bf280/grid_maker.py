"""
ARC Task: 253bf280 (RE-ARC) — LLM-generated grid_maker
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
    from utils import unifint
except Exception:
    def unifint(diff_lb, diff_ub, bounds):
        a, b = bounds
        s = a + int((b - a) * diff_lb)
        e = a + int((b - a) * diff_ub)
        s, e = min(s, e), max(s, e)
        return random.randint(max(a, s), min(b, e))


def sample_colors(num_examples=None) -> dict:
    # generator: colopts = remove(3, interval(0,10,1)); bgc = choice(colopts);
    #            fgcol = choice(remove(bgc, colopts))   -> both random, both fixed here
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    fgcol = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgcol": fgcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgcol) -> dict:
    hb = max(3, min(30, max_h))
    wb = max(3, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (3, hb))
    w = unifint(diff_lb, diff_ub, (3, wb))

    inds = [(i, j) for i in range(h) for j in range(w)]
    ub = max(2, (h * w) // 4)
    num = unifint(diff_lb, diff_ub, (2, ub))
    num = max(2, min(num, len(inds)))
    s = random.sample(inds, num)

    gi = [[bgc] * w for _ in range(h)]
    for (i, j) in s:
        gi[i][j] = fgcol

    go = [row[:] for row in gi]
    # horizontal connects: leftmost fgcol -> rightmost fgcol in each row with >1
    for i in range(h):
        cs = [j for j in range(w) if gi[i][j] == fgcol]
        if len(cs) > 1:
            for j in range(min(cs), max(cs) + 1):
                if gi[i][j] != fgcol:
                    go[i][j] = 3
    # vertical connects: topmost fgcol -> bottommost fgcol in each column with >1
    for j in range(w):
        rs = [i for i in range(h) if gi[i][j] == fgcol]
        if len(rs) > 1:
            for i in range(min(rs), max(rs) + 1):
                if gi[i][j] != fgcol:
                    go[i][j] = 3

    return {
        'input': tuple(tuple(r) for r in gi),
        'output': tuple(tuple(r) for r in go),
    }


def derive_operations(I, O):
    """Rule read off I: fgcol = rarest colour in I. Any row holding >=2 fgcol cells
    gets a horizontal connect-line drawn in 3 between its outermost fgcol cells;
    any column holding >=2 fgcol cells gets a vertical connect-line. fgcol cells
    themselves survive, so a line is emitted as its maximal non-fgcol runs.
    Each line object is drawn with its ops adjacent; O is never consulted."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    fg = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    cur = I.copy()

    # ---- horizontal connect-line objects (one row-line at a time) ----
    for r in range(hi):
        cs = [c for c in range(wi) if I[r, c] == fg]
        if len(cs) < 2:
            continue
        c0, c1 = min(cs), max(cs)
        c = c0
        while c <= c1:
            if I[r, c] == fg:
                c += 1
                continue
            e = c
            while e <= c1 and I[r, e] != fg:
                e += 1
            if not np.all(cur[r, c:e] == 3):
                ops.append(3)
                sels.append([r, c, 0, e - 1 - c])
                cur[r, c:e] = 3
            c = e

    # ---- vertical connect-line objects (one column-line at a time) ----
    for c in range(wi):
        rs = [r for r in range(hi) if I[r, c] == fg]
        if len(rs) < 2:
            continue
        r0, r1 = min(rs), max(rs)
        r = r0
        while r <= r1:
            if I[r, c] == fg:
                r += 1
                continue
            e = r
            while e <= r1 and I[e, c] != fg:
                e += 1
            if not np.all(cur[r:e, c] == 3):
                ops.append(3)
                sels.append([r, c, e - 1 - r, 0])
                cur[r:e, c] = 3
            r = e

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 253bf280"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 253bf280"
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
                                f"for task 253bf280"
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
                    f"Failed to build a complete episode for task 253bf280 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"253bf280-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
