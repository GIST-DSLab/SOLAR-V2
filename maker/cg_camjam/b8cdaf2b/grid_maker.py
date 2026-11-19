"""
ARC Task: b8cdaf2b (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "dotc": dotc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    v = int(round(random.uniform(a, b)))
    return max(lo, min(hi, v))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc) -> dict:
    k = random.choice([0, 1, 2, 3])            # identity, rot90, rot180, rot270
    if k % 2 == 0:
        hb, wb = max_h, max_w
    else:
        hb, wb = max_w, max_h                  # canvas is transposed by the rotation
    h = _unifint(diff_lb, diff_ub, 3, min(30, hb))
    w = _unifint(diff_lb, diff_ub, 3, min(30, wb))
    winv = _unifint(diff_lb, diff_ub, 2, w - 1)
    w2 = w - winv
    w2 = min(max(w2, 1), w - 2)
    locj = random.randint(1, w - w2 - 1)

    gi = np.full((h, w), bgc, dtype=int)
    gi[0, :] = linc
    gi[0, locj:locj + w2] = dotc
    gi[1, locj:locj + w2] = linc
    go = gi.copy()
    r, c = 2, locj - 1
    while 0 <= r < h and 0 <= c < w:
        if go[r, c] == bgc:
            go[r, c] = dotc
        r += 1; c -= 1
    r, c = 2, locj + w2
    while 0 <= r < h and 0 <= c < w:
        if go[r, c] == bgc:
            go[r, c] = dotc
        r += 1; c += 1

    # rot90 (CW) = np.rot90(k=3); rot180 = k=2; rot270 = k=1
    npk = {0: 0, 1: 3, 2: 2, 3: 1}[k]
    gi = np.rot90(gi, npk)
    go = np.rot90(go, npk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): the two rays leaving the ends of
    the dot bar form a symmetric "V" that opens outward from the bar. The
    unit is one DEPTH LEVEL. At each level the pair of mirror cells (one off
    each end of the bar, the same distance in) is painted together. The V
    grows level by level, moving from the bar into the grid, until both arms
    have left the grid.
    Everything is measured from I: background = majority colour, dot colour =
    least-common colour, and the edge line that holds the bar gives the
    inward direction.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    dotc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    dots = [(r, c) for r in range(h) for c in range(w) if I[r, c] == dotc]

    ops, sels = [], []
    if dots:
        rows = {r for r, _ in dots}
        cols = {c for _, c in dots}
        d = a = None
        if len(rows) == 1:
            r0 = next(iter(rows))
            if r0 in (0, h - 1) and all(I[r0, c] != bgc for c in range(w)):
                d = (1, 0) if r0 == 0 else (-1, 0)
                a = (0, 1)
        if d is None and len(cols) == 1:
            c0 = next(iter(cols))
            if c0 in (0, w - 1) and all(I[r, c0] != bgc for r in range(h)):
                d = (0, 1) if c0 == 0 else (0, -1)
                a = (1, 0)

        if d is not None:
            key = lambda p: p[0] * a[0] + p[1] * a[1]
            ps = min(dots, key=key)
            pe = max(dots, key=key)
            oa = (ps[0] + 2 * d[0] - a[0], ps[1] + 2 * d[1] - a[1])
            sa = (d[0] - a[0], d[1] - a[1])
            ob = (pe[0] + 2 * d[0] + a[0], pe[1] + 2 * d[1] + a[1])
            sb = (d[0] + a[0], d[1] + a[1])

            def inb(p):
                return 0 <= p[0] < h and 0 <= p[1] < w

            k = 0
            while True:
                pa = (oa[0] + k * sa[0], oa[1] + k * sa[1])
                pb = (ob[0] + k * sb[0], ob[1] + k * sb[1])
                if not inb(pa) and not inb(pb):
                    break
                level = [p for p in (pa, pb) if inb(p) and I[p] == bgc]
                if level:
                    ops.append(int(dotc))
                    sels.append(sel_of(level))   # the mirror pair at this depth
                k += 1

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b8cdaf2b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8cdaf2b"
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
                                f"for task b8cdaf2b"
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
                    f"Failed to build a complete episode for task b8cdaf2b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8cdaf2b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
