"""
ARC Task: bd4472b8 (RE-ARC) — LLM-generated grid_maker
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

# Whole-grid orientation of the (header strip, separator line, field) layout.
ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": k} for k in ex]
    plan.append(dict(random.choice(plan)))      # test orientation was shown
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    def unifint(lo, hi):
        a = lo + (hi - lo) * diff_lb
        b = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, round(random.uniform(a, b))))

    if rot is None:
        rot = random.choice(ROTS)
    # canonical grid is (h+2) x w; odd rotations swap the axes
    mh, mw = (max_h, max_w) if rot in (0, 2) else (max_w, max_h)
    h = unifint(1, max(1, min(28, mh - 2)))
    w = unifint(2, max(2, min(8, mw)))
    remcols = [c for c in range(1, 10) if c != bgc]
    ccols = random.sample(remcols, w)
    gi = np.full((h + 2, w), bgc, dtype=int)
    gi[0] = ccols
    gi[1] = linc
    go = gi.copy()
    for r in range(2, h + 2):
        go[r] = ccols[(r - 2) % w]
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading: the header strip, read sideways, is a column of colors; stripe line j
    (counting away from the separator) is a whole line of header color j. Those w
    uniform lines form one block, and that block is the unit that repeats away from
    the header until the grid ends (last copy cut short).
    Everything is measured from I: orientation, header colors, w, h.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    def canon_ok(G):
        Hc, Wc = G.shape
        if Hc < 3 or Wc < 2 or Wc > 8:
            return False
        if len(set(G[0].tolist())) < 2:        # header: distinct colors
            return False
        if len(set(G[1].tolist())) != 1:       # separator: uniform line
            return False
        rest = set(G[2:].flatten().tolist())   # plain field
        return len(rest) == 1 and rest.pop() != int(G[1][0])

    k = next(kk for kk in range(4) if canon_ok(np.rot90(I, -kk)))
    G = np.rot90(I, -k)
    Hc, Wc = G.shape
    w, h = Wc, Hc - 2
    ccols = G[0].tolist()

    def M(r, c):                     # canonical -> actual coordinates
        if k == 0:
            return (r, c)
        if k == 1:
            return (Wc - 1 - c, r)
        if k == 2:
            return (Hc - 1 - r, Wc - 1 - c)
        return (c, Hc - 1 - r)

    def line(r):                     # one whole stripe line (parallel to header)
        return [M(r, c) for c in range(w)]

    def origin(cells):
        return (min(p[0] for p in cells), min(p[1] for p in cells))

    ops, sels = [], []
    s = min(w, h)
    # 1. header read sideways: stripe line j becomes a full line of header color j
    for j in range(s):
        ops.append(int(ccols[j]))
        sels.append(sel_of(line(2 + j)))
    nxt = 2 + s

    # 2. the w-line block is the repeating unit: copy it, stamp it onward
    if s == w and h // w >= 2:
        block = [p for j in range(w) for p in line(2 + j)]
        ops.append(29); sels.append(sel_of(block))
        for t in range(1, h // w):
            dst = [p for j in range(w) for p in line(2 + t * w + j)]
            ops.append(30); sels.append(sel_of([origin(dst)]))
        nxt = 2 + (h // w) * w

    # 3. the cut-short final block: continue the color cycle line by line
    for r in range(nxt, Hc):
        ops.append(int(ccols[(r - 2) % w]))
        sels.append(sel_of(line(r)))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])   # whole grid: an exact full rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
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
                                f"for task bd4472b8"
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
                    f"Failed to build a complete episode for task bd4472b8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bd4472b8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
