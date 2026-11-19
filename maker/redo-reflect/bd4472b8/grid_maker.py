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

# The generator applies one of four whole-grid rotations to every instance, so the
# header strip can sit on any of the four edges.  That is a discrete structural
# variant -> plan it per instance so the episode is learnable.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))                      # generator uses colors 1..9
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test rotation was shown
    # ccols are pure "read the header" colors -> the rule is color-agnostic there,
    # so they stay per-instance random (only bgc / line color are fixed roles).
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    # canonical grid is (h+2, w); rot 1/3 transposes the dims
    if rot in (0, 2):
        hmax, wmax = min(28, max_h - 2), min(8, max_w)
    else:
        hmax, wmax = min(28, max_w - 2), min(8, max_h)
    hmax, wmax = max(1, hmax), max(2, wmax)
    h = unifint(diff_lb, diff_ub, (1, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))

    ccols = random.sample([c for c in range(1, 10) if c != bgc], w)
    gi = [list(ccols), [linc] * w] + [[bgc] * w for _ in range(h)]
    go = [list(ccols), [linc] * w] + [[ccols[i % w]] * w for i in range(h)]
    gi = np.rot90(np.array(gi, dtype=int), rot).tolist()
    go = np.rot90(np.array(go, dtype=int), rot).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """
    Rule: the header strip of w colors is reflected across the diagonal (a 90 deg turn
    of the strip), each color then spreads across its whole stripe line, and the
    resulting w x w block repeats away from the header line until the grid is full.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    def canon_ok(G):
        Hc, Wc = G.shape
        if Hc < 3 or Wc < 2 or Wc > 8:
            return False
        if len(set(G[0].tolist())) < 2:        # header strip: distinct colors
            return False
        if len(set(G[1].tolist())) != 1:       # separator line: uniform
            return False
        rest = set(G[2:].flatten().tolist())   # plain background field
        if len(rest) != 1:
            return False
        return rest.pop() != int(G[1][0])

    # find which whole-grid rotation was applied (header strip on top -> canonical)
    k, Gc, fb, fb_G = None, None, None, None
    for kk in range(4):
        G = np.rot90(I, -kk)
        if not canon_ok(G):
            continue
        if fb is None:
            fb, fb_G = kk, G
        Hc, Wc = G.shape
        Gp = G.copy()
        for r in range(2, Hc):
            Gp[r, :] = G[0, (r - 2) % Wc]
        if np.array_equal(np.rot90(Gp, kk), O):
            k, Gc = kk, G
            break
    if k is None:
        k, Gc = fb, fb_G

    Hc, Wc = Gc.shape
    w, h = Wc, Hc - 2
    ccols = Gc[0].tolist()

    def M(r, c):                    # canonical -> actual grid coordinates (a rotation)
        if k == 0:
            return (r, c)
        if k == 1:
            return (Wc - 1 - c, r)
        if k == 2:
            return (Hc - 1 - r, Wc - 1 - c)
        return (c, Hc - 1 - r)

    def origin_of(cells):
        return (min(p[0] for p in cells), min(p[1] for p in cells))

    ops, sels = [], []
    s = min(w, h)                   # size of the square the reflection can use
    next_row = 2

    if s >= 2:
        # copy the header strip (as much of it as there is room for)
        src = [M(0, i) for i in range(s)]
        ops.append(28); sels.append(sel_of(src))
        # lay it down on the far edge of the square block that starts the stripes
        dst = [M(s + 1, i) for i in range(s)]
        ops.append(30); sels.append(sel_of([origin_of(dst)]))
        # THE REFLECTION: turn the strip a quarter turn so it runs across the stripes
        block = [M(r, c) for r in range(2, s + 2) for c in range(s)]
        ops.append(25); sels.append(sel_of(block))
        # spread each color along its own stripe line (its seed cell is already right)
        for r in range(2, s + 2):
            ops.append(int(ccols[(r - 2) % w]))
            sels.append(sel_of([M(r, c) for c in range(1, w)]))
        next_row = s + 2

    # repeat the finished w x w block onward while whole copies still fit
    if s == w and h // w >= 2:
        tile = [M(r, c) for r in range(2, w + 2) for c in range(w)]
        ops.append(29); sels.append(sel_of(tile))
        for t in range(1, h // w):
            dcells = [M(r, c) for r in range(2 + t * w, 2 + (t + 1) * w) for c in range(w)]
            ops.append(30); sels.append(sel_of([origin_of(dcells)]))
        next_row = 2 + (h // w) * w

    # the trailing partial block: continue the cycle one stripe line at a time
    for r in range(next_row, Hc):
        ops.append(int(ccols[(r - 2) % w]))
        sels.append(sel_of([M(r, c) for c in range(w)]))

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
