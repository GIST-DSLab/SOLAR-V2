"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # 0=identity, 1=rot90(CW), 2=rot180, 3=rot270


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})  # test orientation was shown
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    # base-frame dims; odd rotations swap h/w so the final grid fits max_h x max_w
    mh, mw = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = _unifint(diff_lb, diff_ub, (3, max(3, mh)))
    w = _unifint(diff_lb, diff_ub, (3, max(3, mw)))
    barth = _unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = _unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[0:loci + 1, locj:locj + barth] = fgc
    go = np.full((h, w), bgc, dtype=int)
    for k in range(16):
        for kk, col in ((2 * k, fgc), (2 * k + 1, 8)):
            for sgn in (1, -1):
                r0, r1 = -kk, loci - kk
                c0 = locj + sgn * kk * barth
                c1 = locj + barth - 1 + sgn * kk * barth
                a, b = max(0, r0), min(h - 1, r1)
                c, d = max(0, c0), min(w - 1, c1)
                if a <= b and c <= d:
                    go[a:b + 1, c:d + 1] = col
    kmap = {0: 0, 1: 3, 2: 2, 3: 1}  # DSL rot90 = CW = np.rot90(k=3)
    gi = np.rot90(gi, kmap[rot]).copy()
    go = np.rot90(go, kmap[rot]).copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the bar touching one grid edge is echoed in bar-width strips on
    # both sides. Strip j (j = 1, 2, ...) is the bar pulled back j cells toward
    # its edge, so each strip is one cell shorter than the one before it. Colours
    # alternate: 8 when j is odd, the bar's colour when j is even.
    # The code-golf entry reads the task the same way: it fills outward from the
    # bar, symmetric on both sides, each strip shorter by one, colours
    # alternating. Its per-column loop is just the barth=1 case of the
    # bar-width strip. The two readings agree, so the decomposition is unchanged:
    # one Color op per strip, each strip a full rectangle named from the bar's
    # position, edge and width, ordered from nearest to farthest.
    # Strips never overlap each other or the bar, and they land on background
    # (bgc is never 8 or fgc), so every op visibly changes the grid.
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    full = [(r, c) for r in range(hi) for c in range(wi)]
    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[np.argmax(cnts)])
    mask = I != bgc
    if not mask.any():
        return [34], [sel_of(full)]
    rs, cs = np.where(mask)
    r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
    fgc = int(I[r0, c0])

    # the generator guarantees the bar touches exactly one edge
    if r0 == 0:
        edge = 'top'
    elif r1 == hi - 1:
        edge = 'bottom'
    elif c0 == 0:
        edge = 'left'
    else:
        edge = 'right'

    if edge in ('top', 'bottom'):
        barth = c1 - c0 + 1
        recede = (-1, 0) if edge == 'top' else (1, 0)
        side = (0, barth)
    else:
        barth = r1 - r0 + 1
        recede = (0, -1) if edge == 'left' else (0, 1)
        side = (barth, 0)

    ops, sels = [], []
    for k in range(1, max(hi, wi) + 1):
        col = 8 if k % 2 == 1 else fgc
        for sgn in (1, -1):
            dr = k * recede[0] + sgn * k * side[0]
            dc = k * recede[1] + sgn * k * side[1]
            a, b = max(0, r0 + dr), min(hi - 1, r1 + dr)
            c, d = max(0, c0 + dc), min(wi - 1, c1 + dc)
            if a > b or c > d:
                continue  # strip fully receded or off-grid: nothing to paint
            ops.append(col)
            sels.append(sel_of([(r, cc) for r in range(a, b + 1)
                                for cc in range(c, d + 1)]))
    ops.append(34)
    sels.append(sel_of(full))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
