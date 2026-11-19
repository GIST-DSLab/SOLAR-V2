"""
ARC Task: a3df8b1e (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc = random.sample(range(10), 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    from collections import Counter as _C
    if bgc is None or linc is None:
        bgc, linc = random.sample(range(10), 2)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    cap = min(max_h, max_w, 30)
    w = unifint(diff_lb, diff_ub, (2, max(2, min(10, cap - 1))))
    h = unifint(diff_lb, diff_ub, (w + 1, max(w + 1, cap)))
    gi = [[bgc] * w for _ in range(h)]
    sp = (h - 1, 0)
    gi[sp[0]][sp[1]] = linc
    go = [row[:] for row in gi]
    direc = 1
    while True:
        sp = (sp[0] - 1, sp[1] + direc)
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        if not (0 <= sp[0] < h and 0 <= sp[1] < w):
            break
        go[sp[0]][sp[1]] = linc

    A = lambda g: np.array(g)
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: g[:, ::-1],
        lambda g: g[::-1, :],
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    gi, go = A(gi), A(go)
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    gix, gox = gi.copy(), go.copy()
    gi, go = gi.copy(), go.copy()
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    trans = []
    if numlins > 1:
        trans.append(lambda g: g[::-1, :])
    if numlins > 2:
        trans.append(lambda g: g[:, ::-1])
    if numlins > 3:
        trans.append(lambda g: g[::-1, ::-1])
    for t in trans:
        gi[t(gix) == linc] = linc
        go[t(gox) == linc] = linc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: each marked corner launches a diagonal ray that bounces between the two
    # long walls until it leaves the grid. Unit of action = one straight diagonal segment
    # of that ray (corner/wall to next wall), painted as one line with one Color op.
    # Segments are computed purely from grid shape + marker corners (never from O).
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    cset = set(corners)
    bgc = Counter(int(I[r, c]) for r in range(h) for c in range(w)
                  if (r, c) not in cset).most_common(1)[0][0]
    others = [int(v) for v in np.unique(I) if v != bgc]
    linc = others[0] if others else bgc
    seeds = [p for p in corners if int(I[p]) == linc]
    portrait = h > w

    work = I.copy()
    ops, sels = [], []
    for (r0, c0) in seeds:
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1
        r, c = r0, c0
        segs, cur = [], []
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            cur.append((r, c))
            if portrait and (c == 0 or c == w - 1):
                dc = -dc
                segs.append(cur)
                cur = []
            elif (not portrait) and (r == 0 or r == h - 1):
                dr = -dr
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            if any(work[p] != linc for p in seg):   # skip segments already drawn by another ray
                ops.append(int(linc))
                sels.append(sel_of(seg))
                for p in seg:
                    work[p] = linc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3df8b1e"
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
                                f"for task a3df8b1e"
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
                    f"Failed to build a complete episode for task a3df8b1e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3df8b1e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
