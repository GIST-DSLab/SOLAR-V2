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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unif(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    d = random.uniform(diff_lb, diff_ub)
    v = lb + int(round(d * (ub - lb)))
    return max(lb, min(ub, v))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, **kw) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample(list(range(10)), 2)
    m = min(max_h, max_w, 30)
    w = _unif(diff_lb, diff_ub, 2, max(2, min(10, m - 1)))
    h = _unif(diff_lb, diff_ub, w + 1, max(w + 1, m))
    gi = np.full((h, w), bgc, dtype=int)
    gi[h - 1, 0] = linc
    go = gi.copy()
    r, c, direc = h - 1, 0, 1
    while True:
        r, c = r - 1, c + direc
        if c == w - 1 or c == 0:
            direc *= -1
        if not (0 <= r < h and 0 <= c < w) or go[r, c] == linc:
            break
        go[r, c] = linc
    mfs = [
        lambda g: g,
        lambda g: g.T,
        lambda g: np.rot90(g, 2).T,
        lambda g: np.fliplr(g),
        lambda g: np.flipud(g),
        lambda g: np.rot90(g, 3),
        lambda g: np.rot90(g, 2),
        lambda g: np.rot90(g, 1),
    ]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi).copy()
        go = fn(go).copy()
    gix, gox = gi.copy(), go.copy()
    numlins = _unif(diff_lb, diff_ub, 1, 4)
    mirrors = []
    if numlins > 1:
        mirrors.append(np.flipud)
    if numlins > 2:
        mirrors.append(np.fliplr)
    if numlins > 3:
        mirrors.append(lambda g: np.flipud(np.fliplr(g)))
    for f in mirrors:
        gi[f(gix) == linc] = linc
        go[f(gox) == linc] = linc
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (shared by maker and golf entry): each marked corner launches a
    # diagonal ray that advances one cell per line along the long axis and
    # bounces off the two short-side walls. The golf entry walks it line by
    # line; here each straight diagonal segment between bounces is one stroke
    # (one Color op on that segment's cells), so every op draws one line.
    from collections import Counter
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    corner_list = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    cs = set(corner_list)
    cnt = Counter(int(I[r, c]) for r in range(h) for c in range(w) if (r, c) not in cs)
    bgc = cnt.most_common(1)[0][0]
    rest = [int(v) for v in np.unique(I) if int(v) != bgc]
    linc = rest[0] if rest else bgc
    seeds = [(r, c) for (r, c) in corner_list if int(I[r, c]) == linc]
    portrait = h > w

    work = I.copy()
    ops, sels = [], []
    for (r0, c0) in seeds:
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1
        r, c = r0, c0
        segments, seg = [], []
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            seg.append((r, c))
            bounced = False
            if portrait:
                if c == 0 or c == w - 1:
                    dc = -dc
                    bounced = True
            else:
                if r == 0 or r == h - 1:
                    dr = -dr
                    bounced = True
            if bounced:
                segments.append(seg)
                seg = []
        if seg:
            segments.append(seg)
        for seg in segments:
            cells = [(a, b) for (a, b) in seg if work[a, b] != linc]
            if not cells:
                continue
            ops.append(int(linc))
            sels.append(sel_of(cells))
            for (a, b) in cells:
                work[a, b] = linc

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
