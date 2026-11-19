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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    linc = random.choice(remcols)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    cols = interval(1, 10, 1)
    lim = max(1, min(max_h, max_w) - 2)
    h = unifint(diff_lb, diff_ub, (1, min(28, lim)))
    w = unifint(diff_lb, diff_ub, (2, 8))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, w)
    cc = (tuple(ccols),)
    br = canvas(linc, (1, w))
    lp = canvas(bgc, (h, w))
    gi = vconcat(vconcat(cc, br), lp)
    go = vconcat(vconcat(cc, br), lp)
    pt = hupscale(dmirror(cc), w)
    pto = asobject(pt)
    idx = 2
    while idx < h + 3:
        go = paint(go, shift(pto, (idx, 0)))
        idx += w
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Reading: stack the header strip w times beneath the separator (a w x w
    square whose every line is the header), give that square a quarter turn so
    each header colour becomes its own full stripe, then stamp the finished
    block repeatedly down the field until the grid is full.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    def canon_ok(G):
        Hc, Wc = G.shape
        if Hc < 3 or Wc < 2 or Wc > 8:
            return False
        if len(set(G[0].tolist())) != Wc:      # header: distinct colours
            return False
        if len(set(G[1].tolist())) != 1:       # separator: uniform
            return False
        rest = set(G[2:].flatten().tolist())   # plain field
        if len(rest) != 1:
            return False
        return rest.pop() not in set(G[0].tolist()) | {int(G[1][0])}

    k, Gc = None, None
    for kk in range(4):
        G = np.rot90(I, -kk)
        if canon_ok(G):
            k, Gc = kk, G
            break
    if k is None:   # fallback: weaker check
        for kk in range(4):
            G = np.rot90(I, -kk)
            if G.shape[0] >= 3 and len(set(G[1].tolist())) == 1 and len(set(G[2:].flatten().tolist())) == 1:
                k, Gc = kk, G
                break
    Hc, Wc = Gc.shape
    w, h = Wc, Hc - 2
    hdr = Gc[0].tolist()

    def M(r, c):   # canonical -> actual coordinates
        if k == 0:
            return (r, c)
        if k == 1:
            return (Wc - 1 - c, r)
        if k == 2:
            return (Hc - 1 - r, Wc - 1 - c)
        return (c, Hc - 1 - r)

    def origin(cells):
        return [min(p[0] for p in cells), min(p[1] for p in cells), 0, 0]

    def line(r):
        return [M(r, c) for c in range(w)]

    ops, sels = [], []
    if h >= w:
        # 1. copy the header strip
        ops.append(28); sels.append(sel_of(line(0)))
        # 2. stack it w times under the separator -> w x w square of header lines
        for r in range(2, 2 + w):
            ops.append(30); sels.append(origin(line(r)))
        # 3. quarter turn of the square: each header colour becomes a stripe
        block = [M(r, c) for r in range(2, 2 + w) for c in range(w)]
        ops.append(25); sels.append(sel_of(block))
        # 4. stamp the finished block down the rest of the field:
        #    a stamp whose top-left lies on the grid is pasted (overhang clipped);
        #    one that would start off-grid is continued line by line
        stamps, tail = [], None
        r0 = 2 + w
        while r0 < Hc:
            tile = [M(r, c) for r in range(r0, r0 + w) for c in range(w)]
            o = origin(tile)
            if o[0] >= 0 and o[1] >= 0:
                stamps.append(o)
            else:
                tail = r0
                break
            r0 += w
        if stamps:
            ops.append(29); sels.append(sel_of(block))
            for o in stamps:
                ops.append(30); sels.append(o)
        if tail is not None:
            for r in range(tail, Hc):
                ops.append(int(hdr[(r - 2) % w])); sels.append(sel_of(line(r)))
    else:
        # field too short for a full square: each stripe line takes its colour
        for r in range(2, Hc):
            ops.append(int(hdr[(r - 2) % w])); sels.append(sel_of(line(r)))

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
