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

ROTS = [0, 1, 2, 3]  # number of np.rot90 quarter turns applied to the canonical layout


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    linc = random.choice(remcols)
    perm = list(remcols)
    random.shuffle(perm)            # header colours: first w entries of this order
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": k} for k in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "linc": linc, "ccol_order": perm, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None,
             ccol_order=None, rot=None) -> dict:
    from random import randint
    cols = list(range(1, 10))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    if linc is None or linc == bgc:
        linc = random.choice(remcols)
    if ccol_order is None:
        ccol_order = random.sample(remcols, len(remcols))
    ccol_order = [c for c in ccol_order if c != bgc]
    if rot is None:
        rot = random.choice(ROTS)
    lim = min(max_h, max_w)
    hmax = max(1, min(28, lim - 2))
    wmax = max(2, min(8, lim))

    def uni(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + int(round(d * (hi - lo)))))

    h = uni(1, hmax)
    w = uni(2, wmax)
    ccols = ccol_order[:w]
    gi = np.full((h + 2, w), bgc, dtype=int)
    gi[0, :] = ccols
    gi[1, :] = linc
    go = gi.copy()
    for r in range(2, h + 2):
        go[r, :] = ccols[(r - 2) % w]
    gi = np.rot90(gi, rot)
    go = np.rot90(go, rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: every stripe line beyond the separator is one unit. Line k (counting
    from the separator outward) takes the colour of header cell k mod w and is
    painted whole in that colour. One Color op per line, going outward from the header.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    def canon_ok(G):
        Hc, Wc = G.shape
        if Hc < 3 or Wc < 2:
            return False
        if len(set(G[0].tolist())) != Wc:      # header strip: all distinct colours
            return False
        if len(set(G[1].tolist())) != 1:       # separator line: uniform
            return False
        rest = set(G[2:].flatten().tolist())   # uniform background field
        return len(rest) == 1 and next(iter(rest)) != int(G[1][0])

    k = None
    for kk in range(4):
        if canon_ok(np.rot90(I, -kk)):
            k = kk
            break
    if k is None:
        k = 0
    Gc = np.rot90(I, -k)
    Hc, Wc = Gc.shape
    ccols = Gc[0].tolist()

    def M(r, c):  # canonical coords -> actual grid coords
        if k == 0:
            return (r, c)
        if k == 1:
            return (Wc - 1 - c, r)
        if k == 2:
            return (Hc - 1 - r, Wc - 1 - c)
        return (c, Hc - 1 - r)

    ops, sels = [], []
    for r in range(2, Hc):
        color = int(ccols[(r - 2) % Wc])
        line = [M(r, c) for c in range(Wc)]   # the whole stripe line
        ops.append(color)
        sels.append(sel_of(line))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])  # the whole grid, which is a full rectangle
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
