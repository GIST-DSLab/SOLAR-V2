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

ROTS = ["identity", "rot90", "rot180", "rot270"]


# ----------------------------------------------------------------------------
# 1. episode-level colors + structural plan (which of the 4 rotations)
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    try:
        return unifint(diff_lb, diff_ub, bounds)  # noqa: F821
    except NameError:
        a, b = bounds
        return random.randint(a, b)


def _apply_rot(g, rot):
    a = np.array(g, dtype=int)
    k = {"identity": 0, "rot90": 3, "rot180": 2, "rot270": 1}[rot]
    return np.rot90(a, k).tolist()


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    # canonical grid is (h+2, w); rot90/rot270 transpose it
    if rot in ("identity", "rot180"):
        h_cap, w_cap = max_h - 2, max_w
    else:
        h_cap, w_cap = max_w - 2, max_h
    h_cap = max(1, min(28, h_cap))
    w_cap = max(2, min(8, w_cap))

    h = _unifint(diff_lb, diff_ub, (1, h_cap))
    w = _unifint(diff_lb, diff_ub, (2, w_cap))

    remcols = [c for c in range(1, 10) if c != bgc]
    ccols = random.sample(remcols, w)

    gi = [list(ccols), [linc] * w] + [[bgc] * w for _ in range(h)]
    go = [row[:] for row in gi]
    for k in range(h):
        go[2 + k] = [ccols[k % w]] * w

    return {"input": _apply_rot(gi, rot), "output": _apply_rot(go, rot)}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def _detect_structure(I):
    """Locate, purely from I: the uniform separator line, the multi-color strip
    just outside it, and the ordered background lines running away from the line."""
    H, W = I.shape
    cands = []
    if H >= 3:
        cands.append(("row", 0, 1, list(range(2, H))))
        cands.append(("row", H - 1, H - 2, list(range(H - 3, -1, -1))))
    if W >= 3:
        cands.append(("col", 0, 1, list(range(2, W))))
        cands.append(("col", W - 1, W - 2, list(range(W - 3, -1, -1))))

    for kind, si, li, bgidx in cands:
        get = (lambda k: I[k, :].tolist()) if kind == "row" else (lambda k: I[:, k].tolist())
        line = get(li)
        strip = get(si)
        if len(set(line)) != 1:            # separator must be one solid color
            continue
        if len(set(strip)) < 2:            # color strip holds >= 2 distinct colors
            continue
        bgvals = set()
        ok = True
        for k in bgidx:
            v = set(get(k))
            if len(v) != 1:
                ok = False
                break
            bgvals |= v
        if not ok or len(bgvals) != 1:
            continue
        if bgvals.pop() == line[0]:
            continue
        return kind, si, li, bgidx, strip
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    det = _detect_structure(I)
    if det is not None:
        kind, si, li, bgidx, strip = det

        # propagation direction v points from the line into the background;
        # the strip is read along u = v rotated 90 CCW  (canonical: v=down, u=right)
        if kind == "row":
            v = (1, 0) if si == 0 else (-1, 0)
        else:
            v = (0, 1) if si == 0 else (0, -1)
        u = (-v[1], v[0])                       # CCW rotation of v
        reverse = (u == (0, -1)) or (u == (-1, 0))
        ccols = strip[::-1] if reverse else list(strip)
        period = len(ccols)

        # stamp one stripe per background line, walking outward from the line
        for k, idx in enumerate(bgidx):
            color = int(ccols[k % period])
            if kind == "row":
                cells = [(idx, c) for c in range(W)]
            else:
                cells = [(r, idx) for r in range(H)]
            ops.append(color)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
