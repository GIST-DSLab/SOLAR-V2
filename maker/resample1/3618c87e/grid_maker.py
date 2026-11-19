"""
ARC Task: 3618c87e (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    # dot colour non-zero so the dot can be carried by Move (0 is transparent)
    dotc = random.choice([c for c in cols if c not in (bgc, linc, 0)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, linc=1, dotc=2, rot=None) -> dict:
    from dsl import canvas, connect, fill, interval, remove, rot90, rot180, rot270, identity
    try:
        from utils import unifint
    except ImportError:
        def unifint(lb, ub, bounds):
            a, b = bounds
            d = random.uniform(lb, ub)
            return min(b, max(a, round(a + (b - a) * d)))
    if rot is None:
        rot = random.choice(ROTS)
    if rot in (1, 3):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    hb = max(4, min(30, hb)); wb = max(4, min(30, wb))
    h = unifint(diff_lb, diff_ub, (4, hb))
    w = unifint(diff_lb, diff_ub, (4, wb))
    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, w // 2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch - 1, opts)
        opts = remove(ch + 1, opts)
    gi = fill(c, linc, ln)
    go = fill(c, linc, ln)
    for j in locs:
        hh = random.randint(1, h - 3)
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        go = fill(go, linc, lnx)
        gi = fill(gi, dotc, {(hh + 1, j)})
        go = fill(go, dotc, {(0, j)})
    rotf = [identity, rot90, rot180, rot270][rot]
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Four edges: (name, cells, direction toward that edge (dr, dc), move op)
    edges = [
        ("top", I[0, :], (-1, 0), 20),
        ("bottom", I[hi - 1, :], (1, 0), 21),
        ("left", I[:, 0], (0, -1), 23),
        ("right", I[:, wi - 1], (0, 1), 22),
    ]
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    edict = {e[0]: e for e in edges}
    base = None
    for name, line, d, op in edges:
        if len(set(line.tolist())) != 1:
            continue
        oline = edict[opposite[name]][1]
        if len(set(oline.tolist())) != 1 or oline[0] == line[0]:
            continue
        L, B = int(line[0]), int(oline[0])
        others = [c for c in set(I.flatten().tolist()) if c not in (L, B)]
        if len(others) != 1:
            continue
        D = others[0]
        dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == D]
        ok = all(0 <= r + d[0] < hi and 0 <= c + d[1] < wi and I[r + d[0], c + d[1]] == L
                 for r, c in dots)
        # every perpendicular lane carrying the line colour off the base is a spike
        # and must end in a dot (bgc lanes off the opposite edge never do)
        if ok:
            if name in ("top", "bottom"):
                lanes = [I[1:, k] if name == "top" else I[:-1, k] for k in range(wi)]
            else:
                lanes = [I[k, 1:] if name == "left" else I[k, :-1] for k in range(hi)]
            ok = all((D in ln.tolist()) for ln in lanes if L in ln.tolist())
        if ok:
            base = (name, d, op, L, B, D, dots)
            break
    ops, sels = [], []
    if base is not None:
        name, (dr, dc), mop, L, bgc, D, dots = base
        # each dot slides along its spike back onto the base line
        dots.sort()
        for (r, c) in dots:
            if name in ("top", "bottom"):
                steps = r if name == "top" else hi - 1 - r
            else:
                steps = c if name == "left" else wi - 1 - c
            if D == 0:
                # colour 0 cannot be carried by Move: paint instead
                ops.append(bgc); sels.append(sel_of([(r, c)]))
                ops.append(D); sels.append(sel_of([(r + dr * steps, c + dc * steps)]))
                continue
            ops.append(mop); sels.append(sel_of([(r, c)]))      # grab the dot
            for _ in range(steps - 1):
                ops.append(mop); sels.append(sel_of([]))        # continue same grab
            if bgc != 0:
                ops.append(bgc); sels.append(sel_of([(r, c)]))  # repair vacated footprint
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
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
                                f"for task 3618c87e"
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
                    f"Failed to build a complete episode for task 3618c87e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3618c87e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
