"""
ARC Task: 5521c0d9 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    variants = [{"rot": k} for k in range(4)]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rot=None) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    if rot is None:
        rot = random.randrange(4)
    mh = max(4, min(30, int(max_h)))
    mw = max(4, min(30, int(max_w)))
    if rot % 2 == 1 and (mw < 4 or mh < 6):
        rot = 0
    if rot % 2 == 1:
        hb = (4, max(4, mw)); wb = (6, max(6, mh))
    else:
        hb = (4, max(4, mh)); wb = (6, max(6, mw))
    h = max(4, _unifint(diff_lb, diff_ub, hb))
    w = max(6, _unifint(diff_lb, diff_ub, wb))

    inds = list(range(w))
    speps = None
    for _ in range(200):
        nobjs = _unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
        cand = random.sample(inds, min(2 * nobjs, w))
        if len(cand) % 2 == 1:
            cand = cand[:-1]
        if not cand or 0 in cand or (w - 1) in cand:
            continue
        speps = sorted(cand)
        break
    if speps is None:
        speps = [1, 2]
    starts = speps[::2]
    ends = speps[1::2]

    remcols = [c for c in range(10) if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, max(2, min(9, ncols)))

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    forb = -1
    for sp, ep in zip(starts, ends):
        pool = [c for c in ccols if c != forb] or list(ccols)
        col = random.choice(pool)
        forb = col
        hdev = _unifint(diff_lb, diff_ub, (0, h // 2))
        hei = random.choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        gi[h - hei:h, sp:ep + 1] = col
        go[max(0, h - 2 * hei):h - hei, sp:ep + 1] = col
    if rot:
        gi = np.rot90(gi, k=rot)
        go = np.rot90(go, k=rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Same rule (every bar slides away from its edge by its own thickness),
    different unit: bars are grouped by THICKNESS. All bars of one thickness
    form a single band sharing one displacement, so the whole band is grabbed
    once and slid together; then the band's old footprint is restored to
    background in one stroke."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # 3 of the 4 border lines are always fully background
    border = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
              + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    bgc = Counter(int(I[r, c]) for r, c in border).most_common(1)[0][0]

    lines = [
        ("bottom", [(h - 1, c) for c in range(w)]),
        ("top",    [(0, c) for c in range(w)]),
        ("left",   [(r, 0) for r in range(h)]),
        ("right",  [(r, w - 1) for r in range(h)]),
    ]
    side = next((n for n, cells in lines if any(I[r, c] != bgc for r, c in cells)), None)
    if side is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # canonical view: bars stand on the last row and travel upward
    if side == "bottom":
        V = I; back = lambda i, j: (i, j); mop = 20
    elif side == "top":
        V = I[::-1, :]; back = lambda i, j: (h - 1 - i, j); mop = 21
    elif side == "left":
        V = np.rot90(I, 1); back = lambda i, j: (j, w - 1 - i); mop = 22
    else:
        V = np.rot90(I, 3); back = lambda i, j: (h - 1 - j, i); mop = 23
    H, W = V.shape

    # per column: bar colour and thickness, read from I; group by thickness
    bands = {}
    for c in range(W):
        col = int(V[H - 1, c])
        if col == bgc:
            continue
        r = H - 1
        while r - 1 >= 0 and V[r - 1, c] == col:
            r -= 1
        ht = H - r
        b = bands.setdefault(ht, {"move": [], "zero": []})
        if col == 0:
            b["zero"].append(c)
        else:
            b["move"].extend((i, c) for i in range(r, H))

    for ht in sorted(bands):
        r = H - ht
        band = bands[ht]
        if band["move"]:
            src = [back(i, j) for i, j in band["move"]]
            ops.append(mop); sels.append(sel_of(src))          # grab the whole band
            for _ in range(ht - 1):
                ops.append(mop); sels.append(sel_of([]))       # keep it gliding (ARCLE clips off-grid)
            if bgc != 0:
                # destination rows all lie beyond the footprint: whole footprint vacated
                ops.append(bgc); sels.append(sel_of(src))
        if band["zero"]:
            # colour 0 is transparent to object ops: erase and redraw these columns
            zsrc = [back(i, j) for j in band["zero"] for i in range(r, H)]
            ops.append(bgc); sels.append(sel_of(zsrc))
            zdst = [back(i, j) for j in band["zero"] for i in range(max(0, r - ht), r)]
            ops.append(0); sels.append(sel_of(zdst))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5521c0d9"
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
                                f"for task 5521c0d9"
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
                    f"Failed to build a complete episode for task 5521c0d9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5521c0d9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
