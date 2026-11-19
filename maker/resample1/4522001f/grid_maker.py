"""
ARC Task: 4522001f (RE-ARC) — LLM-generated grid_maker
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
    bgc, sqc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int((b - a) * lb)
    hi = a + int(round((b - a) * ub))
    return random.randint(lo, max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, sqc=1, dotc=2) -> dict:
    h = _unifint(diff_lb, diff_ub, (3, max(3, min(10, max_h // 3))))
    w = _unifint(diff_lb, diff_ub, (3, max(3, min(10, max_w // 3))))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((3 * h, 3 * w), bgc, dtype=int)
    sqi = [(dotc, (1, 1)), (sqc, (0, 0)), (sqc, (0, 1)), (sqc, (1, 0))]

    def place(loc):
        for col, (i, j) in sqi:
            gi[loc[0] + i, loc[1] + j] = col
        for k0 in (0, 4):
            go[loc[0] + k0:loc[0] + k0 + 4, loc[1] + k0:loc[1] + k0 + 4] = sqc

    def cells(loc):
        return {(loc[0] + i, loc[1] + j) for _, (i, j) in sqi}

    def dn(cs):
        out = set()
        for r, c in cs:
            out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        return out

    loc = (random.randint(0, min(h - 2, 3 * h - 8)), random.randint(0, min(w - 2, 3 * w - 8)))
    place(loc)
    iinds = {(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc} - dn(cells(loc))
    noccs = _unifint(diff_lb, diff_ub, (0, (h * w) // 9))
    succ = tr = 0
    while tr < 10 * noccs and succ < noccs:
        tr += 1
        cands = [ij for ij in iinds if ij[0] <= h - 2 and ij[1] <= w - 2]
        if not cands:
            break
        loc = random.choice(sorted(cands))
        pc = cells(loc)
        if pc.issubset(iinds):
            succ += 1
            iinds = (iinds - pc) - dn(pc)
            place(loc)
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    H, W = 3 * h, 3 * w
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    cnt = Counter(v for v in I.flatten().tolist() if v != bgc)
    cols = sorted(cnt, key=lambda c: cnt[c])
    dotc, sqc = cols[0], cols[-1]          # each L has 3 square-colour cells, 1 dot

    # each object: corner cell (opposite the dot) and direction corner -> dot
    objs = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != dotc:
                continue
            for a in (-1, 1):
                for b in (-1, 1):
                    pts = [(r + a, c), (r, c + b), (r + a, c + b)]
                    if all(0 <= x < h and 0 <= y < w and I[x, y] == sqc for x, y in pts):
                        objs.append(((r + a, c + b), (-a, -b)))
    dr, dc = objs[0][1]
    offr = 2 * h if dr < 0 else 0
    offc = 2 * w if dc < 0 else 0

    ops, sels = [], []
    G = np.zeros((H, W), dtype=int)
    # expand canvas to 3h x 3w (whole rectangle intended)
    ops.append(33); sels.append([0, 0, H - 1, W - 1])
    G[:h, :w] = I
    if offr == 0 and offc == 0:
        if bgc != 0:
            pad = [(r, c) for r in range(H) for c in range(W) if not (r < h and c < w)]
            ops.append(bgc); sels.append(sel_of(pad))
            for r, c in pad:
                G[r, c] = bgc
    else:
        if bgc != 0:
            # whole canvas becomes background (full rectangle intended)
            ops.append(bgc); sels.append([0, 0, H - 1, W - 1])
            G[:] = bgc
        else:
            nz = [(r, c) for r in range(h) for c in range(w) if G[r, c] != 0]
            ops.append(0); sels.append(sel_of(nz))
            for r, c in nz:
                G[r, c] = 0
        # copy whole input rectangle, paste at the corner the pattern grows from
        ops.append(28); sels.append([0, 0, h - 1, w - 1])
        ops.append(30); sels.append([offr, offc, 0, 0])
        for r in range(h):
            for c in range(w):
                if I[r, c] != 0:
                    G[offr + r, offc + c] = I[r, c]

    # grow each L into two diagonal 4x4 squares toward its dot
    for (cr, cc), (a, b) in objs:
        r0, c0 = cr + offr, cc + offc
        for k0 in (0, 4):
            sq = [(r0 + a * (k0 + i), c0 + b * (k0 + j)) for i in range(4) for j in range(4)]
            sq = [(r, c) for r, c in sq if G[r, c] != sqc]
            if sq:
                ops.append(sqc); sels.append(sel_of(sq))
                for r, c in sq:
                    G[r, c] = sqc

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4522001f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4522001f"
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
                                f"for task 4522001f"
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
                    f"Failed to build a complete episode for task 4522001f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4522001f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
