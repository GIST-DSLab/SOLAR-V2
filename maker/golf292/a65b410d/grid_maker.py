"""
ARC Task: a65b410d (RE-ARC) — LLM-generated grid_maker
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
    colopts = [c for c in range(10) if c not in (1, 3)]
    bgc = random.choice(colopts)
    linc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kw) -> dict:
    def unifint(bounds):
        a, b = bounds
        if b <= a:
            return a
        d = random.uniform(diff_lb, diff_ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    h = unifint((3, max(3, max_h)))
    w = unifint((3, max(3, max_w)))
    mpi, mpj = h // 2, w // 2
    devi = unifint((0, mpi))
    devj = unifint((0, mpj))
    if random.choice((True, False)):
        loci, locj = devi, devj
    else:
        loci, locj = h - devi, w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci, 0:locj + 1] = linc
    go = gi.copy()
    k = 1
    while loci + k < h and locj - k >= 0:          # blue wedge below
        go[loci + k, 0:locj - k + 1] = 1
        k += 1
    k = 1
    while loci - k >= 0:                             # green wedge above
        go[loci - k, 0:min(w - 1, locj + k) + 1] = 3
        k += 1
    rk = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, rk)
    go = np.rot90(go, rk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # line colour = rarest colour of I
    cnt = Counter(I.flatten().tolist())
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # normalise orientation: line becomes a row starting at col 0
    idx = np.arange(h * w).reshape(h, w)
    J, Jidx = I, idx
    for k in range(4):
        Jc = np.rot90(I, k)
        cells = np.argwhere(Jc == linc)
        if len(set(cells[:, 0].tolist())) == 1 and cells[:, 1].min() == 0:
            J, Jidx = Jc, np.rot90(idx, k)
            break
    H, W = J.shape
    cells = np.argwhere(J == linc)
    Rl = int(cells[0, 0])
    Cmax = int(cells[:, 1].max())

    def rect(r0, r1, c0, c1):
        """normalized rectangle -> actual bbox [r, c, dh, dw]"""
        pts = []
        for R in (r0, r1):
            for C in (c0, c1):
                f = int(Jidx[R, C])
                pts.append((f // w, f % w))
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]

    def origin(r0, r1, c0, c1):
        b = rect(r0, r1, c0, c1)
        return [b[0], b[1], 0, 0]

    ops, sels = [], []

    # ---- green wedge (above line): row Rl-k spans cols 0..min(Cmax+k, W-1)
    Ls = min(Rl, max(0, W - 2 - Cmax))       # staircase rows (not full width)
    row = lambda k: Rl - k
    built = 0
    while built < Ls:
        # the col-0 paste leaves bgc right of each shorter row; the shifted
        # paste must reach back over it, which bounds the step size n
        n = min(built, Ls - built, (Cmax + built + 2) // 2)
        if n > 3:
            # copy the n outermost finished rows (whole rectangle), paste one
            # step further out at col 0 and shifted right by n: the staircase
            # extends itself
            ks, ke = built + 1 - n, built
            X = Cmax + built
            ops.append(29); sels.append(rect(row(ke), row(ks), 0, X))
            ops.append(30); sels.append(origin(row(ke + n), row(ks + n), 0, X))
            ops.append(30); sels.append(origin(row(ke + n), row(ks + n), n, n + X))
            built += n
        else:
            k = built + 1
            ops.append(3); sels.append(rect(row(k), row(k), 0, Cmax + k))  # full row segment
            built += 1
    if Rl > Ls:   # remaining rows reach the far edge: one full rectangle
        ops.append(3); sels.append(rect(0, row(Ls + 1), 0, W - 1))

    # ---- blue wedge (below line): row Rl+k spans cols 0..Cmax-k
    Lb = min(H - 1 - Rl, Cmax)
    rowb = lambda k: Rl + k
    built = 0
    while built < Lb:
        n = min(built, Lb - built)
        if n > 2:
            # copy the n outermost finished rows minus their first n cols
            # (a rectangle), paste n rows further out at col 0: staircase
            # steps left
            ks, ke = built + 1 - n, built
            X = Cmax - ks
            ops.append(29); sels.append(rect(rowb(ks), rowb(ke), n, X))
            ops.append(30); sels.append(origin(rowb(ks + n), rowb(ke + n), 0, X - n))
            built += n
        else:
            k = built + 1
            ops.append(1); sels.append(rect(rowb(k), rowb(k), 0, Cmax - k))  # full row segment
            built += 1

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a65b410d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a65b410d"
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
                                f"for task a65b410d"
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
                    f"Failed to build a complete episode for task a65b410d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a65b410d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
