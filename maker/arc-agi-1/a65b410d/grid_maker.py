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

VARIANTS = [{"rotk": 0}, {"rotk": 1}, {"rotk": 2}, {"rotk": 3}]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 3)]
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rotk=None) -> dict:
    if rotk is None:
        rotk = random.choice([0, 1, 2, 3])
    # rotation by odd k swaps dims -> sample pre-rotation bounds accordingly
    if rotk % 2 == 1:
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    h = _unifint(diff_lb, diff_ub, (3, hb))
    w = _unifint(diff_lb, diff_ub, (3, wb))
    mpi = h // 2
    mpj = w // 2
    devi = _unifint(diff_lb, diff_ub, (0, mpi))
    devj = _unifint(diff_lb, diff_ub, (0, mpj))
    if random.choice((True, False)):
        loci, locj = devi, devj
    else:
        loci, locj = h - devi, w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)

    gi = [[bgc] * w for _ in range(h)]
    for c in range(locj + 1):
        gi[loci][c] = linc
    go = [row[:] for row in gi]

    # up-right ray from tip, each ray point swept left to col 0  -> color 3
    k = 1
    while loci - k >= 0:
        e = min(locj + k, w - 1)
        for c in range(e + 1):
            go[loci - k][c] = 3
        k += 1
    # down-left ray from tip, each ray point swept left to col 0  -> color 1
    k = 1
    while loci + k <= h - 1:
        e = locj - k
        if e >= 0:
            for c in range(e + 1):
                go[loci + k][c] = 1
        k += 1

    gi = np.rot90(np.array(gi, dtype=int), rotk).tolist()
    go = np.rot90(np.array(go, dtype=int), rotk).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # 1. line colour = rarest colour in I
    cnt = Counter(I.flatten().tolist())
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # 2. normalise orientation: rotate I until the line is a row anchored at col 0
    idx = np.arange(h * w).reshape(h, w)
    kk, J, Jidx = 0, I, idx
    for k in range(4):
        Jc = np.rot90(I, k)
        cells = np.argwhere(Jc == linc)
        if len(set(cells[:, 0].tolist())) == 1 and cells[:, 1].min() == 0:
            kk, J, Jidx = k, Jc, np.rot90(idx, k)
            break
    H, W = J.shape
    cells = np.argwhere(J == linc)
    Rl = int(cells[0, 0])
    Cmax = int(cells[:, 1].max())        # free tip of the line

    # 3. regions implied by the two diagonal rays leaving the tip, each row of a
    #    ray swept back to col 0
    def rows_of(sign):
        seq = []
        k = 1
        while 0 <= Rl + sign * k <= H - 1:
            e = Cmax + k if sign < 0 else Cmax - k
            if sign < 0:
                e = min(e, W - 1)
            elif e < 0:
                break
            seq.append((Rl + sign * k, e))
            k += 1
        return seq

    def group(seq):
        out = []
        for r, e in seq:
            if out and out[-1][1] == e:
                out[-1][0].append(r)
            else:
                out.append(([r], e))
        return out

    ops, sels = [], []

    def paint(seq, color):
        for rows, e in group(seq):
            a, b = min(rows), max(rows)
            pts = []
            for R in (a, b):
                for C in (0, e):
                    f = int(Jidx[R, C])
                    pts.append((f // w, f % w))
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            ops.append(color)
            sels.append([min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)])

    paint(rows_of(-1), 3)   # up-right wedge, from the line outward
    paint(rows_of(+1), 1)   # down-left wedge, from the line outward

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
