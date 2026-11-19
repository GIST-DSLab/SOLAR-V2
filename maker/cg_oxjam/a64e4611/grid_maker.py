"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"top_open": t, "sides": s}
    for t in (True, False) for s in ("L", "R", "LR")
]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        # make sure both the open-top and the closed-top case are shown
        opn = [v for v in VARIANTS if v["top_open"]]
        cls = [v for v in VARIANTS if not v["top_open"]]
        examples = [dict(random.choice(opn))]
        if n_ex > 1:
            examples.append(dict(random.choice(cls)))
        rest = [v for v in VARIANTS if v not in examples]
        examples += [dict(v) for v in random.sample(rest, max(0, n_ex - len(examples)))]
        random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec,
             top_open=None, sides=None) -> dict:
    if top_open is None or sides is None:
        v = random.choice(VARIANTS)
        top_open, sides = v["top_open"], v["sides"]

    def unif(lo, hi):
        if hi <= lo:
            return lo
        a = lo + int(round((hi - lo) * diff_lb))
        b = lo + int(round((hi - lo) * diff_ub))
        a, b = max(lo, min(a, b)), min(hi, max(a, b))
        return random.randint(a, b)

    for _ in range(200):
        h = unif(min(18, max_h), max_h)
        w = unif(min(18, max_w), max_w)
        gi = [[noisec] * w for _ in range(h)]
        cells = [(i, j) for i in range(h) for j in range(w)]
        nb = unif(int(0.4 * h * w), int(0.5 * h * w))
        for (i, j) in random.sample(cells, nb):
            gi[i][j] = bgc
        addn, addb = set(), set()
        for i in range(h - 2):
            for j in range(w - 2):
                blk = {gi[i + a][j + b] for a in range(3) for b in range(3)}
                if blk == {bgc}:
                    addn.add((i + random.randint(0, 2), j + random.randint(0, 2)))
                elif blk == {noisec}:
                    addb.add((i + random.randint(0, 2), j + random.randint(0, 2)))
        for (i, j) in addn:
            gi[i][j] = noisec
        for (i, j) in addb:
            gi[i][j] = bgc
        go = [row[:] for row in gi]

        dim = random.randint(random.randint(3, 8), 8)
        if w - dim - 4 < 3 or h < 18:
            continue
        locj = random.randint(3, w - dim - 4)
        spi = 0 if top_open else random.randint(3, h // 2)
        for j in range(locj, locj + dim):
            for i in range(spi, h):
                gi[i][j] = bgc
                go[i][j] = bgc
        for j in range(locj + 1, locj + dim - 1):
            for i in range(spi + 1 if spi > 0 else spi, h):
                go[i][j] = 3

        def add_arms(sgns, startloc, hh):
            for sgn in sgns:
                for ii in range(startloc, startloc + hh):
                    j = locj
                    while 0 <= j < w:
                        gi[ii][j] = bgc
                        if go[ii][j] != 3:
                            go[ii][j] = bgc
                        j += sgn
            for sgn in sgns:
                for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                    j = locj + dim - 2 if sgn == -1 else locj + 1
                    while 0 <= j < w:
                        go[ii][j] = 3
                        j += sgn

        sgns = {"L": (-1,), "R": (1,), "LR": (-1, 1)}[sides]
        if h - 6 < spi + 3:
            continue
        startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        hh = random.randint(3, min(8, h - startloc - 3))
        add_arms(sgns, startloc, hh)
        if len(sgns) == 1 and unif(0, 1) == 1:
            startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
            hh = random.randint(3, min(8, h - startloc - 3))
            add_arms((-sgns[0],), startloc, hh)

        # keep only instances whose corridor reads unambiguously from the input
        try:
            ops, sels = derive_operations(gi, go)
        except Exception:
            continue
        sim = np.array(gi, dtype=int)
        for op, s in zip(ops, sels):
            if op < 10:
                for r, c in s["cells"]:
                    sim[r, c] = op
        if np.array_equal(sim, np.array(go, dtype=int)):
            return {"input": gi, "output": go}
    raise RuntimeError("could not build an unambiguous instance")


def derive_operations(I, O=None, examples=None):
    # Reading (taken from the golf entry, not the maker's one-rectangle-per-band): the empty
    # corridor -- the vertical band plus every arm that runs sideways to the grid edge -- is
    # ONE region. The rule paints its inside 3, leaving a one-cell margin wherever it
    # touches the noise. Grid edges count as open. Everything is measured from I alone;
    # 3 is the colour the rule names.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # 1. The vertical corridor: the widest run of adjacent columns whose lower half is
    #    one solid colour. That colour is the background.
    half = h // 2
    solid = [bool((I[half:, j] == I[h - 1, j]).all()) for j in range(w)]
    best = None
    j = 0
    while j < w:
        if solid[j]:
            k = j
            while k + 1 < w and solid[k + 1] and I[h - 1, k + 1] == I[h - 1, j]:
                k += 1
            if best is None or k - j > best[1] - best[0]:
                best = (j, k)
            j = k + 1
        else:
            j += 1
    c0, c1 = best
    bgc = int(I[h - 1, c0])
    top = h - 1
    while top - 1 >= 0 and (I[top - 1, c0:c1 + 1] == bgc).all():
        top -= 1

    corridor = set((r, c) for r in range(top, h) for c in range(c0, c1 + 1))
    # 2. Rows that leave the corridor sideways and stay background all the way to the grid edge.
    for r in range(top, h):
        if c1 + 1 < w and (I[r, c1 + 1:] == bgc).all():
            corridor.update((r, c) for c in range(c0, w))
        if c0 > 0 and (I[r, :c0] == bgc).all():
            corridor.update((r, c) for c in range(0, c1 + 1))

    # 3. The corridor's interior: the cells with no noise among their 8 neighbours
    #    (neighbours outside the grid count as open).
    interior = []
    for (r, c) in corridor:
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in corridor:
                    ok = False
        if ok:
            interior.append((r, c))

    ops, sels = [], []
    ops.append(3)                     # Color3 over the whole corridor interior as one unit
    sels.append(sel_of(interior))
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
