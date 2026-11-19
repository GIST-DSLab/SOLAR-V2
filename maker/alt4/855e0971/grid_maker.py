"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colours ---
def sample_colors(num_examples=None) -> dict:
    """Fix the dot colour (the colour that grows into lines) and the palette the
    bars may use.  Colour 0 is kept out of the episode entirely: 0 means
    "nothing" to ARCLE's object/clipboard ops, which this maker relies on."""
    cols = list(range(1, 10))
    dotc = random.choice(cols)
    barcols = [c for c in cols if c != dotc]

    # discrete structural variant: bars stacked as rows, or (dmirror) as columns
    VARIANTS = [{"transpose": False}, {"transpose": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"dotc": dotc, "barcols": barcols, "instance_plan": plan}


# --------------------------------------------------------------- generator ---
def generate(diff_lb, diff_ub, max_h, max_w, dotc, barcols, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    def uf(bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    # the grid is built with horizontal bars and transposed afterwards,
    # so the size limits swap when the instance is transposed
    hlim = min(30, max_w if transpose else max_h)
    wlim = min(30, max_h if transpose else max_w)
    hlim = max(4, hlim)
    wlim = max(3, wlim)

    nbarsd = uf((1, 4))
    nbars = random.choice([nbarsd, 11 - nbarsd])
    nbars = max(3, nbars)
    nbars = min(nbars, hlim // 2)
    nbars = max(2, nbars)

    h = uf((nbars, hlim))
    w = uf((3, wlim))

    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[random.randrange(nbars)] += 1

    nloclbs = [random.choice([0, 1]) for _ in range(nbars)]
    if sum(nloclbs) < 2:
        i1, i2 = random.sample(range(nbars), 2)
        nloclbs[i1] = 1
        nloclbs[i2] = 1

    gi, go = [], []
    lastcol = None
    for bs, nloclb in zip(barsizes, nloclbs):
        opts = [c for c in barcols if c != lastcol]
        col = random.choice(opts)
        gim = [[col] * w for _ in range(bs)]
        gom = [[col] * w for _ in range(bs)]
        nl = uf((nloclb, max(nloclb, w // 2)))
        nl = min(nl, w)
        for jj in random.sample(range(w), nl):
            ii = random.randrange(bs)
            gim[ii][jj] = dotc
            for rr in range(bs):
                gom[rr][jj] = dotc
        lastcol = col
        gi.extend(gim)
        go.extend(gom)

    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ------------------------------------------------------------- derivation ---
def derive_operations(I, O, examples=None):
    """Rule: the grid is a stack of solid-colour bars; inside each bar every dot
    grows into a full line across the bar.  Equivalently: each bar's dots are
    gathered onto the bar's first line, and that line is then stamped over the
    whole bar.  That is what is emitted -- Move (a dot travels to the bar's
    first line) + CopyO/Paste (that line is replicated through the bar)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # ---- which colour is the dot colour (the convention the examples fix) ----
    dotc = None
    if examples:
        votes = Counter()
        for pair in examples:
            ei = np.asarray(pair[0], dtype=int)
            eo = np.asarray(pair[1], dtype=int)
            ci = Counter(ei.flatten().tolist())
            co = Counter(eo.flatten().tolist())
            grew = [c for c in co if co[c] > ci.get(c, 0)]
            if len(grew) == 1:
                votes[grew[0]] += 1
        for c, _ in votes.most_common():
            if bool((I == c).any()):
                dotc = c
                break
    if dotc is None:
        # fallback: the dots are the colour whose cells sit alone (no same-colour
        # 8-neighbour); bar cells always touch their own kind
        best = -1
        for c in sorted(set(I.flatten().tolist())):
            cnt = 0
            for r, cc in zip(*np.where(I == c)):
                iso = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr or dc:
                            rr, c2 = r + dr, cc + dc
                            if 0 <= rr < h and 0 <= c2 < w and I[rr, c2] == c:
                                iso = False
                if iso:
                    cnt += 1
            if cnt > best:
                best, dotc = cnt, c

    # ---- orientation: bars are rows iff no row mixes more than 2 colours ----
    horizontal = all(len(set(I[r].tolist())) <= 2 for r in range(h))
    T = I if horizontal else I.T
    na, nb = T.shape

    def real(a, b):
        return (a, b) if horizontal else (b, a)

    # ---- cut the stack into bars (neighbouring bars always differ in colour) --
    barcol = []
    for a in range(na):
        vals = [v for v in T[a].tolist() if v != dotc]
        barcol.append(Counter(vals).most_common(1)[0][0] if vals else dotc)
    bars = []
    start = 0
    for a in range(1, na + 1):
        if a == na or barcol[a] != barcol[a - 1]:
            bars.append((start, a - 1))
            start = a

    MOVE = 20 if horizontal else 23  # toward the bar's first line (up / left)

    W = I.copy()
    ops, sels = [], []

    def line(a):
        return [int(W[real(a, b)]) for b in range(nb)]

    for (a0, a1) in bars:
        dotpos = [b for b in range(nb)
                  if any(T[a, b] == dotc for a in range(a0, a1 + 1))]
        if not dotpos:
            continue  # a bar without dots is already finished

        # 1. gather: every dot travels to the bar's first line
        present = set(b for b in range(nb) if W[real(a0, b)] == dotc)
        for b in dotpos:
            if b in present:
                continue
            src = [a for a in range(a0 + 1, a1 + 1) if T[a, b] == dotc]
            if not src:
                continue
            a = src[0]
            steps = a - a0
            ops.append(MOVE)
            sels.append(sel_of([real(a, b)]))          # grab this dot
            for _ in range(steps - 1):
                ops.append(MOVE)
                sels.append(sel_of([]))                # keep it gliding
            W[real(a, b)] = 0                          # ARCLE leaves the origin at 0
            W[real(a0, b)] = dotc
            present.add(b)

        # 2. stamp: the gathered line is replicated over the rest of the bar
        master = line(a0)
        targets = [a for a in range(a0 + 1, a1 + 1) if line(a) != master]
        if targets:
            # selection is exactly the bar's first full row / full column
            ops.append(29)
            sels.append([a0, 0, 0, nb - 1] if horizontal else [0, a0, nb - 1, 0])
            for a in targets:
                ops.append(30)
                sels.append([a, 0, 0, 0] if horizontal else [0, a, 0, 0])
                for b in range(nb):
                    W[real(a, b)] = master[b]

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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
