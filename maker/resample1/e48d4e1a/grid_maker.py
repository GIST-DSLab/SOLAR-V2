"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]

def sample_colors(num_examples=None) -> dict:
    # line colour must be non-zero: ARCLE Move cannot carry 0-valued cells
    fgc = random.choice(range(1, 10))
    bgc, dotc = random.sample([c for c in range(10) if c != fgc], 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS] + [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, dotc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a: b = a
        lo = a + round(lb * (b - a)); hi = a + round(ub * (b - a))
        return random.randint(min(lo, hi), max(lo, hi))
    if rot is None:
        rot = random.choice(ROTS)
    side = max_h if rot % 2 == 0 else max_w
    other = max_w if rot % 2 == 0 else max_h
    h = unifint(diff_lb, diff_ub, (3, min(30, side)))
    w = unifint(diff_lb, diff_ub, (3, min(30, other)))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = [(i, j) for i in range(loci) for j in range(locj)]
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = random.sample(inds, nn)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, :] = fgc; gi[:, locj] = fgc
    for (i, j) in ss:
        gi[i, j] = dotc
    go[loci + nn, :] = fgc; go[:, locj + nn] = fgc
    gi = np.rot90(gi, -rot); go = np.rot90(go, -rot)
    return {"input": [list(map(int, r)) for r in gi], "output": [list(map(int, r)) for r in go]}

def _line_colors(G):
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    res = set()
    for c in set(G.flatten().tolist()):
        nr = sum(1 for r in range(h) if (G[r] == c).all())
        nc = sum(1 for k in range(w) if (G[:, k] == c).all())
        if nr == 1 and nc == 1:
            res.add(c)
    return res

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # line colour: the colour forming exactly one full row and one full column in all demos + I
    grids = [I]
    for ex in (examples or []):
        grids.append(np.asarray(ex[0], dtype=int)); grids.append(np.asarray(ex[1], dtype=int))
    cands = None
    for G in grids:
        s = _line_colors(G)
        cands = s if cands is None else (cands & s)
    cands = sorted(cands) if cands else sorted(_line_colors(I))
    if len(cands) > 1:
        pref = [c for c in cands if (I == c).sum() == hi + wi - 1]
        cands = pref or cands
    fgc = cands[0]
    # background: the only other colour in demo outputs
    bgc = None
    if examples:
        oc = set()
        for ex in examples:
            oc |= set(np.asarray(ex[1]).flatten().tolist())
        oc.discard(fgc)
        if len(oc) == 1:
            bgc = oc.pop()
    if bgc is None:
        bgc = Counter([v for v in I.flatten().tolist() if v != fgc]).most_common(1)[0][0]
    loci = [r for r in range(hi) if (I[r] == fgc).all()][0]
    locj = [c for c in range(wi) if (I[:, c] == fgc).all()][0]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] not in (fgc, bgc)]
    nn = len(dots)
    ops, sels = [], []
    if nn and fgc == 0:
        # 0-coloured lines cannot be grabbed by Move; redraw them instead
        ops.append(bgc); sels.append(sel_of(dots))
        dr = 1 if dots[0][0] < loci else -1
        dc = 1 if dots[0][1] < locj else -1
        nr, nc = loci + dr * nn, locj + dc * nn
        ops.append(bgc); sels.append(sel_of([(loci, c) for c in range(wi)] + [(r, locj) for r in range(hi) if r != loci]))
        ops.append(0); sels.append(sel_of([(nr, c) for c in range(wi)] + [(r, nc) for r in range(hi) if r != nr]))
    elif nn:
        # erase the dots
        ops.append(bgc); sels.append(sel_of(dots))
        # lines move away from the dots, by as many cells as there are dots
        dr = 1 if dots[0][0] < loci else -1
        dc = 1 if dots[0][1] < locj else -1
        # horizontal line (crossing cell stays with the vertical line)
        src = [(loci, c) for c in range(wi) if c != locj]
        for k in range(nn):
            ops.append(21 if dr > 0 else 20); sels.append(sel_of(src if k == 0 else []))
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of(src))
        nr = loci + dr * nn
        # vertical line (crossing with the moved horizontal line stays put)
        src = [(r, locj) for r in range(hi) if r != nr]
        for k in range(nn):
            ops.append(22 if dc > 0 else 23); sels.append(sel_of(src if k == 0 else []))
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of(src))
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
