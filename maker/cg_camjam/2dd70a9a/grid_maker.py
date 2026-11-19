"""
ARC Task: 2dd70a9a (RE-ARC) — LLM-generated grid_maker
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


def _unif(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    return random.randint(int(round(a)), max(int(round(a)), int(round(b))))


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    if c0 == c1:
        return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
    return set()


MODES = ["vertical_legs", "horizontal_legs"]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    ex = list(MODES) if n_ex >= 2 else random.sample(MODES, n_ex)
    ex += [random.choice(MODES) for _ in range(n_ex - len(ex))]
    random.shuffle(ex)
    plan = [{"mode": m} for m in ex] + [{"mode": random.choice(ex)}]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def _candidates(I):
    """All turning lines consistent with the rule, as
    (strict, cellfn, o, conn, leg2, leg3, L2, L3)."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    m3 = [tuple(map(int, p)) for p in np.argwhere(I == 3)]
    m2 = [tuple(map(int, p)) for p in np.argwhere(I == 2)]
    if len(m3) < 2 or len(m2) < 2:
        return []
    horiz = len({r for r, _ in m2}) == 1
    def cell(line, pos):
        return (line, pos) if horiz else (pos, line)
    nlines, npos = (h, w) if horiz else (w, h)
    def g(line, pos):
        return I[cell(line, pos)]
    L3 = m3[0][0] if horiz else m3[0][1]
    L2 = m2[0][0] if horiz else m2[0][1]
    pos3 = [c if horiz else r for r, c in m3]
    pos2 = [c if horiz else r for r, c in m2]
    if L2 == L3:
        return []
    s = 1 if L2 > L3 else -1
    cands = []
    for d in (1, -1):
        # ray out of the 3-domino's free end, along its own line, until blocked
        head = max(pos3) if d == 1 else min(pos3)
        p = head + d
        leg3 = []
        while 0 <= p < npos and g(L3, p) == bgc:
            leg3.append(p); p += d
        if not leg3 or not (0 <= p < npos):
            continue
        o = leg3[-1]                       # the turning line
        conn = list(range(L3 + s, L2, s))  # connector between the two domino lines
        if any(g(l, o) != bgc for l in conn):
            continue
        if o > max(pos2):
            leg2 = list(range(max(pos2) + 1, o + 1))
        elif o < min(pos2):
            leg2 = list(range(min(pos2) - 1, o - 1, -1))
        else:
            continue
        if any(g(L2, q) != bgc for q in leg2):
            continue
        beyond = L2 + s
        strict = bool(0 <= beyond < nlines and g(beyond, o) != bgc)
        cands.append((strict, cell, o, conn, leg2, leg3, L2, L3))
    cands.sort(key=lambda t: not t[0])
    return cands


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, mode=None) -> dict:
    if mode is None:
        mode = random.choice(MODES)
    h = _unif(diff_lb, diff_ub, 10, max(10, max_h))
    w = _unif(diff_lb, diff_ub, 10, max(10, max_w))
    if mode == "vertical_legs":
        oh = _unif(diff_lb, diff_ub, 5, h - 2)
        ow = _unif(diff_lb, diff_ub, 3, w - 2)
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        hli = random.randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = _connect((loci + oh - 1, locj), (hli, locj))
        ln2 = _connect((hli, locj), (hli, locj + ow - 1))
        ln3 = _connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = _unif(diff_lb, diff_ub, 3, h - 2)
        ow = _unif(diff_lb, diff_ub, 3, w - 2)
        loci = random.randint(1, h - oh - 1)
        locj = random.randint(1, w - ow - 1)
        if random.choice((True, False)):
            sp1j = random.randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = random.randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = _connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = _connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = _connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for g in (gi, go):
        for r, c in sp: g[r][c] = 3
        for r, c in ep: g[r][c] = 2
        for r, c in (bp1, bp2): g[r][c] = fgc
    for r, c in (ln1 | ln2 | ln3):
        go[r][c] = 3
    inds = [(r, c) for r in range(h) for c in range(w) if go[r][c] == bgc]
    base_i = [row[:] for row in gi]
    base_o = [row[:] for row in go]
    namt = _unif(diff_lb, diff_ub, 0, len(inds) // 2)
    for _ in range(100):
        gi = [row[:] for row in base_i]
        go = [row[:] for row in base_o]
        for r, c in random.sample(inds, namt):
            gi[r][c] = fgc
            go[r][c] = fgc
        # keep only instances where the rule picks a single, correct path
        cands = _candidates(gi)
        if len(cands) >= 1 and sum(t[0] for t in cands) == 1 and cands[0][0]:
            _, cell, o, conn, leg2, leg3, L2, L3 = cands[0]
            pr = ({cell(l, o) for l in conn} | {cell(L2, q) for q in leg2}
                  | {cell(L3, q) for q in leg3})
            if pr == ((ln1 | ln2 | ln3) - sp - ep):
                break
        namt = max(0, namt - 1 - namt // 10)
    else:
        raise ValueError("ambiguous instance")
    mfs = [lambda a: a, lambda a: a.T, lambda a: np.rot90(a, 2).T,
           np.fliplr, np.flipud, lambda a: np.rot90(a, 3),
           lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 1)]
    gi = np.array(gi); go = np.array(go)
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi); go = fn(go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the code-golf entry): the two dominoes lie on parallel
    lines.  A ray shot out of the 3-domino along its line until it is blocked
    fixes ONE turning line, perpendicular to the dominoes.  That turning line is
    the connector, running straight from the 3-domino's line to the 2-domino's
    line.  Then each domino stretches its own leg along its line to meet the
    connector.  Painted as: connector, the 2-domino's leg, the 3-domino's leg.
    Only I is read.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    cands = _candidates(I)
    if cands:
        _, cell, o, conn, leg2, leg3, L2, L3 = cands[0]
        # 1) connector: the turning line, between the 3-line and the 2-line
        ops.append(3); sels.append(sel_of([cell(l, o) for l in conn]))
        # 2) the 2-domino's leg: out of the 2-domino along its line to the connector
        ops.append(3); sels.append(sel_of([cell(L2, q) for q in leg2]))
        # 3) the 3-domino's leg: out of the 3-domino along its line to its blocker
        ops.append(3); sels.append(sel_of([cell(L3, q) for q in leg3]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2dd70a9a"
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
                                f"for task 2dd70a9a"
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
                    f"Failed to build a complete episode for task 2dd70a9a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2dd70a9a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
