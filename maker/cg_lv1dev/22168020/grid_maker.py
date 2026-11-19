"""
ARC Task: 22168020 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background has to stay fixed. The rule uses each object's own
    # colour, so the object colours can be chosen freshly for every instance.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        x = random.uniform(lb, ub)
        return int(round(a + (b - a) * x))

    def connect(a, b):
        (ai, aj), (bi, bj) = a, b
        si = 0 if ai == bi else (1 if bi > ai else -1)
        sj = 0 if aj == bj else (1 if bj > aj else -1)
        if si != 0 and sj != 0 and abs(bi - ai) != abs(bj - aj):
            return set()
        n = max(abs(bi - ai), abs(bj - aj))
        return {(ai + k * si, aj + k * sj) for k in range(n + 1)}

    def is_hline(p):
        return len({r for r, _ in p}) == 1

    if bgc is None:
        bgc = random.choice(list(range(10)))
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    num = unifint(diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    succ = 0
    tr = 0
    maxtr = 6 * num
    inds = {(i, j) for i in range(h) for j in range(w)}

    def dmirror(p):
        mi = min(r for r, _ in p); mj = min(c for _, c in p)
        return {(c - mj + mi, r - mi + mj) for r, c in p}

    def cmirror(p):
        mi = min(r for r, _ in p); mj = min(c for _, c in p)
        Mi = max(r for r, _ in p); Mj = max(c for _, c in p)
        return {(Mj - c + mi, Mi - r + mj) for r, c in p}

    def hmirror(p):
        mi = min(r for r, _ in p); Mi = max(r for r, _ in p)
        return {(mi + Mi - r, c) for r, c in p}

    def vmirror(p):
        mj = min(c for _, c in p); Mj = max(c for _, c in p)
        return {(r, mj + Mj - c) for r, c in p}

    def ident(p):
        return set(p)

    while tr < maxtr and succ < num:
        d = unifint(diff_lb, diff_ub, (2, 5))
        ow = 2 * d
        if len(inds) == 0 or not remcols:
            tr += 1
            continue
        loci, locj = random.choice(sorted(inds))
        io1 = connect((loci, locj), (loci + d - 1, locj + d - 1))
        io2 = connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = set()
        for a in io:
            for b in io:
                seg = connect(a, b)
                if seg and is_hline(seg):
                    oo |= seg
        mf = random.choice((ident, dmirror, cmirror, hmirror, vmirror))
        # the same mirror must map io and oo; oo shares io's bounding box
        io_m = mf(io)
        oo_m = mf(oo)
        col = random.choice(remcols)
        if oo_m.issubset(inds):
            for r, c in io_m:
                gi[r][c] = col
            for r, c in oo_m:
                go[r][c] = col
            succ += 1
            inds = inds - oo_m
            remcols = [x for x in remcols if x != col]
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # The golf entry reads the task as: along a line, fill between the two
    # same-coloured cells (XOR toggling along each row). The maker connects
    # every same-colour pair. The two readings agree here. The diagonal pairs
    # are already drawn arms, so the only connections that add cells are the
    # straight segments between the two cells an object has on the same line.
    # So the unit is one line segment per row (or per column, if the object is
    # mirrored), painted one at a time, per object.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ex in examples:
            EI = np.asarray(ex[0], dtype=int)
            votes[Counter(EI.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = votes.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand

    G = I.copy()
    colors = sorted({int(v) for v in I.flatten().tolist() if int(v) != bgc})
    for col in colors:
        cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == col]
        # line families: rows, columns, diagonals, anti-diagonals
        families = [
            (lambda p: p[0], lambda p: p[1], lambda k, t: (k, t)),
            (lambda p: p[1], lambda p: p[0], lambda k, t: (t, k)),
            (lambda p: p[0] - p[1], lambda p: p[0], lambda k, t: (t, t - k)),
            (lambda p: p[0] + p[1], lambda p: p[0], lambda k, t: (t, k - t)),
        ]
        for key, pos, mk in families:
            groups = {}
            for p in cells:
                groups.setdefault(key(p), []).append(pos(p))
            for k in sorted(groups):
                ps = groups[k]
                if len(ps) < 2:
                    continue
                seg = [mk(k, t) for t in range(min(ps), max(ps) + 1)]
                if all(G[r, c] == col for r, c in seg):
                    continue  # this line is already fully drawn
                ops.append(col)
                sels.append(sel_of(seg))
                for r, c in seg:
                    G[r, c] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 22168020"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22168020"
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
                                f"for task 22168020"
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
                    f"Failed to build a complete episode for task 22168020 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22168020-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
