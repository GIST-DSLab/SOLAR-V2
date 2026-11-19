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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kw) -> dict:
    # Reject the rare noise layouts where the 3->2 path is not uniquely determined.
    ex = None
    for _ in range(50):
        ex = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, fgc)
        try:
            ops, sels = derive_operations(ex['input'], ex['output'])
            G = np.array(ex['input'], dtype=int)
            for o, s in zip(ops, sels):
                if o == 3:
                    for r, c in s['cells']:
                        G[r, c] = 3
            if (G == np.array(ex['output'])).all():
                return ex
        except Exception:
            pass
    return ex


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, fgc):
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    if choice((True, False)):
        oh = unifint(diff_lb, diff_ub, (5, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        hli = randint(loci + 2, loci + oh - 3)
        sp = {(loci + oh - 1, locj), (loci + oh - 2, locj)}
        ep = {(loci, locj + ow - 1), (loci + 1, locj + ow - 1)}
        bp1 = (hli - 1, locj)
        bp2 = (hli, locj + ow)
        ln1 = connect((loci + oh - 1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj + ow - 1))
        ln3 = connect((hli, locj + ow - 1), (loci + 2, locj + ow - 1))
    else:
        oh = unifint(diff_lb, diff_ub, (3, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        if choice((True, False)):
            sp1j = randint(locj, locj + ow - 3)
            ep1j = locj
        else:
            ep1j = randint(locj, locj + ow - 3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j + 1)}
        ep = {(loci + oh - 1, ep1j), (loci + oh - 1, ep1j + 1)}
        bp1 = (loci, locj + ow)
        bp2 = (loci + oh, locj + ow - 1)
        ln1 = connect((loci, sp1j + 2), (loci, locj + ow - 1))
        ln2 = connect((loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1))
        ln3 = connect((loci + oh - 1, ep1j + 2), (loci + oh - 1, locj + ow - 1))
    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    twos = [tuple(int(v) for v in p) for p in np.argwhere(I == 2)]
    threes = [tuple(int(v) for v in p) for p in np.argwhere(I == 3)]
    # Normalise so both marker pairs lie horizontally (transpose if vertical pair)
    trans = len(twos) > 1 and len(set(c for _, c in twos)) == 1
    G = I.T if trans else I
    H, W = G.shape
    if trans:
        twos = [(c, r) for r, c in twos]
        threes = [(c, r) for r, c in threes]

    bgc = None
    if examples:
        # background = the colour the drawn path paints over in the demonstrations
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            if ei.shape == eo.shape:
                cnt.update(ei[(eo == 3) & (ei != 3)].tolist())
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None or bgc not in set(I.flatten().tolist()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    other = [c for c in set(I.flatten().tolist()) if c not in (2, 3, bgc)]
    fgc = other[0] if other else None

    def center(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return (min(rs) + (max(rs) - min(rs) + 1) // 2,
                min(cs) + (max(cs) - min(cs) + 1) // 2)

    def val(r, c):
        if 0 <= r < H and 0 <= c < W:
            return int(G[r, c])
        return None

    def run_ends(marks, horiz):
        # bg runs along the marker row; candidate turn cells are those next to a blocker
        row = marks[0][0]
        mcs = sorted(c for _, c in marks)
        ends = []
        for start, step in ((mcs[0] - 1, -1), (mcs[-1] + 1, 1)):
            c = start
            cells = []
            while 0 <= c < W and G[row, c] == bgc:
                cells.append((row, c)); c += step
            for (r, cc) in cells:
                nb = (val(r, cc - 1), val(r, cc + 1)) if horiz else (val(r - 1, cc), val(r + 1, cc))
                if fgc is not None and fgc in nb:
                    ends.append((r, cc))
        return ends

    a_cands = run_ends(threes, True)
    b_cands = run_ends(twos, False)
    m3 = sorted(c for _, c in threes)

    def strict(a, b):
        # a: the run's terminal cell, blocked by fgc right past it along the run
        step = -1 if a[1] < m3[0] else 1
        if val(a[0], a[1] + step) != fgc:
            return False
        # b: blocked by fgc just beyond the vertical leg
        vs = 1 if b[0] > a[0] else -1
        return val(b[0] + vs, b[1]) == fgc

    pairs = []
    for a in a_cands:
        for b in b_cands:
            if a[1] != b[1]:
                continue
            lo, hi_ = sorted((a[0], b[0]))
            if all(G[r, a[1]] == bgc for r in range(lo, hi_ + 1)):
                pairs.append((a, b))
    good = [p for p in pairs if strict(*p)]
    a, b = (good or pairs)[0]
    c3 = center(threes); c2 = center(twos)

    def line(p, q):
        if p[0] == q[0]:
            s = 1 if q[1] >= p[1] else -1
            return [(p[0], c) for c in range(p[1], q[1] + s, s)]
        s = 1 if q[0] >= p[0] else -1
        return [(r, p[1]) for r in range(p[0], q[0] + s, s)]

    segA = [p for p in line(c3, a) if G[p] == bgc]             # from 3 marker to first turn
    segB = [p for p in line(a, b) if p != a]                     # the connecting leg
    segC = [p for p in line(b, c2) if p != b and G[p] == bgc]    # from second turn into 2 marker

    ops, sels = [], []
    for seg_cells in (segA, segB, segC):
        if seg_cells:
            cells = [(c, r) for r, c in seg_cells] if trans else seg_cells
            ops.append(3); sels.append(sel_of(cells))
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
