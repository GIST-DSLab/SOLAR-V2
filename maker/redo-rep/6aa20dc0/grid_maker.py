"""
ARC Task: 6aa20dc0 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors
# Task rule: one full "key" object (3 colors: c1 corner, c2 opposite corner,
# fgc body) is shown; elsewhere only the two corner markers of mirrored copies
# appear.  The output completes every partial copy with the mirrored key.
# All four color roles are sampled by the original generator -> fix them all.
# c1/c2/fgc are kept non-zero so CopyI/Paste (which treat 0 as "nothing")
# transport the whole key faithfully; bgc may be 0 (then the destination is
# cleared before the Paste).
def sample_colors(num_examples=None) -> dict:
    fgc, c1, c2 = random.sample(range(1, 10), 3)
    bgc = random.choice([c for c in range(10) if c not in (fgc, c1, c2)])
    return {"bgc": bgc, "fgc": fgc, "c1": c1, "c2": c2}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fgc: int, c1: int, c2: int) -> dict:
    hlo = min(10, max_h)
    wlo = min(10, max_w)
    gi = go = None
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        od = unifint(diff_lb, diff_ub, (2, max(2, min(4, h - 1, w - 1))))
        ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
        sinds = asindices(canvas(-1, (od, od)))
        extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
        extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
        extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
        extracells = frozenset(extracells)
        obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
        obj = obj | dmirror(obj)                      # diagonal symmetry
        if choice((True, False)):
            obj = hmirror(obj)
        gi = canvas(bgc, (h, w))
        loci = randint(0, h - od)
        locj = randint(0, w - od)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        go = tuple(e for e in gi)
        inds = asindices(gi)
        inds = inds - backdrop(outbox(plcd))
        # marker bookkeeping: keeps the puzzle unambiguous (no phantom pairing
        # of a c1 marker of one copy with the c2 marker of another one)
        markers = [([ij for v, ij in plcd if v == c1][0],
                    [ij for v, ij in plcd if v == c2][0])]
        nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
        succ = 0
        tr = 0
        maxtr = 8 * nocc + 20
        while succ < nocc and tr < maxtr:
            tr += 1
            mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
            mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
            mf = compose(mf2, mf1)
            cobj = normalize(mf(obj))                 # scale factor fixed at 1
            cands = sfilter(inds, lambda ij: ij[0] <= h - od and ij[1] <= w - od)
            if len(cands) == 0:
                break
            locc = choice(totuple(cands))
            cobjo = shift(cobj, locc)
            cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
            cobjoi = toindices(cobjo)
            if not cobjoi.issubset(inds):
                continue
            n1 = [ij for v, ij in cobji if v == c1][0]
            n2 = [ij for v, ij in cobji if v == c2][0]
            bad = False
            for (a1, a2) in markers:
                for (p, q) in ((a1, n2), (n1, a2)):
                    if abs(p[0] - q[0]) == od - 1 and abs(p[1] - q[1]) == od - 1:
                        bad = True
            if bad:
                continue
            succ += 1
            markers.append((n1, n2))
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
        if succ >= 1:
            return {'input': gi, 'output': go}
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- trajectory
def derive_operations(I, O):
    """Route: pick the key up once (CopyI), then for every marked spot
    clear what must not survive, Paste the key there, and reflect it into the
    orientation its two corner markers ask for (FlipH / FlipV)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    nonbg = sorted([c for c in cnt if c != bgc])

    # --- find the key (the only object carrying the body color) and pair up
    #     the corner markers of every partial copy -- all read from I alone.
    best = None
    if len(nonbg) == 3:
        for z in nonbg:                                   # z = body color guess
            zc = [(r, c) for r in range(h) for c in range(w) if I[r, c] == z]
            if len(zc) < 2:
                continue
            r0 = min(p[0] for p in zc); r1 = max(p[0] for p in zc)
            c0 = min(p[1] for p in zc); c1b = max(p[1] for p in zc)
            od = r1 - r0 + 1
            if od < 2 or od > 4 or (c1b - c0 + 1) != od:
                continue
            others = [c for c in nonbg if c != z]
            ins, ok = {}, True
            for col in others:
                ps = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1b + 1)
                      if I[r, c] == col]
                if len(ps) != 1:
                    ok = False
                    break
                ins[col] = ps[0]
            if not ok:
                continue
            corners = {(r0, c0), (r0, c1b), (r1, c0), (r1, c1b)}
            pa, pb = ins[others[0]], ins[others[1]]
            if pa not in corners or pb not in corners:
                continue
            if pa[0] == pb[0] or pa[1] == pb[1]:
                continue
            outA = [(r, c) for r in range(h) for c in range(w)
                    if I[r, c] == others[0] and not (r0 <= r <= r1 and c0 <= c <= c1b)]
            outB = [(r, c) for r in range(h) for c in range(w)
                    if I[r, c] == others[1] and not (r0 <= r <= r1 and c0 <= c <= c1b)]
            if len(outA) == 0 or len(outA) != len(outB):
                continue
            used, pairs, good = set(), [], True
            for p in outA:
                mi = -1
                for i, q in enumerate(outB):
                    if i in used:
                        continue
                    if abs(p[0] - q[0]) == od - 1 and abs(p[1] - q[1]) == od - 1:
                        mi = i
                        break
                if mi < 0:
                    good = False
                    break
                used.add(mi)
                pairs.append((p, outB[mi]))
            if not good:
                continue
            cand = (od * od, (r0, c0, od), pa, pairs)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is not None:
        _, (kr, kc, od), pa, pairs = best
        key = I[kr:kr + od, kc:kc + od]
        ktop = (pa[0] == kr)
        kleft = (pa[1] == kc)

        # grab the key object once (full rectangle = exactly the key's box)
        ops.append(28); sels.append([kr, kc, od - 1, od - 1])

        pairs.sort(key=lambda pq: (min(pq[0][0], pq[1][0]), min(pq[0][1], pq[1][1])))
        for (p, q) in pairs:
            r = min(p[0], q[0])
            c = min(p[1], q[1])
            if bgc == 0:
                # Paste cannot write the key's 0-colored cells, so clear only
                # the markers that the incoming copy would leave standing.
                clear = [(rr, cc) for rr in range(r, r + od) for cc in range(c, c + od)
                         if I[rr, cc] != bgc and int(key[rr - r][cc - c]) == bgc]
                if clear:
                    ops.append(0); sels.append(sel_of(clear))
            ops.append(30); sels.append([r, c, 0, 0])          # paste key here
            top = (p[0] == r)
            left = (p[1] == c)
            # reflect the pasted copy so its corner markers land on the ones
            # that were drawn there (full rectangle = the copy's own box)
            if left != kleft:
                ops.append(26); sels.append([r, c, od - 1, od - 1])
            if top != ktop:
                ops.append(27); sels.append([r, c, od - 1, od - 1])
    else:
        # safety net: nothing detected -> paint whatever genuinely differs
        diff = {}
        for r in range(min(h, O.shape[0])):
            for c in range(min(w, O.shape[1])):
                if I[r, c] != O[r, c]:
                    diff.setdefault(int(O[r, c]), []).append((r, c))
        for col in sorted(diff):
            ops.append(col); sels.append(sel_of(diff[col]))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6aa20dc0"
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
                                f"for task 6aa20dc0"
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
                    f"Failed to build a complete episode for task 6aa20dc0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6aa20dc0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
