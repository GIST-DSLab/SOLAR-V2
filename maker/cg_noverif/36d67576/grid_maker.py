"""
ARC Task: 36d67576 (RE-ARC) — LLM-generated grid_maker
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
    # Rule depends only on structure (template vs partial copies); fix bgc only.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    while True:
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        mainc, markerc = sample(difference(cols, (bgc,)), 2)
        remcols = difference(cols, (bgc, mainc, markerc))
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(remcols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            oh2, ow2 = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh2 and ij[1] <= w - ow2)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry -- they agree):
      * the one complete figure (largest 8-connected non-bg component) is the
        template: main colour + marker + completion colours;
      * every other figure is that template, in one of the 8 square-symmetry
        orientations, with only its main+marker cells drawn;
      * stamp the template, in the matching orientation, over each partial.
    The golf version rotates the grid instead of the template and stamps all
    colours of a copy at once; ARCLE has no multi-colour stamp that is safe here
    (0 can be a figure colour, Paste is transparent to 0, and Rotate breaks on
    non-square shapes), so each partial is completed as one unit: one Color op
    per completion colour, finishing that figure before the next.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    submit = [0, 0, h - 1, w - 1]

    # background: the colour that dominates every demonstration input
    bgc = None
    if examples:
        cnt = Counter()
        for ex in examples:
            xi = np.asarray(ex[0], dtype=int)
            c0 = Counter(xi.flatten().tolist()).most_common(1)[0][0]
            cnt[c0] += 1
        cand = cnt.most_common(1)[0][0]
        if cand in set(I.flatten().tolist()):
            bgc = cand
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] != bgc:
                            seen[na, nb] = True
                            stack.append((na, nb))
            comps.append(cells)

    if len(comps) < 2:
        ops.append(34); sels.append(submit)
        return ops, sels

    cid = {}
    for k, cc in enumerate(comps):
        for p in cc:
            cid[p] = k
    tidx = max(range(len(comps)), key=lambda k: len(comps[k]))
    sub_palette = set()
    for k, cc in enumerate(comps):
        if k != tidx:
            for (r, c) in cc:
                sub_palette.add(int(I[r, c]))
    template = frozenset((int(I[r, c]), (r, c)) for (r, c) in comps[tidx])
    if not any(v in sub_palette for v, _ in template):
        ops.append(34); sels.append(submit)
        return ops, sels

    def norm(o):
        mi = min(p[0] for _, p in o)
        mj = min(p[1] for _, p in o)
        return frozenset((v, (p[0] - mi, p[1] - mj)) for v, p in o)

    base = norm(template)
    variants = []
    cur = base
    for _ in range(4):
        cur = norm(frozenset((v, (j, -i)) for v, (i, j) in cur))  # rotate
        for t in (cur, norm(frozenset((v, (i, -j)) for v, (i, j) in cur))):
            if t not in variants:
                variants.append(t)

    placements = []
    for var in variants:
        sub = [(v, p) for v, p in var if v in sub_palette]
        if not sub or len(sub) == len(var):
            continue
        sr = min(p[0] for _, p in sub)
        sc = min(p[1] for _, p in sub)
        a_sub = [(v, (p[0] - sr, p[1] - sc)) for v, p in sub]
        a_full = [(v, (p[0] - sr, p[1] - sc)) for v, p in var]
        maxa = max(p[0] for _, p in a_sub)
        maxb = max(p[1] for _, p in a_sub)
        for i in range(0, h - maxa):
            for j in range(0, w - maxb):
                if not all(I[i + a, j + b] == v for v, (a, b) in a_sub):
                    continue
                # an occurrence must be made of whole figures, and the cells
                # it completes must still be empty background
                M = {(i + a, j + b) for _, (a, b) in a_sub}
                if not all(set(comps[cid[p]]) <= M for p in M):
                    continue
                tgt = [(i + a, j + b) for v, (a, b) in a_full if v not in sub_palette]
                if not all(0 <= r < h and 0 <= c < w and I[r, c] == bgc for r, c in tgt):
                    continue
                comp_ids = frozenset(cid[p] for p in M)
                tgtset = frozenset(tgt)
                placements.append(((i, j), a_full, comp_ids, tgtset))

    # every partial figure belongs to exactly one occurrence: pick a set of
    # occurrences that explains each non-template figure exactly once
    partial_ids = [k for k in range(len(comps)) if k != tidx]
    by_comp = {k: [] for k in partial_ids}
    uniq = {}
    for pl in placements:
        key = (pl[2], pl[3])
        if key in uniq:
            continue
        uniq[key] = pl
        for k in pl[2]:
            if k in by_comp:
                by_comp[k].append(pl)

    def solve(covered, painted, chosen, budget):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        left = [k for k in partial_ids if k not in covered]
        if not left:
            return list(chosen)
        k = min(left, key=lambda q: len(by_comp[q]))
        for pl in by_comp[k]:
            if pl[2] & covered or pl[3] & painted:
                continue
            chosen.append(pl)
            res = solve(covered | pl[2], painted | pl[3], chosen, budget)
            chosen.pop()
            if res is not None:
                return res
        return None

    chosen = solve(frozenset(), frozenset(), [], [20000])
    if chosen is None:
        chosen = list(uniq.values())
    placements = sorted(((p[0], p[1]) for p in chosen), key=lambda t: t[0])

    G = I.copy()
    for (i, j), a_full in placements:
        bycol = {}
        for v, (a, b) in a_full:
            rr, cc = i + a, j + b
            if not (0 <= rr < h and 0 <= cc < w) or G[rr, cc] == v:
                continue
            bycol.setdefault(v, []).append((rr, cc))
        for v in sorted(bycol):
            cells = sorted(bycol[v])
            ops.append(int(v)); sels.append(sel_of(cells))
            for (rr, cc) in cells:
                G[rr, cc] = v

    ops.append(34); sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task 36d67576"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 36d67576"
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
                                f"for task 36d67576"
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
                    f"Failed to build a complete episode for task 36d67576 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"36d67576-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
