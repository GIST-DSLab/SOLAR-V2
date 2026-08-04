"""
ARC Task: 6a1e5592 (RE-ARC) — LLM-generated grid_maker
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

ROTS = ['identity', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    barc, bgc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append({"rot": random.choice(ex)})
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None) -> dict:
    if rot is None:
        rot = choice(ROTS)
    rotf = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}[rot]
    swaps = rot in ('rot90', 'rot270')
    hub = max_w if swaps else max_h
    wub = max_h if swaps else max_w
    h = unifint(diff_lb, diff_ub, (9, max(9, hub)))
    w = unifint(diff_lb, diff_ub, (5, max(5, wub)))
    barh = randint(3, h // 3)
    maxobjh = h - barh - 1
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
    c1 = canvas(barc, (barh, w))
    c2 = canvas(bgc, (h - barh, w))
    gi = vconcat(c1, c2)
    go = tuple(e for e in gi)
    tr = 0
    succ = 0
    maxtr = 10 * nobjs
    placopts = interval(1, w - 1, 1)
    iinds = ofcolor(gi, bgc)
    oinds = asindices(go)
    barinds = ofcolor(gi, barc)
    forbmarkers = set()
    while tr < maxtr and succ < nobjs:
        tr += 1
        oh = randint(1, maxobjh)
        ow = randint(1, min(4, w // 2))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = randint(1, oh * ow)
        sp = choice(totuple(connect((0, 0), (0, ow - 1))))
        obj = {sp}
        for k in range(ncells - 1):
            cands = totuple((bounds - obj) & mapply(dneighbors, obj))
            if len(cands) == 0:
                break
            obj.add(choice(cands))
        obj = normalize(obj)
        oh, ow = shape(obj)
        markerh = randint(1, min(oh, barh - 1))
        markpart = sfilter(obj, lambda ij: ij[0] < markerh)
        markpartn = normalize(markpart)
        isinvalid = False
        for k in range(1, markerh + 1):
            if normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers:
                isinvalid = True
        if isinvalid:
            continue
        for k in range(1, markerh + 1):
            forbmarkers.add(normalize(sfilter(markpartn, lambda ij: ij[0] < k)))
        placoptcands = sfilter(placopts, lambda jj: set(interval(jj, jj + ow + 1, 1)).issubset(set(placopts)))
        if len(placoptcands) == 0:
            continue
        jloc = choice(placoptcands)
        iloc = barh - markerh
        oplcd = shift(obj, (iloc, jloc))
        if oplcd.issubset(oinds):
            icands = sfilter(iinds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(icands) == 0:
                continue
            loc = choice(totuple(icands))
            iplcd = shift(obj, loc)
            if iplcd.issubset(iinds):
                succ += 1
                iinds = (iinds - iplcd) - mapply(neighbors, iplcd)
                oinds = (oinds - oplcd)
                gi = fill(gi, objc, iplcd)
                gi = fill(gi, bgc, oplcd & barinds)
                go = fill(go, 1, oplcd)
                jm = apply(last, ofcolor(go, 1))
                placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
        if len(placopts) == 0:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import deque

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    n, m = I.shape
    full = [0, 0, n - 1, m - 1]

    pal_i = set(int(v) for v in I.flatten().tolist())
    pal_o = set(int(v) for v in O.flatten().tolist())
    extra = pal_i - pal_o
    if not extra:
        return [34], [full]
    objc = sorted(extra)[0]

    # --- locate the bar: the only band whose outer edge line is one solid colour ---
    edges = []
    if len(set(O[0, :].tolist())) == 1:
        c = int(O[0, 0])
        edges.append((sum(1 for r in range(n) if (O[r, :] == c).any()), 0, c))
    if len(set(O[n - 1, :].tolist())) == 1:
        c = int(O[n - 1, 0])
        edges.append((sum(1 for r in range(n) if (O[r, :] == c).any()), 2, c))
    if len(set(O[:, 0].tolist())) == 1:
        c = int(O[0, 0])
        edges.append((sum(1 for j in range(m) if (O[:, j] == c).any()), 3, c))
    if len(set(O[:, m - 1].tolist())) == 1:
        c = int(O[0, m - 1])
        edges.append((sum(1 for j in range(m) if (O[:, j] == c).any()), 1, c))
    edges.sort()
    barh, k, barc = edges[0]            # bar band is always the thinner band
    rest = pal_i - {objc, barc}
    bgc = sorted(rest)[0]

    In = np.rot90(I, k)                 # normalised frame: bar on top
    hn, wn = In.shape

    def inv(i, j):                      # normalised -> original coords
        if k == 0:
            return (i, j)
        if k == 1:
            return (j, m - 1 - i)
        if k == 2:
            return (n - 1 - i, m - 1 - j)
        return (n - 1 - j, i)

    # --- objects (loose pieces in the body) ---
    seen = np.zeros((hn, wn), dtype=bool)
    comps = []
    for r in range(hn):
        for c in range(wn):
            if In[r, c] == objc and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < hn and 0 <= y < wn and not seen[x, y] and In[x, y] == objc:
                            seen[x, y] = True
                            q.append((x, y))
                comps.append(cells)
    if not comps:
        return [34], [full]

    # --- notches: holes bitten out of the bar ---
    N = frozenset((r, c) for r in range(barh) for c in range(wn) if In[r, c] == bgc)

    # --- match each piece's leading rows to a notch -> its slot ---
    cand_lists = []
    for cells in comps:
        r0 = min(a for a, _ in cells)
        c0 = min(b for _, b in cells)
        norm = frozenset((a - r0, b - c0) for a, b in cells)
        oh = max(a for a, _ in norm) + 1
        ow = max(b for _, b in norm) + 1
        cl = []
        for jloc in range(0, wn - ow + 1):
            nw = frozenset(p for p in N if jloc <= p[1] <= jloc + ow - 1)
            if not nw:
                continue
            mk = barh - min(p[0] for p in nw)
            if mk < 1 or mk > barh - 1:
                continue
            iloc = barh - mk
            if iloc + oh > hn:
                continue
            head = frozenset((iloc + a, jloc + b) for a, b in norm if iloc + a < barh)
            if head == nw:
                dest = frozenset((iloc + a, jloc + b) for a, b in norm)
                cl.append((dest, nw))
        cand_lists.append(cl)

    nobj = len(comps)
    assign = [None] * nobj
    order = sorted(range(nobj), key=lambda i: len(cand_lists[i]))
    used = set()

    def bt(idx):
        if idx == len(order):
            return len(used) == len(N)
        i = order[idx]
        for dest, nw in cand_lists[i]:
            if nw & used:
                continue
            if any(assign[j] is not None and (assign[j][0] & dest) for j in range(nobj)):
                continue
            assign[i] = (dest, nw)
            used.update(nw)
            if bt(idx + 1):
                return True
            used.difference_update(nw)
            assign[i] = None
        return False

    if not bt(0):
        for i in range(nobj):
            if assign[i] is None and cand_lists[i]:
                assign[i] = cand_lists[i][0]

    live = [i for i in range(nobj) if assign[i] is not None]
    srcs = {i: frozenset(comps[i]) for i in live}
    dests = {i: assign[i][0] for i in live}

    # --- order pieces: a piece must vanish before another piece is drawn over it ---
    deps = {i: set(j for j in live if j != i and (dests[i] & srcs[j])) for i in live}
    seq, ready = [], [i for i in live if not deps[i]]
    done = set()
    while ready:
        i = ready.pop(0)
        seq.append(i)
        done.add(i)
        for j in live:
            if j not in done and j not in ready and deps[j] <= done:
                ready.append(j)
    cyclic = len(seq) != len(live)
    if cyclic:
        seq = live

    def rects(cells):
        rem = set(cells)
        out = []
        while rem:
            r0, c0 = min(rem)
            wmax = 0
            while (r0, c0 + wmax) in rem:
                wmax += 1
            best = (1, 1)
            for ww in range(1, wmax + 1):
                hh = 0
                while all((r0 + hh, c0 + j) in rem for j in range(ww)):
                    hh += 1
                if ww * hh > best[0] * best[1]:
                    best = (ww, hh)
            ww, hh = best
            for a in range(hh):
                for b in range(ww):
                    rem.discard((r0 + a, c0 + b))
            out.append((r0, c0, hh, ww))
        return out

    ops, sels = [], []

    def erase(i):
        sr, sc = min(srcs[i])
        a, b = inv(sr, sc)
        ops.append(10 + bgc)
        sels.append([a, b, 0, 0])

    def paint(i):
        for (r0, c0, hh, ww) in rects(dests[i]):
            a = inv(r0, c0)
            b = inv(r0 + hh - 1, c0 + ww - 1)
            rmin, rmax = min(a[0], b[0]), max(a[0], b[0])
            cmin, cmax = min(a[1], b[1]), max(a[1], b[1])
            ops.append(1)
            sels.append([rmin, cmin, rmax - rmin, cmax - cmin])

    if cyclic:
        for i in seq:
            erase(i)
        for i in seq:
            paint(i)
    else:
        for i in seq:
            erase(i)
            paint(i)

    ops.append(34)
    sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
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
                                f"for task 6a1e5592"
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
                    f"Failed to build a complete episode for task 6a1e5592 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6a1e5592-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
