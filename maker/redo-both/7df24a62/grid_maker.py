"""
ARC Task: 7df24a62 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc may be anything; the frame colour and the noise colour must be non-zero
    # because the framed block is duplicated with CopyI/Paste (0 == "nothing" there).
    bgc = random.choice(list(range(10)))
    rest = [c for c in range(1, 10) if c != bgc]
    noisec, sqc = random.sample(rest, 2)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, noisec: int, sqc: int) -> dict:
    hub = max(12, min(32, max_h + 2))          # grid is trimmed by 1 on each side
    wub = max(12, min(32, max_w + 2))
    while True:
        h = unifint(diff_lb, diff_ub, (12, hub))
        w = unifint(diff_lb, diff_ub, (12, wub))
        szub = max(3, min(7, h // 3, w // 3))
        oh = unifint(diff_lb, diff_ub, (3, szub))
        ow = oh                                 # square blocks: every D4 image fits the same bbox
        tmpg = canvas(sqc, (oh, ow))
        inbounds = backdrop(inbox(asindices(tmpg)))
        obj = {choice(totuple(inbounds))}
        while shape(obj) != (oh - 2, ow - 2):
            obj.add(choice(totuple(inbounds - obj)))
        pat = fill(tmpg, noisec, obj)
        targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
        sour = asobject(pat)
        gi = canvas(bgc, (h, w))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        plcddd = shift(sour, (loci, locj))
        gi = paint(gi, plcddd)
        inds = ofcolor(gi, bgc) & shift(asindices(canvas(-1, (h - 2, w - 2))), (1, 1))
        inds = inds - (toindices(plcddd) | mapply(dneighbors, toindices(plcddd)))
        if len(inds) == 0:
            continue
        namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
        noise = sample(totuple(inds), namt)
        gi = fill(gi, noisec, noise)

        pairs = []
        seenp = set()
        for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
                t = normalize(fn1(fn2(targ)))
                s = normalize(fn1(fn2(sour)))
                k = frozenset(t)
                if k in seenp:
                    continue
                seenp.add(k)
                pairs.append((t, s))
        targs = [t for t, _ in pairs]

        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (oh * ow * 4))))
        succ, tr, maxtr = 0, 0, 5 * noccs
        while succ < noccs and tr < maxtr:
            tr += 1
            t = choice(targs)
            hh, ww = shape(t)
            cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            tp = shift(t, loc)
            tpi = toindices(tp)
            if tpi.issubset(inds):
                succ += 1
                inds = inds - tpi
                gi = paint(gi, tp)

        # every stamped block must sit fully inside the trimmed canvas and must not
        # overlap another stamp or the source block (otherwise a Paste/Flip on one
        # rectangle would disturb a neighbouring one)
        boxes = set()
        for t in targs:
            hh, ww = shape(t)
            for (i, j) in occurrences(gi, t):
                boxes.add((i, j, hh, ww))
        if len(boxes) == 0:
            continue
        rects = []
        ok = True
        for (i, j, hh, ww) in boxes:
            if not (1 <= i and i + hh <= h - 1 and 1 <= j and j + ww <= w - 1):
                ok = False
                break
            rects.append({(a, b) for a in range(i, i + hh) for b in range(j, j + ww)})
        if not ok:
            continue
        rects.append({(a, b) for a in range(loci, loci + oh) for b in range(locj, locj + ow)})
        clash = False
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                if rects[a] & rects[b]:
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue

        go = replace(gi, sqc, bgc)
        go = paint(go, plcddd)
        res = set()
        for t, s in pairs:
            res |= mapply(lbind(shift, s), occurrences(go, t))
        go = paint(go, res)
        gi = trim(gi)
        go = trim(go)
        return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Everything below is measured from I only: which block is the stamp, where its
    bare noise pattern re-occurs, and in which reflection/rotation it re-occurs there."""
    A = np.asarray(I, dtype=int)
    hi, wi = A.shape
    ops, sels = [], []
    submit_sel = [0, 0, hi - 1, wi - 1]

    cnt = Counter(A.flatten().tolist())
    colors = sorted(cnt)
    if len(colors) < 3:
        return [34], [submit_sel]

    def span(c):
        rs, cs = np.where(A == c)
        return max(int(rs.max() - rs.min()) + 1, int(cs.max() - cs.min()) + 1)

    # frame colour = the colour confined to the smallest region (as the rule's own
    # marker); of the two remaining colours the rarer is the noise, the other the bg
    sqc = min(colors, key=lambda c: (span(c), cnt[c], c))
    others = sorted([c for c in colors if c != sqc], key=lambda c: (cnt[c], c))
    noisec, bgc = others[0], others[-1]

    rs, cs = np.where(A == sqc)
    r0, c0 = int(rs.min()), int(cs.min())
    oh = int(rs.max()) - r0 + 1
    ow = int(cs.max()) - c0 + 1
    S = np.ascontiguousarray(A[r0:r0 + oh, c0:c0 + ow])

    # the eight images of the stamp under the dihedral group
    variants, seen = [], set()
    for k in range(4):
        R = np.rot90(S, k)
        for G in (R, np.fliplr(R)):
            G = np.ascontiguousarray(G)
            key = (G.shape, G.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(G)

    # search the grid (padded with bg, as the rule sees it) for the bare pattern:
    # each variant with its frame cells reading background
    P = np.full((hi + 2, wi + 2), bgc, dtype=int)
    P[1:hi + 1, 1:wi + 1] = A
    occ = {}
    for G in variants:
        gh, gw = G.shape
        T = np.where(G == sqc, bgc, G)
        for i in range(hi + 3 - gh):
            for j in range(wi + 3 - gw):
                if np.array_equal(P[i:i + gh, j:j + gw], T):
                    r, c = i - 1, j - 1
                    if 0 <= r and r + gh <= hi and 0 <= c and c + gw <= wi:
                        occ.setdefault((r, c), G)
    if not occ:
        return [34], [submit_sel]

    def chain_to(G):
        """Shortest sequence of ARCLE flips/rotations carrying S onto G,
        every step visibly changing the block."""
        if G.shape == S.shape and np.array_equal(S, G):
            return []
        moves = [(26, np.fliplr), (27, np.flipud)]
        if S.shape[0] == S.shape[1]:
            moves += [(24, lambda a: np.rot90(a, 1)), (25, lambda a: np.rot90(a, 3))]
        start = (S.shape, S.tobytes())
        visited = {start}
        q = deque([(S, [])])
        while q:
            cur, path = q.popleft()
            if len(path) >= 3:
                continue
            for op, fn in moves:
                nxt = np.ascontiguousarray(fn(cur))
                if nxt.shape == cur.shape and np.array_equal(nxt, cur):
                    continue                      # would be an invisible no-op
                key = (nxt.shape, nxt.tobytes())
                if key in visited:
                    continue
                visited.add(key)
                if nxt.shape == G.shape and np.array_equal(nxt, G):
                    return path + [op]
                q.append((nxt, path + [op]))
        return None

    targets = []
    for (r, c), G in sorted(occ.items()):
        targets.append(((r, c), G, chain_to(G)))

    if any(ch is not None for _, _, ch in targets):
        src_cells = [(a, b) for a in range(r0, r0 + oh) for b in range(c0, c0 + ow)]
        ops.append(28)                       # CopyI: pick up the framed stamp from the input
        sels.append(sel_of(src_cells))       # bbox-shaped: the whole block rectangle

    for (r, c), G, ch in targets:
        gh, gw = G.shape
        rect = [(r + a, c + b) for a in range(gh) for b in range(gw)]
        if ch is not None:
            ops.append(30)                    # Paste the stamp onto the bare pattern
            sels.append(sel_of([(r, c)]))
            for op in ch:                     # reflect / rotate it into the pattern's pose
                ops.append(op)
                sels.append(sel_of(rect))     # bbox-shaped: the whole stamped rectangle
        else:
            # transposed pose of a non-square stamp: no ARCLE rotation can express it,
            # so draw the frame that this pose puts around the pattern already in I
            cells = [(r + a, c + b) for a in range(gh) for b in range(gw) if G[a, b] == sqc]
            ops.append(int(sqc))
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append(submit_sel)
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
                        f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
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
                                f"for task 7df24a62"
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
                    f"Failed to build a complete episode for task 7df24a62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7df24a62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
