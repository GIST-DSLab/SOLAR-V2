"""
ARC Task: 484b58aa (RE-ARC) — LLM-generated grid_maker
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
    import random
    ROTS = ["identity", "rot90", "rot180", "rot270"]
    noisec = random.choice(list(range(10)))
    remcols = [c for c in range(10) if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, ccols, rot=None) -> dict:
    import random
    if rot is None:
        rot = random.choice(["identity", "rot90", "rot180", "rot270"])
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    lim_h, lim_w = max_h, max_w
    if rot in ("rot90", "rot270"):
        lim_h, lim_w = max_w, max_h
    lim_h = max(10, min(30, lim_h))
    lim_w = max(10, min(30, lim_w))
    ccols = list(ccols)

    while True:
        h = unifint(diff_lb, diff_ub, (10, lim_h))
        w = unifint(diff_lb, diff_ub, (10, lim_w))
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
        go = canvas(-1, (h, w))
        locs = set()
        ofs = randint(1, hp - 1)
        for a in range(2 * (h // hp + 1)):
            for b in range(w // wp + 1):
                loci = hp * a - ofs * b
                locj = wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
        for b in range(-1, w // wp + 2):
            for a in range(-2, (h + ofs * abs(b)) // hp + 3):
                loci = hp * a - ofs * b
                locj = wp * b
                if -hp < loci < h and -wp < locj < w:
                    locs.add((loci, locj))
                    go = paint(go, shift(pobj, (loci, locj)))
        if any(-1 in row for row in go):
            continue
        inb = [(li, lj) for (li, lj) in locs
               if 0 <= li and 0 <= lj and li + hp <= h and lj + wp <= w]
        if not inb:
            continue
        numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
        gi = tuple(e for e in go)
        succ = 0
        tr = 0
        maxtr = max(25, 5 * numpatches)
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = randint(2, 6)
            pw = randint(2, 6)
            loci = randint(0, h - ph)
            locj = randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            intact = any(all(gi2[li + di][lj + dj] != noisec
                             for di in range(hp) for dj in range(wp))
                         for (li, lj) in inb)
            if intact:
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                   len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        if succ >= 1:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (line-for-line): the wallpaper repeats whole lines. A row that is
    # hit by noise has a twin row a whole vertical period away; copy that twin
    # row and stamp it over the damaged one. Same for columns. Rows and columns
    # alternate until no line can be repaired further; a line repaired in one
    # pass can itself serve as the twin in the next (CopyO).
    import numpy as np
    from collections import Counter, deque
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- noise colour: present in every demo input, absent from every output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            cand &= set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:   # verifier heuristic on I alone: fewest components, then rarest
        def ncomp(col):
            m = (I == col); seen = np.zeros_like(m); n = 0
            for r in range(H):
                for c in range(W):
                    if m[r, c] and not seen[r, c]:
                        n += 1; q = [(r, c)]; seen[r, c] = True
                        while q:
                            x, y = q.pop()
                            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                                if 0 <= nx < H and 0 <= ny < W and m[nx, ny] and not seen[nx, ny]:
                                    seen[nx, ny] = True; q.append((nx, ny))
            return n
        cols = sorted(set(I.flatten().tolist()))
        nc = {c: ncomp(c) for c in cols}
        mn = min(nc.values())
        cnt = Counter(I.flatten().tolist())
        noisec = min((c for c in cols if nc[c] == mn), key=lambda c: cnt[c])

    if not (I == noisec).any():
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    clean_I = I != noisec

    def line_period(G, cl):
        # smallest d such that line i and line i+d agree wherever both are intact
        h, w = G.shape
        for d in range(1, h):
            A, B = G[:h - d], G[d:]
            m = cl[:h - d] & cl[d:]
            if int(m.sum()) < max(12, (h - d) * w // 8):
                continue
            if (A[m] == B[m]).all():
                return d
        return None

    p_row = line_period(I, clean_I)
    p_col = line_period(I.T, clean_I.T)

    def get_line(G, axis, i):
        return G[i, :] if axis == 0 else G[:, i]

    def usable(sv, tl):
        # a twin line may carry noise only where the target is damaged too
        # (noise 0 is transparent to Paste, so then any twin is fine)
        return noisec == 0 or not ((sv == noisec) & (tl != noisec)).any()

    def gain(sv, tl):
        return int(((tl == noisec) & (sv != noisec)).sum())

    def run_plan(plan, emit):
        # plan: list of (axis, target line, source line). Returns (grid, clip, ok)
        cur = I.copy()
        clip = None
        for axis, t, src in plan:
            sv = get_line(cur, axis, src).copy()
            tl = get_line(cur, axis, t)
            bad = tl == noisec
            if not usable(sv, tl):
                return None, None, False          # twin would spoil intact cells
            zc = np.nonzero(bad & (sv == 0) & (sv != noisec))[0]
            if len(zc):
                # Paste can't write 0: blank the noise cells whose true colour is 0
                cells = [(t, int(i)) for i in zc] if axis == 0 else [(int(i), t) for i in zc]
                if emit:
                    ops.append(0); sels.append(sel_of(cells))
                for rc in cells:
                    cur[rc] = 0
            tl = get_line(cur, axis, t)
            if (bad & (sv != 0) & (sv != noisec)).any():
                want = tl.copy(); want[sv != 0] = sv[sv != 0]
                stale_ok = False
                if clip is not None and clip[0] == axis and clip[1].shape == sv.shape:
                    st = tl.copy(); cv = clip[1]; st[cv != 0] = cv[cv != 0]
                    stale_ok = np.array_equal(st, want)
                if not stale_ok:
                    from_input = bool(np.array_equal(get_line(I, axis, src), sv))
                    if emit:
                        ops.append(28 if from_input else 29)   # copy the twin line
                        sels.append([src, 0, 0, W - 1] if axis == 0 else [0, src, H - 1, 0])
                    clip = (axis, sv.copy())
                if emit:
                    ops.append(30)                              # stamp it over the damaged line
                    sels.append([t, 0, 0, 0] if axis == 0 else [0, t, 0, 0])
                if axis == 0:
                    cur[t, :] = want
                else:
                    cur[:, t] = want
        return cur, clip, True

    # greedy plan: rows, then columns, alternating while lines keep getting repaired
    plan = []
    cur = I.copy()
    while (cur == noisec).any():
        progress = False
        for axis, period in ((0, p_row), (1, p_col)):
            if period is None:
                continue
            n = H if axis == 0 else W
            for cls in range(period):
                members = list(range(cls, n, period))
                for t in members:
                    tl = get_line(cur, axis, t)
                    if not (tl == noisec).any():
                        continue
                    best = None
                    for src in members:
                        if src == t:
                            continue
                        sv = get_line(cur, axis, src)
                        if not usable(sv, tl):
                            continue
                        g = gain(sv, tl)
                        if g == 0:
                            continue
                        key = (-g, int((sv == noisec).sum()), abs(src - t))
                        if best is None or key < best[0]:
                            best = (key, src)
                    if best is None:
                        continue
                    src = best[1]
                    plan.append((axis, t, src))
                    sv = get_line(cur, axis, src)
                    fix = (tl == noisec) & (sv != noisec)
                    tl[fix] = sv[fix]
                    progress = True
        if not progress:
            break

    # drop any line repair whose work another repair already does
    left_full = int((run_plan(plan, False)[0] == noisec).sum())
    k = len(plan) - 1
    while k >= 0:
        trial = plan[:k] + plan[k + 1:]
        g, _, ok = run_plan(trial, False)
        if ok and int((g == noisec).sum()) <= left_full:
            plan = trial
        k -= 1

    cur, clip, _ = run_plan(plan, True)

    # --- fallback: cells no whole line can reach; copy a lattice-equivalent
    # intact row segment (2-D translation of the wallpaper) over what is left.
    if (cur == noisec).any():
        lattice = []
        for dr in range(-H + 1, H):
            for dc in range(-W + 1, W):
                if dr == 0 and dc == 0:
                    continue
                r0, r1 = max(0, -dr), min(H, H - dr)
                c0, c1 = max(0, -dc), min(W, W - dc)
                if r1 <= r0 or c1 <= c0:
                    continue
                A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                m = clean_I[r0:r1, c0:c1] & clean_I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                    continue
                if (A[m] == B[m]).all():
                    lattice.append((abs(dr) + abs(dc), dr, dc))
        lattice.sort()
        for (r, c) in sorted(zip(*np.nonzero(cur == noisec))):
            r, c = int(r), int(c)
            if cur[r, c] != noisec:
                continue
            best = None
            okc = cur != noisec
            for _, dr, dc in lattice:
                sr, sc = r + dr, c + dc
                if not (0 <= sr < H and 0 <= sc < W and okc[sr, sc]):
                    continue
                n = 0
                while (c + n < W and cur[r, c + n] == noisec and 0 <= sc + n < W
                       and okc[sr, sc + n]):
                    n += 1
                if best is None or n > best[0]:
                    best = (n, sr, sc)
            if best is None:
                continue
            n, sr, sc = best
            src_vals = cur[sr, sc:sc + n].copy()
            zc = [(r, c + k) for k in range(n) if src_vals[k] == 0]
            if zc:
                ops.append(0); sels.append(sel_of(zc))
                for rc in zc:
                    cur[rc] = 0
            if (src_vals != 0).any():
                from_input = bool((I[sr, sc:sc + n] == src_vals).all())
                fresh = src_vals.copy()
                tgt = cur[r, c:c + n].copy()
                want = tgt.copy(); want[fresh != 0] = fresh[fresh != 0]
                stale_ok = False
                if clip is not None and clip[0] == 0 and clip[1].shape == fresh.shape:
                    cv = clip[1]
                    st = tgt.copy(); st[cv != 0] = cv[cv != 0]
                    stale_ok = np.array_equal(st, want)
                if not stale_ok:
                    ops.append(28 if from_input else 29); sels.append([sr, sc, 0, n - 1])
                    clip = (0, fresh)
                ops.append(30); sels.append([r, c, 0, 0])
                cur[r, c:c + n] = want

    # --- every op must matter: replay the trajectory on I (ops 0/28/29/30 only)
    # and drop any op whose removal leaves the final grid unchanged.
    def cells_of(sel):
        if isinstance(sel, dict):
            return [tuple(x) for x in sel["cells"]]
        r, c, h, w = sel
        return [(r + i, c + j) for i in range(h + 1) for j in range(w + 1)]

    def replay(oplist, sellist):
        g = I.copy(); clipb = None
        for o, sl in zip(oplist, sellist):
            if o <= 9:
                for rc in cells_of(sl):
                    g[rc] = o
            elif o in (28, 29):
                r, c, h, w = sl
                clipb = (I if o == 28 else g)[r:r + h + 1, c:c + w + 1].copy()
            elif o == 30:
                if clipb is None:
                    continue
                r, c = sl[0], sl[1]
                blk = g[r:r + clipb.shape[0], c:c + clipb.shape[1]]
                cb = clipb[:blk.shape[0], :blk.shape[1]]
                blk[cb != 0] = cb[cb != 0]
        return g

    target = replay(ops, sels)
    changed = True
    while changed:
        changed = False
        for k in range(len(ops) - 1, -1, -1):
            o2 = ops[:k] + ops[k + 1:]; s2 = sels[:k] + sels[k + 1:]
            if np.array_equal(replay(o2, s2), target):
                ops, sels = o2, s2
                changed = True
                break

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
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
                                f"for task 484b58aa"
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
                    f"Failed to build a complete episode for task 484b58aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"484b58aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
