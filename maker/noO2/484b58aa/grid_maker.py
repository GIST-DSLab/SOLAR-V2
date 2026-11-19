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

    # rot90 / rot270 swap the final dimensions -> swap the caps when sampling
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
        # the loop above can leave a corner unreached for extreme hp/ofs/wp; the
        # wallpaper must cover the whole canvas, so redraw over the full lattice
        for b in range(-1, w // wp + 2):
            for a in range(-2, (h + ofs * abs(b)) // hp + 3):
                loci = hp * a - ofs * b
                locj = wp * b
                if -hp < loci < h and -wp < locj < w:
                    locs.add((loci, locj))
                    go = paint(go, shift(pobj, (loci, locj)))
        if any(-1 in row for row in go):
            continue

        # every cell belongs to one lattice class; a fully intact tile placement
        # keeps one clean representative of every class, i.e. keeps the whole
        # wallpaper reconstructible from what is left visible
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
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The grid is a wallpaper: one tile repeated along a lattice of translations.
    Rectangular patches of a single intruder colour have been painted over it.
    The rule carried out here: measure the lattice on the cells the intruder did
    not touch, then for every damaged patch copy the same block of wallpaper from
    one lattice step away and stamp it back over the hole.  Every position, size
    and colour below is measured from I; O is never read."""
    import numpy as np
    from collections import deque
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    cur = I.copy()

    def overlap(dr, dc, clean):
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r1 <= r0 or c1 <= c0:
            return None
        a = I[r0:r1, c0:c1]
        b = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        return a, b, m

    def scan(clean):
        """every translation the visible wallpaper agrees with itself under,
        together with how many undamaged cell pairs back that agreement up"""
        out = []
        for dr in range(-(h - 1), h):
            for dc in range(0, w):
                if (dr, dc) == (0, 0):
                    continue
                o = overlap(dr, dc, clean)
                if o is None:
                    continue
                a, b, m = o
                n = int(m.sum())
                if n and bool(np.all(a[m] == b[m])):
                    out.append((dr, dc, n))
        return out

    # ---- 1. which colour is the intruder ------------------------------------
    # The wallpaper repeats; the patches painted over it do not.  So the intruder
    # is the colour whose cells, once set aside, leave a grid that repeats -- the
    # only colour whose removal lets translations of the grid agree with itself.
    # (Blob shape alone is not enough: a pattern colour can also occur in few,
    # large pieces.)
    noise, found, best = None, [], None
    for col in sorted(set(I.flatten().tolist())):
        S = scan(I != col)
        strong = [s for s in S if s[2] >= max(8, (h * w) // 8)]
        key = (len(strong), -int((I == col).sum()))
        if best is None or key > best:
            best, noise, found = key, col, S
    dmg = (I == noise)
    clean = ~dmg

    def components(mask):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)]); seen[r, c] = True; comp = []
                    while q:
                        rr, cc = q.popleft(); comp.append((rr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True; q.append((nr, nc))
                    out.append(sorted(comp))
        return out

    # ---- 2. the wallpaper's translation lattice ------------------------------
    # Sums and differences of symmetries are symmetries too, so reduce the
    # measured steps to a lattice basis: that also yields the steps too large to
    # be measured directly on the little of them that overlaps the grid.  Trust
    # the best-evidenced steps first, and take in weaker ones only while some
    # damaged cell still has no undamaged twin to be copied from.
    from math import gcd

    def lattice(shifts):
        base = None
        for v in shifts:
            if v[1] == 0:
                continue
            if base is None:
                base = v
                continue
            x, y = base, v
            while y[1] != 0:
                q = x[1] // y[1]
                x, y = y, (x[0] - q * y[0], x[1] - q * y[1])
            base = x
        if base is not None and base[1] < 0:
            base = (-base[0], -base[1])
        vert = 0
        for v in shifts:
            if base is None:
                vert = gcd(vert, abs(v[0]))
            else:
                k = v[1] // base[1]
                vert = gcd(vert, abs(v[0] - k * base[0]))
        out = set()
        ks = range(-w, w + 1) if base is not None else [0]
        ms = range(-h, h + 1) if vert else [0]
        for k in ks:
            for m in ms:
                dr = (k * base[0] if base is not None else 0) + m * vert
                dc = (k * base[1] if base is not None else 0)
                if (dr, dc) == (0, 0) or abs(dr) >= h or abs(dc) >= w:
                    continue
                out.add((dr, dc)); out.add((-dr, -dc))
        out = [v for v in out if agrees(*v)]
        out.sort(key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), abs(d[1])))
        return out

    def agrees(dr, dc):
        o = overlap(dr, dc, clean)
        if o is None:
            return False
        a, b, m = o
        return bool(np.all(a[m] == b[m]))

    def covers(offs):
        for r in range(h):
            for c in range(w):
                if dmg[r, c] and not any(
                        0 <= r - dr < h and 0 <= c - dc < w and not dmg[r - dr, c - dc]
                        for dr, dc in offs):
                    return False
        return True

    strong_t = max(8, (h * w) // 8)
    thresholds = [strong_t] + sorted({s[2] for s in found if s[2] < strong_t}, reverse=True)
    OFFS = []
    for t in thresholds:
        OFFS = lattice([(dr, dc) for (dr, dc, n) in found if n >= t])
        if OFFS and covers(OFFS):
            break

    def stamp(r0, c0, rh, rw, dr, dc):
        """copy the block one lattice step away and paste it over this hole"""
        sr, sc = r0 - dr, c0 - dc
        src = I[sr:sr + rh, sc:sc + rw]
        if any(src[a, b] != 0 and cur[r0 + a, c0 + b] != src[a, b]
               for a in range(rh) for b in range(rw)):
            ops.append(28); sels.append([sr, sc, rh - 1, rw - 1])   # CopyI: full rectangle
            ops.append(30); sels.append([r0, c0, 0, 0])             # Paste at the hole
            for a in range(rh):
                for b in range(rw):
                    if src[a, b] != 0:
                        cur[r0 + a, c0 + b] = src[a, b]
        # Paste carries no 0s, so the black cells of the copied block are
        # laid in separately
        zc = [(r0 + a, c0 + b) for a in range(rh) for b in range(rw)
              if src[a, b] == 0 and cur[r0 + a, c0 + b] != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))
            for r, c in zc:
                cur[r, c] = 0

    # ---- 3. hole by hole, stamp the wallpaper back in ------------------------
    # Damage covers the grid in blocks, so repair it in blocks: starting at the
    # first cell of a hole, take the largest block whose twin one lattice step
    # away is undamaged, and stamp that twin over it.
    ii = np.zeros((h + 1, w + 1), dtype=int)
    ii[1:, 1:] = np.cumsum(np.cumsum(dmg.astype(int), 0), 1)

    def dcount(r, c, rh, rw):
        return int(ii[r + rh, c + rw] - ii[r, c + rw] - ii[r + rh, c] + ii[r, c])

    for comp in components(dmg):
        pending = set(comp)
        while pending:
            r0, c0 = min(pending)
            best, tried = None, 0
            for (dr, dc) in OFFS:
                sr, sc = r0 - dr, c0 - dc
                if not (0 <= sr < h and 0 <= sc < w) or dmg[sr, sc]:
                    continue
                tried += 1
                if tried > 16:
                    break
                maxh = min(h - r0, h - sr)
                rw = 0
                while True:
                    rw += 1
                    if c0 + rw > w or sc + rw > w:
                        break
                    rh = maxh
                    while rh > 0 and dcount(sr, sc, rh, rw) > 0:
                        rh -= 1
                    if rh == 0:
                        break
                    maxh = rh
                    key = (dcount(r0, c0, rh, rw), -(rh * rw))
                    if best is None or key > best[0]:
                        best = (key, (dr, dc), rh, rw)
            if best is None:
                raise ValueError("no undamaged copy of this cell is visible")
            (dr, dc), rh, rw = best[1], best[2], best[3]
            stamp(r0, c0, rh, rw, dr, dc)
            for r in range(r0, r0 + rh):
                for c in range(c0, c0 + rw):
                    pending.discard((r, c))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # full-grid rectangle: submit
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
