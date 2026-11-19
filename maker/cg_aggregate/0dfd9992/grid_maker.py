"""
ARC Task: 0dfd9992 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = sample(cols, 2)
    rotf = choice((identity, rot90, rot180, rot270))
    if rotf in (rot90, rot270):
        hmax, wmax = min(30, max_w), min(30, max_h)
    else:
        hmax, wmax = min(30, max_h), min(30, max_w)
    hmax = max(10, hmax); wmax = max(10, wmax)
    h = unifint(diff_lb, diff_ub, (10, hmax))
    w = unifint(diff_lb, diff_ub, (10, wmax))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading: every row of the wallpaper is a verbatim copy of other rows
    (and likewise every column).  The intruding patches punch holes into some
    rows; a damaged row is repaired by laying an intact row that AGREES with it
    on every cell both still show over it (CopyO the row + transparent Paste).
    Rows first, then columns, repeated until no hole can be filled."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def comps(mask):
        seen = np.zeros(mask.shape, dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)]); seen[r, c] = True; cells = []
                    while q:
                        y, x = q.popleft(); cells.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; q.append((ny, nx))
                    out.append(cells)
        return out

    # ── intruder colour: what the demonstrations erase ──────────────────
    noise = None
    if examples:
        cnt = {}
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            if ei.shape == eo.shape:
                for v in ei[ei != eo].tolist():
                    cnt[v] = cnt.get(v, 0) + 1
        if cnt:
            noise = max(cnt, key=cnt.get)
    if noise is None or not (I == noise).any():
        def periods_ok(ok, axis):
            n = I.shape[axis]
            for p in range(1, n):
                if axis == 0:
                    a, b, ka, kb = I[p:], I[:n - p], ok[p:], ok[:n - p]
                else:
                    a, b, ka, kb = I[:, p:], I[:, :n - p], ok[:, p:], ok[:, :n - p]
                if not ((a != b) & ka & kb).any():
                    return p
            return n
        best = None
        for col in sorted(set(I.flatten().tolist())):
            m = (I == col)
            if (~m.any(axis=1)).sum() < 2 or (~m.any(axis=0)).sum() < 2:
                continue
            pr, pc = periods_ok(~m, 0), periods_ok(~m, 1)
            key = (pr * pc, len(comps(m)), int(m.sum()), col)
            if best is None or key < best:
                best = key
        noise = best[3] if best else int(I[0, 0])

    mask = (I == noise)
    ops, sels = [], []
    g = I.copy()
    known = ~mask

    # ── punch out each intruding patch (connected blob of intruder colour) ──
    for cells in comps(mask):
        if noise != 0:
            ops.append(0); sels.append(sel_of(cells))
        for r, c in cells:
            g[r, c] = 0

    # ── which lines are copies of which: the wallpaper repeats every p lines
    #    and mirrors i <-> A-i (mod p); measured from the untouched cells ──
    def line_classes(vertical):
        n = wi if vertical else hi
        V = I.T if vertical else I
        K = (~mask).T if vertical else ~mask
        ok = np.ones((n, n), dtype=bool)
        sup = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                both = K[i] & K[j]
                sup[i, j] = int(both.sum())
                ok[i, j] = not (both & (V[i] != V[j])).any()
        def per_ok(q):
            return all(ok[i, i + q] for i in range(n - q))
        def mir(q, A):
            s = 0
            for i in range(n):
                for j in range(n):
                    if i != j and (i + j - A) % q == 0:
                        if not ok[i, j]:
                            return -1
                        s += sup[i, j]
            return s
        pers = [q for q in range(1, n) if per_ok(q)]
        for q in pers:
            best = None
            for A in range(q):
                s = mir(q, A)
                if s > 0 and (best is None or s > best[0]):
                    best = (s, A)
            if best is not None:
                A = best[1]
                return lambda t, s: (s - t) % q == 0 or (s + t - A) % q == 0
        if pers:
            q = pers[0]
            return lambda t, s: (s - t) % q == 0
        return lambda t, s: False

    same_row = line_classes(False)
    same_col = line_classes(True)

    def repair_lines(vertical):
        progress = False
        n = wi if vertical else hi
        for t in range(n):
            tg = g[:, t] if vertical else g[t, :]
            tk = known[:, t] if vertical else known[t, :]
            if tk.all():
                continue
            best = None
            for s in range(n):
                if s == t or not (same_col if vertical else same_row)(t, s):
                    continue
                sg = g[:, s] if vertical else g[s, :]
                sk = known[:, s] if vertical else known[s, :]
                both = tk & sk
                if (both & (tg != sg)).any():
                    continue
                gain = (~tk) & sk
                if not gain.any():
                    continue
                key = (int(both.sum()), int(gain.sum()), -abs(s - t))
                if best is None or key > best[0]:
                    best = (key, s)
            if best is None:
                continue
            s = best[1]
            sg = (g[:, s] if vertical else g[s, :]).copy()
            sk = (known[:, s] if vertical else known[s, :]).copy()
            gain = (~tk) & sk
            fill = gain & (sg != 0)
            if fill.any():
                if vertical:
                    ops.append(29); sels.append([0, s, hi - 1, 0])   # whole column s (full rectangle)
                    ops.append(30); sels.append([0, t, 0, 0])
                else:
                    ops.append(29); sels.append([s, 0, 0, wi - 1])   # whole row s (full rectangle)
                    ops.append(30); sels.append([t, 0, 0, 0])
                tg[fill] = sg[fill]
            tk |= gain
            progress = True
        return progress

    for _ in range(12):
        if known.all():
            break
        a = repair_lines(False)
        if known.all():
            break
        b = repair_lines(True)
        if not (a or b):
            break

    # ── drop any copy/paste the final grid does not depend on ──────────
    def run(ol, sl):
        gg = I.copy()
        clip = None
        for op, s in zip(ol, sl):
            if isinstance(s, dict):
                for r, c in s["cells"]:
                    gg[r, c] = op
                continue
            r, c, h, w = s
            if op <= 9:
                gg[r:r + h + 1, c:c + w + 1] = op
            elif op == 29:
                clip = gg[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30 and clip is not None:
                p = clip[:hi - r, :wi - c]
                tgt = gg[r:r + p.shape[0], c:c + p.shape[1]]
                np.copyto(tgt, p, where=(p > 0))
        return gg

    target = run(ops, sels)
    changed = True
    while changed:
        changed = False
        k = 0
        while k + 1 < len(ops):
            if ops[k] == 29 and ops[k + 1] == 30:
                o2 = ops[:k] + ops[k + 2:]; s2 = sels[:k] + sels[k + 2:]
                if (run(o2, s2) == target).all():
                    ops, sels = o2, s2; changed = True
                    continue
            k += 1
        k = 0
        while k < len(ops):
            o2 = ops[:k] + ops[k + 1:]; s2 = sels[:k] + sels[k + 1:]
            if (run(o2, s2) == target).all():
                ops, sels = o2, s2; changed = True
                continue
            k += 1

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
                        f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 0dfd9992"
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
                                f"for task 0dfd9992"
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
                    f"Failed to build a complete episode for task 0dfd9992 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"0dfd9992-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
