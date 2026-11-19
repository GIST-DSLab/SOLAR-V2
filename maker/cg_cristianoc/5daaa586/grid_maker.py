"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    c1, c2, c3, c4 = random.sample([c for c in cols if c != bgc], 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "instance_plan": plan}


def _u(lb, ub, a, b):
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    v = a + int(round((b - a) * d + random.uniform(-0.5, 0.5)))
    return max(a, min(b, v))


def _nfront(g):
    n = 0
    for r in range(g.shape[0]):
        if len(set(g[r].tolist())) == 1:
            n += 1
    for c in range(g.shape[1]):
        if len(set(g[:, c].tolist())) == 1:
            n += 1
    return n


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    mh, mw = (max_w, max_h) if rot % 2 else (max_h, max_w)
    h = _u(diff_lb, diff_ub, 7, min(30, mh))
    w = _u(diff_lb, diff_ub, 7, min(30, mw))
    loci1 = random.randint(1, h - 4)
    locj1 = random.randint(1, w - 4)
    loci1 -= _u(diff_lb, diff_ub, 0, loci1 - 1)
    locj1 -= _u(diff_lb, diff_ub, 0, locj1 - 1)
    loci2 = _u(diff_lb, diff_ub, loci1 + 2, h - 2)
    locj2 = _u(diff_lb, diff_ub, locj1 + 2, w - 2)
    gi = np.full((h, w), bgc, dtype=int)
    fronts = [("r", loci1, c1), ("r", loci2, c2), ("c", locj1, c3), ("c", locj2, c4)]
    random.shuffle(fronts)
    for kind, idx, col in fronts:
        if kind == "r":
            gi[idx, :] = col
        else:
            gi[:, idx] = col
    cands = [(r, c) for r in range(h) for c in range(w) if gi[r, c] == bgc]
    nn = len(cands)
    for _ in range(50):
        g2 = gi.copy()
        nnoise = _u(diff_lb, diff_ub, 1, max(1, nn // 3))
        for r, c in random.sample(cands, nnoise):
            g2[r, c] = c1
        if _nfront(g2) <= 4:
            gi = g2
            break
    else:
        raise ValueError("noise retry")
    go = gi[loci1:loci2 + 1, locj1:locj2 + 1].copy()
    for r, c in zip(*np.nonzero(go == c1)):
        go[:r + 1, c] = c1
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Two readings were compared. The maker finds the four lines. The golf
    # entry finds the background pocket they enclose and takes its outbox.
    # Both pick out the same rectangle, and both draw one ray per column
    # (or row) from the matching line out to the farthest speckle, so the
    # ops come out the same. The one change: the frame is now chosen from
    # the input alone (no O.shape tie-break), so any answer grid gives the
    # same ops.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the episode's canvas colour, most common over all inputs
    cnt = Counter()
    for ex in (examples or []):
        cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    cnt.update(I.flatten().tolist())
    order = [c for c, _ in cnt.most_common()]

    def lines(bgc):
        rows, cols = [], []
        for r in range(h):
            for v, n in Counter(I[r].tolist()).items():
                if v != bgc and n >= w - 2:
                    rows.append((r, v))
        for c in range(w):
            for v, n in Counter(I[:, c].tolist()).items():
                if v != bgc and n >= h - 2:
                    cols.append((c, v))
        return rows, cols

    def find(bgc):
        rows, cols = lines(bgc)
        best = None
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                (r1, vt), (r2, vb) = rows[a], rows[b]
                if r2 - r1 < 2:
                    continue
                for x in range(len(cols)):
                    for y in range(x + 1, len(cols)):
                        (q1, vl), (q2, vr) = cols[x], cols[y]
                        if q2 - q1 < 2 or len({vt, vb, vl, vr}) != 4:
                            continue
                        if any(I[rr, j] != vv and j not in (q1, q2)
                               for rr, vv in ((r1, vt), (r2, vb)) for j in range(w)):
                            continue
                        if any(I[i, qq] != vv and i not in (r1, r2)
                               for qq, vv in ((q1, vl), (q2, vr)) for i in range(h)):
                            continue
                        pal = set(I[r1 + 1:r2, q1 + 1:q2].flatten().tolist()) - {bgc}
                        if len(pal) > 1 or (pal and next(iter(pal)) not in (vt, vb, vl, vr)):
                            continue
                        area = (r2 - r1) * (q2 - q1)
                        if best is None or area > best[0]:
                            best = (area, (r1, r2, q1, q2, vt, vb, vl, vr))
        return best

    frame, bgc = None, None
    for cb in order:
        f = find(cb)
        if f is not None:
            frame, bgc = f[1], cb
            break
    r1, r2, q1, q2, vt, vb, vl, vr = frame
    H, W = r2 - r1 + 1, q2 - q1 + 1
    sub = I[r1:r2 + 1, q1:q2 + 1]

    ops, sels = [], []
    # crop to the framed rectangle; the selection is the full rectangle
    ops.append(33); sels.append([r1, q1, H - 1, W - 1])

    pal = set(sub[1:H - 1, 1:W - 1].flatten().tolist()) - {bgc}
    if len(pal) == 1:
        nc = next(iter(pal))
        side = {vt: 'up', vb: 'down', vl: 'left', vr: 'right'}.get(nc)
        cur = sub.copy()
        segs = []
        if side in ('up', 'down'):
            for j in range(1, W - 1):
                hits = [i for i in range(1, H - 1) if sub[i, j] == nc]
                if hits:
                    rng = range(1, max(hits) + 1) if side == 'up' else range(min(hits), H - 1)
                    segs.append([(i, j) for i in rng])
        elif side in ('left', 'right'):
            for i in range(1, H - 1):
                hits = [j for j in range(1, W - 1) if sub[i, j] == nc]
                if hits:
                    rng = range(1, max(hits) + 1) if side == 'left' else range(min(hits), W - 1)
                    segs.append([(i, j) for j in rng])
        for seg in segs:
            if all(cur[a, b] == nc for a, b in seg):
                continue
            ops.append(int(nc)); sels.append(sel_of(seg))
            for a, b in seg:
                cur[a, b] = nc

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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
