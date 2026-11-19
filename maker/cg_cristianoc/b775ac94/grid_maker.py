"""
ARC Task: b775ac94 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour. Each object picks its own four
    # colours, and the rule (mirror the blob toward each marker, in that marker's
    # colour) does not depend on which colours they are.
    return {"bgc": random.choice(range(10))}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + d * (b - a)))))


def _nbrs(p):
    r, c = p
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kwargs) -> dict:
    remcols = [c for c in range(10) if c != bgc]
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = _unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    maxtr = 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(2, 5)
        ow = random.randint(2, 5)
        c1, c2, c3, c4 = random.sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        for _ in range(ncells):
            cand = set()
            for p in obj:
                cand |= _nbrs(p)
            cand = sorted((bounds - obj) & cand)
            obj.add(random.choice(cand))
        gLR = np.full((oh, ow), bgc, dtype=int)
        for (i, j) in obj:
            gLR[i, j] = c1

        def rep(g, a, b):
            g = g.copy()
            g[g == a] = b
            return g

        gLL = rep(np.fliplr(gLR), c1, c2)
        gUR = rep(np.flipud(gLR), c1, c3)
        gUL = rep(np.fliplr(np.flipud(gLR)), c1, c4)
        g = np.vstack([np.hstack([gUL, gUR]), np.hstack([gLL, gLR])])
        g2 = np.full((oh * 2, ow * 2), bgc, dtype=int)
        for (i, j) in obj:
            g2[i + oh, j + ow] = c1
        nkeep = _unifint(diff_lb, diff_ub, (1, 3))
        keep = random.sample((c2, c3, c4), nkeep)
        for cc in (c2, c3, c4):
            if cc not in keep:
                g = rep(g, cc, bgc)
            else:
                oi = -1 if cc in (c3, c4) else 0
                oj = -1 if cc in (c2, c4) else 0
                g2[oh + oi, ow + oj] = cc
        k = random.choice((0, 3, 2, 1))  # identity, rot90, rot180, rot270
        g = np.rot90(g, k)
        g2 = np.rot90(g2, k)
        objo = [(int(g[i, j]), (i, j)) for i in range(g.shape[0])
                for j in range(g.shape[1]) if g[i, j] != bgc]
        obji = [(int(g2[i, j]), (i, j)) for i in range(g2.shape[0])
                for j in range(g2.shape[1]) if g2[i, j] != bgc]
        r0 = min(p[0] for _, p in objo)
        c0 = min(p[1] for _, p in objo)
        objo = [(v, (i - r0, j - c0)) for v, (i, j) in objo]
        obji = [(v, (i - r0, j - c0)) for v, (i, j) in obji]
        ohh = max(p[0] for _, p in objo) + 1
        oww = max(p[1] for _, p in objo) + 1
        cands = [ij for ij in inds if ij[0] <= h - ohh and ij[1] <= w - oww]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        plcdoi = {(i + li, j + lj) for _, (i, j) in objo}
        if plcdoi.issubset(inds):
            succ += 1
            nb = set()
            for p in plcdoi:
                nb |= _nbrs(p)
            inds = (inds - plcdoi) - nb
            for v, (i, j) in obji:
                gi[i + li, j + lj] = v
            for v, (i, j) in objo:
                go[i + li, j + lj] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Rule (worked out from I and the demonstrations only):
      Each object is a blob S of one colour with at least two cells, plus 1-3
      single-pixel markers touching ONE corner of S's bounding box (beside it,
      above/below it, or diagonal to it). Each marker spawns a copy of S reflected
      across the bbox edge(s) the marker sits beyond, painted in the marker's colour.

    The code-golf reading AGREES with this one. It also treats "one marker = one
    mirrored copy of the main-colour blob" as the unit, and the side a marker sits on
    (relative to the blob's corner cell) picks the reflection. So nothing changes:
    one Color op stamps each whole mirrored copy. Its selection is the set of cells
    that copy lands on, which comes from I alone. O is used only for the Submit
    extent, and that equals I's shape.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour shared by the demo inputs (fallback: majority of I)
    bgc = None
    if examples:
        try:
            cnt = Counter()
            for ex in examples:
                ei = np.asarray(ex["input"] if isinstance(ex, dict) else ex[0], dtype=int)
                cnt[Counter(ei.flatten().tolist()).most_common(1)[0][0]] += 1
            bgc = cnt.most_common(1)[0][0]
            if bgc not in set(I.flatten().tolist()):
                bgc = None
        except Exception:
            bgc = None
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    seen = set()
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or (sr, sc) in seen:
                continue
            # one placed object = one 8-connected non-background component
            comp = []
            dq = deque([(sr, sc)])
            seen.add((sr, sc))
            while dq:
                pr, pc = dq.popleft()
                comp.append((pr, pc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        qr, qc = pr + dr, pc + dc
                        if 0 <= qr < hi and 0 <= qc < wi and (qr, qc) not in seen \
                                and I[qr, qc] != bgc:
                            seen.add((qr, qc))
                            dq.append((qr, qc))
            groups = {}
            for p in comp:
                groups.setdefault(int(I[p]), []).append(p)
            # blob = the largest colour group; markers are single pixels
            shape_col, S = max(groups.items(), key=lambda kv: (len(kv[1]), -kv[0]))
            if len(S) < 2:
                continue
            r0 = min(p[0] for p in S); r1 = max(p[0] for p in S)
            c0 = min(p[1] for p in S); c1 = max(p[1] for p in S)
            markers = []
            for mcol, cells in groups.items():
                if mcol == shape_col:
                    continue
                for (mr, mc) in cells:
                    dr = -1 if mr == r0 - 1 else (1 if mr == r1 + 1 else 0)
                    dc = -1 if mc == c0 - 1 else (1 if mc == c1 + 1 else 0)
                    if dr == 0 and dc == 0:
                        continue
                    markers.append((dr, dc, mcol))
            markers.sort()
            for dr, dc, mcol in markers:
                copy = set()
                for (a, b) in S:
                    na = a if dr == 0 else (2 * r0 - 1 - a if dr < 0 else 2 * r1 + 1 - a)
                    nb = b if dc == 0 else (2 * c0 - 1 - b if dc < 0 else 2 * c1 + 1 - b)
                    if 0 <= na < hi and 0 <= nb < wi:
                        copy.add((na, nb))
                # a stamp whose cells all already hold mcol would be a no-op
                if not copy or all(I[p] == mcol for p in copy):
                    continue
                # stamp the whole mirrored blob in the marker's colour
                ops.append(int(mcol))
                sels.append(sel_of(sorted(copy)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b775ac94"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b775ac94"
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
                                f"for task b775ac94"
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
                    f"Failed to build a complete episode for task b775ac94 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b775ac94-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
