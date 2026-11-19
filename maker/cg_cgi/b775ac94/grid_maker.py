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
import numpy as np
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is fixed per episode. The blob and marker colours are
    # per-object roles: the rule copies whatever colour each marker has.
    return {"bgc": random.choice(range(10))}


def _unif(lb, ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(lb, ub)
    return max(lo, min(hi, lo + int(round((hi - lo) * d + random.uniform(-0.5, 0.5)))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.randrange(10)
    remcols = [c for c in range(10) if c != bgc]
    h = _unif(diff_lb, diff_ub, 10, max(10, max_h))
    w = _unif(diff_lb, diff_ub, 10, max(10, max_w))
    nobjs = _unif(diff_lb, diff_ub, 1, (h * w) // 25)
    gi = np.full((h, w), bgc, dtype=int)
    go = gi.copy()
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = tr = 0
    while succ < nobjs and tr < 5 * nobjs:
        tr += 1
        oh, ow = random.randint(2, 5), random.randint(2, 5)
        c1, c2, c3, c4 = random.sample(remcols, 4)
        obj = {(0, 0)}
        nd = _unif(diff_lb, diff_ub, 0, (oh * ow) // 2)
        nc = random.choice((nd, oh * ow - nd))
        nc = min(max(1, nc), oh * ow - 1)
        for _ in range(nc):
            fr = sorted({(a + da, b + db) for a, b in obj for da in (-1, 0, 1) for db in (-1, 0, 1)
                         if 0 <= a + da < oh and 0 <= b + db < ow} - obj)
            obj.add(random.choice(fr))
        g = np.full((2 * oh, 2 * ow), bgc, dtype=int)
        g2 = g.copy()
        keep = random.sample((c2, c3, c4), _unif(diff_lb, diff_ub, 1, 3))
        for a, b in obj:
            g[oh + a, ow + b] = c1
            g2[oh + a, ow + b] = c1
            if c2 in keep:
                g[oh + a, ow - 1 - b] = c2
            if c3 in keep:
                g[oh - 1 - a, ow + b] = c3
            if c4 in keep:
                g[oh - 1 - a, ow - 1 - b] = c4
        if c2 in keep:
            g2[oh, ow - 1] = c2
        if c3 in keep:
            g2[oh - 1, ow] = c3
        if c4 in keep:
            g2[oh - 1, ow - 1] = c4
        k = random.randint(0, 3)
        g = np.rot90(g, k)
        g2 = np.rot90(g2, k)
        objo = [(int(g[i, j]), i, j) for i in range(g.shape[0]) for j in range(g.shape[1]) if g[i, j] != bgc]
        obji = [(int(g2[i, j]), i, j) for i in range(g2.shape[0]) for j in range(g2.shape[1]) if g2[i, j] != bgc]
        r0 = min(x[1] for x in objo)
        q0 = min(x[2] for x in objo)
        objo = [(v, i - r0, j - q0) for v, i, j in objo]
        obji = [(v, i - r0, j - q0) for v, i, j in obji]
        ph = max(x[1] for x in objo) + 1
        pw = max(x[2] for x in objo) + 1
        cands = [p for p in inds if p[0] <= h - ph and p[1] <= w - pw]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        cells = {(i + li, j + lj) for _, i, j in objo}
        if cells <= inds:
            succ += 1
            nb = {(a + da, b + db) for a, b in cells for da in (-1, 0, 1) for db in (-1, 0, 1)}
            inds = inds - cells - nb
            for v, i, j in obji:
                gi[i + li, j + lj] = v
            for v, i, j in objo:
                go[i + li, j + lj] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the code-golf entry): each single-pixel marker at a corner
    of a blob S starts a mirror image of S. The image is S reflected across the
    bbox edge(s) the marker sits beyond, and the marker is its first cell. The
    image GROWS outward from the marker: cells at king-move distance d from the
    marker are painted together in the marker's colour, as one wave per d.
    The maker stamps each whole copy in one op; this paints it wave by wave from
    its seed. Every selection comes from I alone (S's cells reflected).
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: the canvas colour every demonstration input shares
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0]).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    ops, sels = [], []
    seen = set()
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or (sr, sc) in seen:
                continue
            # one placed object = one 8-connected non-background component of I
            comp = []
            dq = deque([(sr, sc)])
            seen.add((sr, sc))
            while dq:
                pr, pc = dq.popleft()
                comp.append((pr, pc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        q = (pr + dr, pc + dc)
                        if 0 <= q[0] < hi and 0 <= q[1] < wi and q not in seen and I[q] != bgc:
                            seen.add(q)
                            dq.append(q)
            groups = {}
            for p in comp:
                groups.setdefault(int(I[p]), []).append(p)
            # the blob is the largest colour group; markers are single pixels
            scol, S = max(groups.items(), key=lambda kv: (len(kv[1]), -kv[0]))
            if len(S) < 2:
                continue
            r0 = min(p[0] for p in S)
            r1 = max(p[0] for p in S)
            c0 = min(p[1] for p in S)
            c1 = max(p[1] for p in S)

            markers = []
            for mcol, cells in groups.items():
                if mcol == scol:
                    continue
                for (mr, mc) in cells:
                    dr = -1 if mr == r0 - 1 else (1 if mr == r1 + 1 else 0)
                    dc = -1 if mc == c0 - 1 else (1 if mc == c1 + 1 else 0)
                    if dr or dc:
                        markers.append((dr, dc, mcol, mr, mc))
            markers.sort()

            for dr, dc, mcol, mr, mc in markers:
                # the mirror image of S across the edge(s) beyond which the marker sits
                copy = set()
                for a, b in S:
                    na = a if dr == 0 else (2 * r0 - 1 - a if dr < 0 else 2 * r1 + 1 - a)
                    nb = b if dc == 0 else (2 * c0 - 1 - b if dc < 0 else 2 * c1 + 1 - b)
                    if 0 <= na < hi and 0 <= nb < wi:
                        copy.add((na, nb))
                # grow it outward from the marker cell, one wave per distance
                dist = {(mr, mc): 0}
                dq = deque([(mr, mc)])
                waves = {}
                while dq:
                    p = dq.popleft()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            q = (p[0] + da, p[1] + db)
                            if q in copy and q not in dist:
                                dist[q] = dist[p] + 1
                                dq.append(q)
                                waves.setdefault(dist[q], []).append(q)
                rest = sorted(copy - set(dist))
                order = [waves[d] for d in sorted(waves)] + ([rest] if rest else [])
                for wave in order:
                    wave = [p for p in wave if I[p] != mcol]
                    if wave:
                        ops.append(int(mcol))
                        sels.append(sel_of(sorted(wave)))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
