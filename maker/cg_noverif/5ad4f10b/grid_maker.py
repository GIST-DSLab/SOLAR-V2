"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None, **kw) -> dict:
    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)
    max_h = min(30, max_h)
    max_w = min(30, max_w)

    def bad(obj):
        # generator condition: some cell not covered by all four 2x2 windows
        for (i, j) in obj:
            for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                if sum((i + di + a, j + dj + b) in obj for a in (0, 1) for b in (0, 1)) < 4:
                    return False
        return True

    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    while True:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(random.sample(bounds, ncells))
        if bad(obj):
            continue
        # the scale factor must be recoverable: the small pattern itself must not be
        # a k-fold upscale of something smaller (e.g. a solid rectangle)
        mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
        nobj = {(i - mi, j - mj) for i, j in obj}
        nh = max(i for i, _ in nobj) + 1
        nw = max(j for _, j in nobj) + 1
        amb = False
        for k in range(2, min(nh, nw) + 1):
            if nh % k == 0 and nw % k == 0 and all(
                    ((i // k * k, j // k * k) in nobj) == ((i, j) in nobj)
                    for i in range(nh) for j in range(nw)):
                amb = True
                break
        if not amb:
            break
    mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    go = [[noisec if (i, j) in obj else bgc for j in range(ow)] for i in range(oh)]
    fmax = min((max_h - 2) // oh, (max_w - 2) // ow, 28 // oh, 28 // ow)
    if fmax < 2:
        raise ValueError("grid too small")
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    gh, gw = oh * fac, ow * fac
    h = unifint(diff_lb, diff_ub, (gh + 2, max_h))
    w = unifint(diff_lb, diff_ub, (gw + 2, max_w))
    loci = random.randint(1, h - gh - 1)
    locj = random.randint(1, w - gw - 1)
    gi = [[bgc] * w for _ in range(h)]
    for i in range(gh):
        for j in range(gw):
            if go[i // fac][j // fac] == noisec:
                gi[loci + i][locj + j] = objc
            else:
                gi[loci + i][locj + j] = bgc
    cands = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc]
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    for i, j in random.sample(cands, namt):
        gi[i][j] = noisec
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the maker and the golf entry -- they agree): the big object is a
    # k-fold upscale; take one cell per k x k block, object cells become the noise colour,
    # everything else background.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background: dominant colour of the border ring (the big object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    present = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    objc = noisec = None
    # colour roles are fixed per episode: read them from the demonstrations
    if examples:
        oc, nc = Counter(), Counter()
        for ex in examples:
            ei, eo = (ex['input'], ex['output']) if isinstance(ex, dict) else ex
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            eb = Counter(ei[0, :].tolist() + ei[-1, :].tolist()
                         + ei[:, 0].tolist() + ei[:, -1].tolist()).most_common(1)[0][0]
            outs = set(int(v) for v in np.unique(eo).tolist()) - {eb}
            ins = set(int(v) for v in np.unique(ei).tolist()) - {eb}
            for v in outs:
                nc[v] += 1
            for v in ins - outs:
                oc[v] += 1
        if nc:
            cand = nc.most_common(1)[0][0]
            if cand in present:
                noisec = cand
        if oc:
            cand = oc.most_common(1)[0][0]
            if cand in present and cand != noisec:
                objc = cand

    # fallback: the object is a union of solid k x k blocks (every cell sits in a
    # monochrome 2x2 square); noise is scattered
    def block_score(c):
        m = (I == c)
        cs = list(zip(*np.nonzero(m)))
        cov = 0
        for r, cc in cs:
            ok = False
            for dr in (0, -1):
                for dc in (0, -1):
                    rr, c2 = r + dr, cc + dc
                    if 0 <= rr < hi - 1 and 0 <= c2 < wi - 1 and m[rr:rr + 2, c2:c2 + 2].all():
                        ok = True
            cov += ok
        return (cov == len(cs), cov, len(cs))

    if objc is None:
        cands = [c for c in present if c != noisec]
        objc = max(cands, key=block_score)
    if noisec is None:
        rest = [c for c in present if c != objc]
        noisec = rest[0]

    # the upscaled object, its bounding box and its scale factor
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        small = M.reshape(H // k, k, W // k, k).any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    P = M[::fac, ::fac]
    ph, pw = P.shape

    ops, sels = [], []
    # 1. lay a clean background base on the top-left ph x pw corner of the object's box
    win = I[r0:r0 + ph, c0:c0 + pw]
    if bool(np.any(win != bgc)):
        ops.append(int(bgc)); sels.append([r0, c0, ph - 1, pw - 1])  # full rectangle

    # 2. draw the shrunk object there in the noise colour, one connected piece at a time
    filled = set(map(tuple, np.argwhere(P).tolist()))
    seen = set()
    for cell in sorted(filled):
        if cell in seen:
            continue
        comp, q = set(), deque([cell])
        seen.add(cell)
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in filled and nb not in seen:
                    seen.add(nb); q.append(nb)
        cur = I[r0:r0 + ph, c0:c0 + pw] if not ops else None
        tgt = [(r0 + r, c0 + c) for r, c in sorted(comp)]
        if cur is not None and all(I[a, b] == noisec for a, b in tgt):
            continue
        ops.append(int(noisec)); sels.append(sel_of(tgt))

    # 3. keep only the shrunk object
    ops.append(33); sels.append([r0, c0, ph - 1, pw - 1])
    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
