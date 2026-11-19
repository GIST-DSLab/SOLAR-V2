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
from collections import Counter
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, a + round(d * (b - a))))


def sample_colors(num_examples=None) -> dict:
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def _is_upscaled(P):
    P = np.asarray(P, dtype=bool)
    h, w = P.shape
    for k in range(2, min(h, w) + 1):
        if h % k or w % k:
            continue
        if np.array_equal(P[::k, ::k].repeat(k, 0).repeat(k, 1), P):
            return True
    return False


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, noisec=1, objc=2) -> dict:
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = [{(a + di, b + dj) for a, b in nbh} for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1))]
    for _ in range(1000):
        oh = _unifint(diff_lb, diff_ub, (2, 6))
        ow = _unifint(diff_lb, diff_ub, (2, 6))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]

        def mk():
            ncellsd = _unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
            ncells = random.choice((ncellsd, oh * ow - ncellsd))
            ncells = min(max(1, ncells), oh * ow - 1)
            return set(random.sample(bounds, ncells))

        obj = mk()
        tries = 0
        while not any(
            sum(len(obj & {(a + i, b + j) for a, b in nb}) < 4 for nb in nbhs) > 0
            for (i, j) in obj
        ):
            obj = mk()
            tries += 1
            if tries > 100:
                break
        mi = min(i for i, j in obj); mj = min(j for i, j in obj)
        obj = {(i - mi, j - mj) for i, j in obj}
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        P = np.zeros((oh, ow), dtype=bool)
        for i, j in obj:
            P[i, j] = True
        if _is_upscaled(P):
            continue  # scale factor would be ambiguous
        go = np.full((oh, ow), bgc, dtype=int)
        go[P] = noisec
        fmax = min(min(28, max_h - 2) // oh, min(28, max_w - 2) // ow)
        if fmax < 2:
            continue
        fac = _unifint(diff_lb, diff_ub, (2, fmax))
        G = P.repeat(fac, 0).repeat(fac, 1)
        gh, gw = G.shape
        h = _unifint(diff_lb, diff_ub, (gh + 2, max_h))
        w = _unifint(diff_lb, diff_ub, (gw + 2, max_w))
        loci = random.randint(1, h - gh - 1)
        locj = random.randint(1, w - gw - 1)
        gi = np.full((h, w), bgc, dtype=int)
        gi[loci:loci + gh, locj:locj + gw][G] = objc
        cands = [tuple(x) for x in np.argwhere(gi == bgc).tolist()]
        namt = _unifint(diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
        for r, c in random.sample(cands, namt):
            gi[r, c] = noisec
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    # background: dominant colour of the border ring (the object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # scale reading: the object colour is the one whose cells, inside their own bbox,
    # form an exact k-fold upscale (k >= 2); scattered noise does not.
    def scale_of(c):
        cells = np.argwhere(I == c)
        a0, b0 = cells.min(0); a1, b1 = cells.max(0)
        M = I[a0:a1 + 1, b0:b1 + 1] == c
        H, W = M.shape
        for k in range(min(H, W), 1, -1):
            if H % k or W % k:
                continue
            if np.array_equal(M[::k, ::k].repeat(k, 0).repeat(k, 1), M):
                return k
        return 1

    objc = noisec = None
    # colour roles are fixed per episode: read them from the demonstrations
    if examples:
        out_cols = set()
        for ei, eo in examples:
            out_cols |= set(np.unique(np.asarray(eo)).tolist())
        cand_obj = [c for c in others if c not in out_cols]
        cand_noise = [c for c in others if c in out_cols]
        if len(cand_obj) == 1 and len(cand_noise) == 1:
            objc, noisec = cand_obj[0], cand_noise[0]
    if objc is None:
        objc = max(others, key=lambda c: (scale_of(c), int((I == c).sum())))
        noisec = [c for c in others if c != objc][0]

    fac = scale_of(objc)
    cells = np.argwhere(I == objc)
    r0, c0 = (int(v) for v in cells.min(0))
    r1, c1 = (int(v) for v in cells.max(0))
    M = I[r0:r1 + 1, c0:c1 + 1] == objc
    P = M[::fac, ::fac]                 # one sample per fac x fac block
    ph, pw = P.shape

    ops, sels = [], []
    # 1. the shrunk picture lives in the bbox's top-left ph x pw window:
    #    lay the background over that whole rectangle (full rectangle intended)
    win = I[r0:r0 + ph, c0:c0 + pw]
    if bool(np.any(win != bgc)):
        ops.append(int(bgc)); sels.append([r0, c0, ph - 1, pw - 1])
    # 2. draw the sampled object (one cell per block) in the noise colour, as one unit
    small = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(P)]
    ops.append(int(noisec)); sels.append(sel_of(small))
    # 3. keep only the window (full rectangle)
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
