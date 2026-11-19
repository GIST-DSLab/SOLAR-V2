"""
ARC Task: e76a88a6 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    dmyc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dmyc": dmyc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dmyc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    def dn(ij):
        i, j = ij
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    def norm(cells):
        mi = min(i for i, j in cells); mj = min(j for i, j in cells)
        return {(i - mi, j - mj) for i, j in cells}

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if dmyc is None or dmyc == bgc:
        dmyc = random.choice([c for c in cols if c != bgc])
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = {(i, j) for i in range(objh) for j in range(objw)}
    shp = {random.choice(sorted(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for _ in range(nc):
        nb = set()
        for c in shp:
            nb |= dn(c)
        cand = sorted((bounds - shp) & nb)
        shp.add(random.choice(cand))
    shp = norm(shp)
    remcols = [c for c in cols if c not in (bgc, dmyc)]
    oh = max(i for i, j in shp) + 1
    ow = max(j for i, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    shpp = {(i + loci, j + locj) for i, j in shp}
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = random.sample(remcols, numco)
    while True:
        shppc = {ij: random.choice(colll) for ij in shpp}
        if len(set(shppc.values())) > 1:
            break
    shppcn = {(i - loci, j - locj): v for (i, j), v in shppc.items()}
    gi = np.full((h, w), bgc, dtype=int)
    for (i, j), v in shppc.items():
        gi[i, j] = v
    go = gi.copy()
    ub = max(1, int(((h * w) / (oh * ow)) // 2))
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0; fails = 0; maxfails = 5 * numlocs
    allc = {(i, j) for i in range(h) for j in range(w)}
    nbs = set()
    for c in shpp:
        nbs |= dn(c)
    idns = (allc - shpp) - nbs
    idns = {ij for ij in idns if ij[0] <= h - oh and ij[1] <= w - ow}
    while cnt < numlocs and fails < maxfails:
        if not idns:
            break
        loc = random.choice(sorted(idns))
        plcd = {(i + loc[0], j + loc[1]): v for (i, j), v in shppcn.items()}
        plcdi = set(plcd)
        if plcdi <= idns:
            for (i, j), v in plcd.items():
                go[i, j] = v
                gi[i, j] = dmyc
            cnt += 1
            nb2 = set()
            for c in plcdi:
                nb2 |= dn(c)
            idns = (idns - plcdi) - nb2
        else:
            fails += 1
    return {"input": gi.tolist(), "output": go.tolist()}


def _components(I, bgc):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                a, b = stack.pop(); cells.append((a, b))
                for na, nb in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                    if 0 <= na < H and 0 <= nb < W and not seen[na, nb] and I[na, nb] != bgc:
                        seen[na, nb] = True; stack.append((na, nb))
            comps.append(cells)
    return comps


def _bgc(I, examples):
    # background = most frequent colour of I that also appears in every demo input
    if examples:
        common = None
        for ei, _ in examples:
            s = set(np.asarray(ei).flatten().tolist())
            common = s if common is None else common & s
        cnt = Counter(I.flatten().tolist())
        cands = [c for c, _ in cnt.most_common() if common is None or c in common]
        if cands:
            return cands[0]
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc = _bgc(I, examples)
    comps = _components(I, bgc)
    ops, sels = [], []
    if comps:
        # template = the multicolour component (argmax numcolors)
        ncol = [len({int(I[r, c]) for r, c in cells}) for cells in comps]
        ti = int(np.argmax(ncol))
        tmpl = comps[ti]
        tr = min(r for r, c in tmpl); tc = min(c for r, c in tmpl)
        rel = {(r - tr, c - tc): int(I[r, c]) for r, c in tmpl}
        relshape = set(rel)
        # placeholders = single-colour components with the template's shape
        targets = []
        for k, cells in enumerate(comps):
            if k == ti or ncol[k] != 1:
                continue
            r0 = min(r for r, c in cells); c0 = min(c for r, c in cells)
            if {(r - r0, c - c0) for r, c in cells} == relshape:
                targets.append((r0, c0))
        if targets:
            # copy ONLY the template's own cells (mask), not its bbox background
            ops.append(28); sels.append(sel_of(tmpl))
            zero_rel = [p for p, v in rel.items() if v == 0]
            for (r0, c0) in sorted(targets):
                ops.append(30); sels.append(sel_of([(r0, c0)]))
                if zero_rel:
                    # colour-0 template cells are not carried by the clipboard
                    ops.append(0); sels.append(sel_of([(r0 + a, c0 + b) for a, b in zero_rel]))
    ops.append(34); sels.append(sel_of([(r, c) for r in range(H) for c in range(W)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e76a88a6"
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
                                f"for task e76a88a6"
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
                    f"Failed to build a complete episode for task e76a88a6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e76a88a6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
