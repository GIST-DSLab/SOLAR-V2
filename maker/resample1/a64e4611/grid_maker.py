"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    if bgc is None or noisec is None:
        bgc, noisec = random.sample([c for c in range(10) if c != 3], 2)
    # The raw generator sometimes paints 3s that the task rule (3 = interior of every
    # straight background corridor >= 3 wide reaching the grid edge) does not imply,
    # or leaves stray noise corridors. Keep only instances consistent with the rule.
    def _raw():
        mh = min(max_h, 30); mw = min(max_w, 30)
        h = unifint(diff_lb, diff_ub, (min(18, mh), mh))
        w = unifint(diff_lb, diff_ub, (min(18, mw), mw))
        lb = int(0.4 * h * w)
        ub = int(0.5 * h * w)
        nbgc = unifint(diff_lb, diff_ub, (lb, ub))
        gi = canvas(noisec, (h, w))
        inds = totuple(asindices(gi))
        bgcinds = sample(inds, nbgc)
        gi = fill(gi, bgc, bgcinds)
        sinds = asindices(canvas(-1, (3, 3)))
        bgcf = recolor(bgc, sinds)
        noisecf = recolor(noisec, sinds)
        addn = set()
        addb = set()
        for occ in occurrences(gi, bgcf):
            occi, occj = occ
            addn.add((randint(0, 2) + occi, randint(0, 2) + occj))
        for occ in occurrences(gi, noisecf):
            occi, occj = occ
            addb.add((randint(0, 2) + occi, randint(0, 2) + occj))
        gi = fill(gi, noisec, addn)
        gi = fill(gi, bgc, addb)
        go = tuple(e for e in gi)
        dim = randint(randint(3, 8), 8)
        locj = randint(3, w - dim - 4)
        spi = choice((0, randint(3, h // 2)))
        for j in range(locj, locj + dim):
            ln = connect((spi, j), (h - 1, j))
            gi = fill(gi, bgc, ln)
            go = fill(go, bgc, ln)
        for j in range(locj + 1, locj + dim - 1):
            ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
            go = fill(go, 3, ln)
        sgns = choice(((-1,), (1,), (-1, 1)))
        startloc = choice((spi, randint(spi + 3, h - 6)))
        hh = randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh, 1):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
                go = fill(go, 3, ln)
        if len(sgns) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
            sgns = (-sgns[0],)
            startloc = choice((spi, randint(spi + 3, h - 6)))
            hh = randint(3, min(8, h - startloc - 3))
            for sgn in sgns:
                for ii in range(startloc, startloc + hh, 1):
                    ln = shoot((ii, locj), (0, sgn))
                    gi = fill(gi, bgc, ln)
                    go = fill(go, bgc, ln - ofcolor(go, 3))
            for sgn in sgns:
                for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                    ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
                    go = fill(go, 3, ln)
        return {"input": gi, "output": go}

    for _ in range(100):
        ex = _raw()
        gi = np.asarray(ex["input"], dtype=int); go = np.asarray(ex["output"], dtype=int)
        if not (go == 3).any():
            continue
        ops, sels = derive_operations(gi, go, examples=[(gi, go)])
        G = gi.copy()
        for op, s in zip(ops, sels):
            if op < 10:
                for r, c in s["cells"]:
                    G[r, c] = op
        if (G == go).all():
            return ex
    raise ValueError("no rule-consistent instance")


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the colour the corridor is cut from: in the demos, the input
    # colour of every cell that becomes 3
    bgc = None
    if examples:
        cnt = Counter()
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                cnt.update(ei[eo == 3].tolist())
        if cnt:
            bgc = cnt.most_common(1)[0][0]
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    MARK = -1
    P = np.full((hi + 2, wi + 2), bgc, dtype=int)
    P[1:-1, 1:-1] = I

    def strip_mask(G):
        # centre row of every 3x14 all-background window
        Hh, Ww = G.shape
        m = np.zeros_like(G, dtype=bool)
        B = (G == bgc)
        for i in range(Hh - 2):
            for j in range(Ww - 13):
                if B[i:i + 3, j:j + 14].all():
                    m[i + 1, j + 1:j + 13] = True
        return m

    hmask = strip_mask(P)
    vmask = strip_mask(P.T).T
    G = P.copy()
    G[hmask | vmask] = MARK
    G[0, :] = MARK; G[-1, :] = MARK; G[:, 0] = MARK; G[:, -1] = MARK

    def occ(G, pat):
        Hh, Ww = G.shape
        ph = max(a for a, b, v in pat) + 1
        pw = max(b for a, b, v in pat) + 1
        res = []
        for i in range(Hh - ph + 1):
            for j in range(Ww - pw + 1):
                if all(G[i + a, j + b] == v for a, b, v in pat):
                    res.append((i, j))
        return res

    # trim 1- and 2-wide stubs poking into background (all 4 orientations, twice)
    pat1 = [(0, 1, bgc), (1, 0, bgc), (1, 2, bgc), (1, 1, MARK)]
    pat2 = [(1, 3, bgc), (0, 2, bgc), (0, 1, bgc), (1, 0, bgc), (1, 1, MARK), (1, 2, MARK)]
    for _ in range(8):
        for pat in (pat2, pat1):
            for (i, j) in occ(G, pat):
                for a, b, v in pat:
                    G[i + a, j + b] = bgc
        G = np.rot90(G, -1)
    S = G[1:-1, 1:-1] == MARK

    ops, sels = [], []
    # vertical corridor first, then each horizontal arm
    vcells = [(r, c) for r in range(hi) for c in range(wi) if S[r, c] and vmask[r + 1, c + 1]]
    if vcells:
        ops.append(3); sels.append(sel_of(vcells))
    rest = S.copy()
    for r, c in vcells:
        rest[r, c] = False
    seen = np.zeros_like(rest)
    for r in range(hi):
        for c in range(wi):
            if rest[r, c] and not seen[r, c]:
                comp = []; st = [(r, c)]; seen[r, c] = True
                while st:
                    y, x = st.pop(); comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and rest[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                ops.append(3); sels.append(sel_of(comp))
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
