"""
ARC Task: a78176bb (RE-ARC) — LLM-generated grid_maker
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

# The generator ends with a coin flip that hmirrors the whole scene.  That flip is the
# only discrete structural variant of this task: it decides whether the diagonals read
# "\" (canonical, what the verifier's rule is written against) or "/" (mirrored, where
# the verifier first hmirrors, applies the rule, and mirrors back).
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    random.shuffle(rem)
    fgcols = rem[:4]                      # fixed palette for line / triangle colors
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test variant was shown
    return {"bgc": bgc, "fgcols": fgcols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgcols=None, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])
    if not fgcols:
        rem = [c for c in range(10) if c != bgc]
        random.shuffle(rem)
        fgcols = rem[:4]
    remcols = [c for c in fgcols if c != bgc]
    if len(remcols) < 3:
        remcols = [c for c in range(10) if c != bgc]

    hmax = max(6, min(30, int(max_h)))
    wmax = max(6, min(30, int(max_w)))
    fallback = None

    for _att in range(300):
        h = unifint(diff_lb, diff_ub, (6, hmax))
        w = unifint(diff_lb, diff_ub, (6, wmax))
        nlns = unifint(diff_lb, diff_ub, (1, max(1, (h + w) // 8)))

        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        fullinds = asindices(gi)
        inds = set(fullinds)
        spopts = [(i, 0) for i in range(h - 5, -1, -1)] + [(0, j) for j in range(1, w - 4)]

        succ, tr, maxtr = 0, 0, 10 * nlns
        while succ < nlns and tr < maxtr and spopts:
            tr += 1
            sp = random.choice(spopts)
            ln = frozenset((sp[0] + k, sp[1] + k) for k in range(max(h, w))) & fullinds
            if len(ln) < 5 or not ln.issubset(inds):
                continue
            lno = sorted(ln)
            mx = min(5, len(lno) - 3)
            if mx < 2:
                continue

            cands = []
            for which in (1, 2):
                n = random.randint(2, mx)
                base = lno[1:-n - 1]
                if not base:
                    continue
                p = random.choice(base)
                if which == 1:
                    # upper-right wedge, hypotenuse on the line; its line sits (n+2) up-right
                    tri = frozenset((p[0] + i, p[1] + 1 + j)
                                    for i in range(n) for j in range(n) if j >= i)
                    spo = (sp[0] - n - 2, sp[1])
                else:
                    # lower-left wedge; its line sits (n+2) down-left
                    tri = frozenset((p[0] + 1 + i, p[1] + j)
                                    for i in range(n) for j in range(n) if i >= j)
                    spo = (sp[0], sp[1] - n - 2)
                nex = frozenset((spo[0] + k, spo[1] + k)
                                for k in range(-2 * max(h, w), 2 * max(h, w))) & fullinds
                if len(nex) < 3:
                    continue
                cands.append((tri, nex))
            if not cands:
                continue
            random.shuffle(cands)

            lncol = random.choice(remcols)
            others = [c for c in remcols if c != lncol]
            if len(others) < 2:
                continue
            tcs = random.sample(others, 2)

            group = set(ln)
            acc = []
            for tri, nex in cands:
                cells = set(tri) | set(nex)
                if cells.issubset(inds) and cells.isdisjoint(group):
                    acc.append((tri, nex, tcs[len(acc)]))
                    group |= cells
            if not acc:
                continue

            succ += len(acc)
            excl = set(group)
            for (r, c) in group:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        excl.add((r + dr, c + dc))
            inds -= excl

            for tri, nex, tc in acc:
                gi = fill(gi, tc, tri)
                go = fill(go, lncol, nex)
            gi = fill(gi, lncol, ln)
            go = fill(go, lncol, ln)

        if succ < 1:
            continue
        if fallback is None:
            fallback = (gi, go)
        cnt = Counter(v for row in gi for v in row)
        omax = max([m for c, m in cnt.items() if c != bgc] or [0])
        if cnt[bgc] <= 2 * omax:          # keep background unambiguously dominant
            continue
        if mirrored:
            gi = hmirror(gi)
            go = hmirror(go)
        return {'input': gi, 'output': go}

    gi, go = fallback if fallback else (canvas(bgc, (6, 6)), canvas(bgc, (6, 6)))
    if mirrored:
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def components(G):
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or G[r, c] == bgc:
                    continue
                col = int(G[r, c])
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and G[nr, nc] == col:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                out.append((col, cells))
        return out

    def is_bslash_line(cells):
        if len(cells) < 5:
            return False
        s = sorted(cells)
        r0, c0 = s[0]
        return all(s[k] == (r0 + k, c0 + k) for k in range(len(s)))

    ops, sels = [], []
    full_sel = [0, 0, h - 1, w - 1]       # bbox == the whole grid rectangle, background included

    comps = components(I)
    canonical = any(is_bslash_line(cl) for _, cl in comps)
    if canonical:
        G = I.copy()
        flipped = False
    else:
        # The scene is the up-down reflection of the canonical one: perform that reflection
        # (FlipV = hmirror) so the rule can be carried out, and undo it at the end.
        ops.append(27); sels.append(full_sel)
        G = np.flipud(I).copy()
        comps = components(G)
        flipped = True

    lines = [(col, cl) for col, cl in comps if is_bslash_line(cl)]
    tris = [(col, cl) for col, cl in comps if not is_bslash_line(cl)]

    linemap = {}
    line_key = {}
    for idx, (col, cl) in enumerate(lines):
        line_key[idx] = min(cl)
        for p in cl:
            linemap[p] = idx

    items = []
    for col, cl in tris:
        cs = set(cl)
        rs = [p[0] for p in cl]
        cls_ = [p[1] for p in cl]
        r0, r1, c0, c1 = min(rs), max(rs), min(cls_), max(cls_)
        if (r0, c1) in cs:
            # upper-right wedge: its parallel line runs through (r0-1, c1+1)
            anchor = (r0, c0 - 1)
            k = (c1 + 1) - (r0 - 1)
        else:
            # lower-left wedge: its parallel line runs through (r1+1, c0-1)
            anchor = (r0 - 1, c0)
            k = (c0 - 1) - (r1 + 1)
        lid = linemap.get(anchor, -1)
        items.append((lid, anchor, k, cl))

    items.sort(key=lambda it: (line_key.get(it[0], (10 ** 6, 10 ** 6)), it[1]))

    for lid, anchor, k, cl in items:
        ar, ac = anchor
        lncol = int(G[max(0, min(h - 1, ar)), max(0, min(w - 1, ac))])
        newcells = [(r, r + k) for r in range(h) if 0 <= r + k < w and G[r, r + k] != lncol]
        if newcells:
            # draw the wedge's parallel diagonal, in the colour of the line it belongs to
            ops.append(lncol); sels.append(sel_of(newcells))
            for (r, c) in newcells:
                G[r, c] = lncol
        wedge = [(r, c) for (r, c) in cl if G[r, c] != bgc]
        if wedge:
            # the wedge itself is consumed by the rule
            ops.append(int(bgc)); sels.append(sel_of(wedge))
            for (r, c) in wedge:
                G[r, c] = bgc

    if flipped:
        ops.append(27); sels.append(full_sel)   # reflect back to the scene's own orientation

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a78176bb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a78176bb"
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
                                f"for task a78176bb"
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
                    f"Failed to build a complete episode for task a78176bb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a78176bb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
