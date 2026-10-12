"""분할·병합으로 만든 경로가 '새 경로'로 세어지는가.

앞서 변환이 유효한지(재생해서 답에 닿는지)만 봤다. 그것으로 충분하지 않다 --
LLM이 이미 낸 경로와 같으면 다양성에 보태는 게 없다. 그래서 변환 결과를 그
과제의 기존 경로들과 같은 잣대로 비교한다.

순열은 뺀다. 297개 과제에서 평균 61개씩 나오지만 읽는 사람에게는 같은
풀이이고, 수치만 부풀린다.
"""
import sys, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
OUT = Path("/hdd_data/yunho/cctmp/mechroutes.json")
N = 5
MERGEABLE = {f"Color{i}" for i in range(10)} | {f"FloodFill{i}" for i in range(10)}

def cells(sel):
    m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
    rr, cc = np.where(m)
    return frozenset(zip(rr.tolist(), cc.tolist()))

def blobs(cs):
    todo, out = set(cs), []
    while todo:
        seed = todo.pop(); comp = {seed}; stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in todo:
                        todo.discard(p); comp.add(p); stack.append(p)
        out.append(sorted(comp))
    return out

def steps_of(ops, sels):
    return tuple((cm.NAME.get(int(o)), cells(s)) for o, s in zip(ops, sels))

def transform(ops, sels, kind):
    n = len(ops)
    if kind == "분할":
        for i in range(n - 1):
            bl = blobs(cells(sels[i]))
            if len(bl) > 1:
                return (list(ops[:i]) + [ops[i]] * len(bl) + list(ops[i+1:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in b]} for b in bl]
                        + list(sels[i+1:]))
    else:
        for i in range(n - 2):
            if ops[i] == ops[i+1] and cm.NAME.get(int(ops[i])) in MERGEABLE:
                u = sorted(cells(sels[i]) | cells(sels[i+1]))
                return (list(ops[:i]) + [ops[i]] + list(ops[i+2:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in u]}]
                        + list(sels[i+2:]))
    return None

def arms_for(t):
    return [d.name for d in sorted(ROOT.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and (d / t / "grid_maker.py").exists()]

tasks = sorted(p.name for p in (ROOT / "arc-agi-1").iterdir()
               if p.is_dir() and p.name != "__pycache__")
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {r["task"] for r in rows}
todo = [t for t in tasks if t not in done]
print(f"{len(todo)} to go", flush=True)

for k, t in enumerate(todo, 1):
    pairs = cm.draw(t, N, [0])
    if not pairs:
        rows.append({"task": t, "new": {}}); continue
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    # 기존 경로들
    known = set()
    for a in arms_for(t):
        d = cm.load_derive(ROOT / a / t / "grid_maker.py")
        if d is None:
            continue
        sig = []
        for i, (I, O) in enumerate(pairs):
            shown = eps[i % len(eps)][0] if eps else None
            g, o, s = cm.replay(d, I, O, shown)
            if g is None or g.shape != O.shape or not bool((g == O).all()):
                sig = None; break
            sig.append(steps_of(o, s))
        if sig:
            known.add(tuple(sig))
    base = cm.load_derive(ROOT / "arc-agi-1" / t / "grid_maker.py")
    new = {}
    for kind in ("분할", "병합"):
        sig, ok = [], True
        for i, (I, O) in enumerate(pairs):
            shown = eps[i % len(eps)][0] if eps else None
            g, o, s = cm.replay(base, I, O, shown)
            if o is None: ok = False; break
            tr = transform(o, s, kind)
            if tr is None: ok = False; break
            o2, s2 = tr
            g2 = cm._run(I, O, o2, s2)
            if g2 is None or g2.shape != O.shape or not bool((g2 == O).all()):
                ok = False; break
            sig.append(steps_of(o2, s2))
        new[kind] = bool(ok and tuple(sig) not in known)
    rows.append({"task": t, "new": new})
    tmp = OUT.with_suffix(".part"); tmp.write_text(json.dumps(rows)); tmp.replace(OUT)
    if k % 50 == 0: print(f"[{k}/{len(todo)}]", flush=True)

c = collections.Counter()
for r in rows:
    for kind, v in r["new"].items():
        if v: c[kind] += 1
both = sum(1 for r in rows if any(r["new"].values()))
print(f"\n분할이 새 경로를 내는 과제 {c['분할']}")
print(f"병합이 새 경로를 내는 과제 {c['병합']}")
print(f"둘 중 하나라도 새 경로 {both}/{len(rows)}")
