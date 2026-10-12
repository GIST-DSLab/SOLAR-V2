"""만든 mech maker가 (1) 답에 닿는가 (2) 정말 새 경로인가.

mechroutes.py는 변환을 메모리 안에서 적용해 봤을 뿐이다. 파일로 굳힌 뒤에도
같은 결과가 나오는지는 따로 봐야 한다 -- 인자 수, 이름 해석, 셀 좌표 형식이
거기서 틀어질 수 있다.
"""
import sys, os, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
sys.path.insert(0, "/hdd_data/yunho/cctmp")
import isolate
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
ARMS = ["mech_split", "mech_merge"]
OUT = Path("/hdd_data/yunho/cctmp/verify_mech.json")
N = 5
CHUNK = int(os.environ.get("SOLAR_CHUNK", "40"))

def sig(derive, pairs, eps):
    out = []
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if g is None or g.shape != O.shape or not bool((g == O).all()):
            return None
        steps = []
        for op, sel in zip(ops, sels):
            m = cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)
            rr, cc = np.where(m)
            steps.append((cm.NAME.get(int(op)), frozenset(zip(rr.tolist(), cc.tolist()))))
        out.append(tuple(steps))
    return tuple(out)

def others(t):
    return [d.name for d in sorted(ROOT.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and d.name not in ARMS and (d / t / "grid_maker.py").exists()]

want = sorted({(a, p.name) for a in ARMS if (ROOT / a).is_dir()
               for p in (ROOT / a).iterdir() if p.is_dir()})
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {(r["arm"], r["task"]) for r in rows}
todo = [x for x in want if x not in done][:CHUNK]
if not todo:
    print("done"); raise SystemExit

def measure(arm, t):
    pairs = cm.draw(t, N, [0])
    if not pairs:
        return {"arm": arm, "task": t, "skip": True}
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    d = cm.load_derive(ROOT / arm / t / "grid_maker.py")
    if d is None:
        return {"arm": arm, "task": t, "ok": False, "why": "load"}
    s = sig(d, pairs, eps)
    if s is None:
        return {"arm": arm, "task": t, "ok": False, "why": "replay"}
    old = set()
    for a in others(t):
        dd = cm.load_derive(ROOT / a / t / "grid_maker.py")
        if dd is None: continue
        ss = sig(dd, pairs, eps)
        if ss: old.add(ss)
    return {"arm": arm, "task": t, "ok": True, "new": s not in old, "old": len(old)}

for arm, t in todo:
    got, why = isolate.run(measure, arm, t, gib=3.0, seconds=600)
    rows.append(got if got else {"arm": arm, "task": t, "ok": False, "why": why})
    tmp = OUT.with_suffix(".part"); tmp.write_text(json.dumps(rows)); tmp.replace(OUT)
print(f"{len(rows)}/{len(want)}")
