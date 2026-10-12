"""과제마다 팀별 code-golf 풀이를 모으고, 각각 실제로 맞는지 확인한다.

집계 저장소는 과제당 채택본 하나만 싣는다. 원본 팀 저장소를 각각 받으면
같은 과제에 대해 서로 다른 사람이 쓴 풀이가 여러 개 생긴다 -- 그게 다양성
재료로서 집계본보다 나은 이유다.

맞는지는 추측하지 않는다. ARC 원본의 train+test 쌍 전부에 돌려서 재현되는
것만 센다.
"""
import json, importlib.util, signal, hashlib
from pathlib import Path

ARC = Path("/hdd_data/yunho/ARC-AGI/data/training")
T = Path("/hdd_data/yunho/golfteams")
SRC = {
    "aggregate": Path("/hdd_data/yunho/arc-golf2/solutions/task{n:03d}.py"),
    "cgi":       T / "cgi/solutions/task{n:03d}.py",
    "oxjam":     T / "oxjam/submission/task{n:03d}.py",
    "itertools": T / "itertools/solutions/task{n:03d}.py",
    "camjam":    T / "camjam/solutions/task{n:03d}.py",
    "lv1dev":    T / "lv1dev/sol/{n:03d}.py",
    "cristianoc": Path("/hdd_data/yunho/arc-golf/task{n:03d}.py"),
    "cctofu":    T / "golf-championship/code/task{n:03d}.py",
    "sakana":    T / "google-code-golf-2025/solutions/task{n:03d}.py",
}
NUM = json.load(open("/hdd_data/yunho/cctmp/golfmap2.json"))["map"]   # tid -> n

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

def check(path, pairs):
    try:
        spec = importlib.util.spec_from_file_location("sol_" + hashlib.md5(
            str(path).encode()).hexdigest()[:8], path)
        m = importlib.util.module_from_spec(spec)
        signal.alarm(10); spec.loader.exec_module(m); signal.alarm(0)
        fn = m.p
        for I, O in pairs:
            signal.alarm(10)
            got = fn([r[:] for r in I])
            signal.alarm(0)
            if [list(map(int, r)) for r in got] != O:
                return False
        return True
    except Exception:
        signal.alarm(0)
        return False
    finally:
        signal.alarm(0)

out = {}
for tid, n in sorted(NUM.items()):
    d = json.loads((ARC / f"{tid}.json").read_text())
    pairs = [(p["input"], p["output"]) for p in d["train"] + d["test"]]
    got = {}
    seen = set()
    for team, tmpl in SRC.items():
        p = Path(str(tmpl).format(n=n))
        if not p.is_file():
            continue
        # 바이트로 읽는다. 바이트를 깎으려고 UTF-8이 아닌 바이트열을 쓴
        # 파일이 있어서, 텍스트로 읽으면 거기서 죽는다.
        h = hashlib.md5(p.read_bytes().strip()).hexdigest()
        if h in seen:
            got[team] = "dup"
            continue
        if check(p, pairs):
            got[team] = "ok"
            seen.add(h)
        else:
            got[team] = "bad"
    out[tid] = got

json.dump(out, open("/hdd_data/yunho/cctmp/golfteams9.json", "w"))
import collections
per = collections.Counter(sum(1 for v in g.values() if v == "ok") for g in out.values())
print("과제당 '맞는 서로 다른 풀이' 개수 분포")
for k in sorted(per): print(f"  {k}개: {per[k]:4d} 과제")
print(f"\n총 유효 풀이 {sum(sum(1 for v in g.values() if v=='ok') for g in out.values())}")
byteam = collections.Counter()
for g in out.values():
    for team, v in g.items():
        if v == "ok": byteam[team] += 1
print("팀별 유효:", dict(byteam.most_common()))
