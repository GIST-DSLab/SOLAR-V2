"""같은 과제의 서로 다른 풀이 6개를 각각 따로 보여준다.

묻는 것: 사람 여섯이 각자 푼 것이 ARCLE 궤적 여섯 가지로 이어지는가.
그래서 한 arm에는 한 팀의 풀이만 넣는다. 여섯을 한꺼번에 보여주면 모델이
그중 하나로 수렴할 수 있고, 그러면 무엇이 무엇을 낳았는지 가릴 수 없다.
"""
import sys, json, random
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker/arc-agi-1")
T = Path("/hdd_data/yunho/golfteams")
SRC = {
    "cristianoc": Path("/hdd_data/yunho/arc-golf/task{n:03d}.py"),
    "cctofu":    T / "golf-championship/code/task{n:03d}.py",
    "sakana":    T / "google-code-golf-2025/solutions/task{n:03d}.py",
    "aggregate": Path("/hdd_data/yunho/arc-golf2/solutions/task{n:03d}.py"),
    "cgi":       T / "cgi/solutions/task{n:03d}.py",
    "oxjam":     T / "oxjam/submission/task{n:03d}.py",
    "itertools": T / "itertools/solutions/task{n:03d}.py",
    "camjam":    T / "camjam/solutions/task{n:03d}.py",
    "lv1dev":    T / "lv1dev/sol/{n:03d}.py",
}
NUM = json.load(open("/hdd_data/yunho/cctmp/golfmap2.json"))["map"]
VALID = json.load(open("/hdd_data/yunho/cctmp/golfteams9.json"))

TAIL = """

That is how one code-golf team solved the same task. It is written to be short in bytes, not to be read, so its shape is not the point -- what is worth taking is the reading of the task underneath it.
Where that reading carves the problem differently from the maker above -- a different thing treated as one unit, a different order, a different quantity held fixed -- write the operations that reading implies. Where the two agree there is nothing to change and you should say so.
Do not try to make your route short. The reference is short because bytes were the objective there; here they are not. A longer route that follows a different reading is the wanted answer.
Two limits hold regardless.
Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations."""

# 400개 전부. 그 팀 풀이가 검증된 과제만 담기므로 팀마다 개수가 다르다.
pick = sorted(t for t in VALID if (ROOT / t / "grid_maker.py").exists())

made = {}
for team in ["cristianoc", "sakana", "cctofu"]:   # 기존 6팀은 이미 생성됨
    out = {}
    for t in pick:
        if VALID[t].get(team) != "ok":
            continue
        p = Path(str(SRC[team]).format(n=NUM[t]))
        ref = p.read_bytes().decode("utf-8", "replace").strip()
        if len(ref) > 6000:
            ref = ref[:6000] + "\n# (truncated)"
        src = (ROOT / t / "grid_maker.py").read_text()
        i = src.find("def derive_operations"); j = src.find("\nclass GridMaker", i)
        code = src[i:j if j > 0 else None].rstrip()
        out[t] = {"feedback":
            "Task-specific quality requirement: this task's maker already solves "
            "it. Its derive_operations is:\n\n```python\n" + code + "\n```\n\n"
            "Here is an entry from a code-golf competition that solves the same "
            "task in as few bytes of Python as its author could manage:\n\n"
            "```python\n" + ref + "\n```" + TAIL}
    json.dump(out, open(f"/hdd_data/yunho/cctmp/cg_{team}_feedback.json", "w"), indent=1)
    json.dump(sorted(out), open(f"/hdd_data/yunho/cctmp/cg_{team}.json", "w"))
    made[team] = len(out)

print("과제", len(pick))
print("arm별 프롬프트 수:", made, " 합계", sum(made.values()))
