#!/usr/bin/env python3
"""Can the published episode be solved from its own three demonstrations?

Every other check in this repository asks whether the maker reaches the answer.
It is handed I and O and judged on the grid it produces. Nobody has asked the
question a reader of the dataset actually faces: you are shown three worked
pairs and one test input, and nothing else. If the three pairs do not pin the
rule -- a role colour that never appears in them, a case the test exercises and
they do not -- the episode is unanswerable however clean its trajectory is.

So: show a solver the episode's three demonstrations and its test input, ask
for the test output, and compare. No operations, no ARCLE. Whether the route
can be expressed is a different question and would confound this one.

A solver that fails proves nothing on its own -- the task may simply be hard.
Two controls run, and only where ours failed, so they cost nothing elsewhere.

The first is other episodes of the same task. If the rule is unreadable from
one draw's three demonstrations and readable from another's, the fault is in
that draw, which is the finding this probe exists for. If every draw fails,
the draw is not what is wrong.

The second is the ORIGINAL ARC task's own demonstrations. It is the weaker of
the two and is recorded rather than relied on: ARC-AGI-1's training set is
published, so a solver may be reciting it rather than reading it. It is still
worth having, because a task the solver cannot do even from the original is
one this probe can say nothing about.

    python pipeline/probe_readable.py --draw $SOLAR_DATA_ROOT/ARC_rearc_draw19 \
        --out /path/to/readable.json --parallel 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import datetime
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PAD = 10
ARC_DIR = Path(os.environ.get("SOLAR_ARC_DIR",
                              "/hdd_data/yunho/ARC-AGI/data/training"))
LIMIT = threading.Event()
PRINT = threading.Lock()
# A spent budget does not always say so. Overnight the CLI began returning
# something this probe could not parse, for every call, and the keyword guard
# below did not fire: a hundred "answers" came back in one minute and were
# recorded as a solver that could not read a single ARC task. Matching the
# message was the wrong instrument, because it assumes the message. These two
# do not.
MIN_SECONDS = 2.0      # a real completion does not return this fast
DEAD_RUN = 12          # consecutive unparseable replies that end the run
DEAD = threading.Semaphore(0)
_dead_lock = threading.Lock()
_dead = 0
_kept = 0
RAW = None             # where to keep the first few unparseable replies
DEADLINE = None        # local time after which no new task is started

SYSTEM = (
    "You are solving an ARC puzzle. You are shown several worked example pairs "
    "(input grid -> output grid) that all follow ONE hidden rule, and then a "
    "test input.\n\n"
    "Infer the rule from the examples and apply it to the test input.\n\n"
    "Reply with ONLY the test output grid, as rows of space-separated digits "
    "inside a single ``` fenced block. No explanation, no commentary, nothing "
    "before or after the block. Colours are digits 0-9. The output may be a "
    "different size from the input; give whatever size the rule produces."
)


def grid_str(g) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def unpad(flat, dim):
    h, w = dim
    return [[c for c in row[:w] if c != PAD] for row in flat[:h]]


def user_prompt(pairs, test_in) -> str:
    parts = []
    for k, (i, o) in enumerate(pairs):
        parts += [f"Example {k + 1} input ({len(i)}x{len(i[0])}):", grid_str(i),
                  f"Example {k + 1} output ({len(o)}x{len(o[0])}):", grid_str(o), ""]
    parts += [f"Test input ({len(test_in)}x{len(test_in[0])}):", grid_str(test_in),
              "", "Give the test output grid."]
    return "\n".join(parts)


FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)


def save(out: Path, recs) -> None:
    """Write the results so an interrupted run cannot leave half a file.

    The run is stopped on purpose -- at a deadline, at a budget -- and it is
    written after every single task, so the two will coincide. A plain write
    that is cut in the middle leaves JSON that will not parse, and the next
    run starts from nothing.
    """
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_text(json.dumps(sorted(recs, key=lambda r: r["task_id"]), indent=1))
    tmp.replace(out)


def note(why: str, code, out: str, err: str) -> None:
    """Keep the first few replies that could not be read, and say so once.

    The overnight run recorded four hundred unreadable answers and kept none
    of them, so there was nothing to look at afterwards -- only the shape of
    the damage. A handful of raw replies is all it takes to tell a spent
    budget from a solver that started writing prose.
    """
    global _kept
    with _dead_lock:
        if _kept >= 5:
            return
        _kept += 1
    with PRINT:
        print(f"  [{why}] exit={code} out={out[:160]!r} err={err[:160]!r}",
              flush=True)
    if RAW:
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(f"--- {why} exit={code}\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")


def saw(parsed) -> None:
    """Count unreadable replies in a row and stop the run at a run of them.

    Any single one can be the solver being chatty. A dozen in a row is the
    tool, not the task, and every call after it would be scored as a miss the
    solver never made.
    """
    global _dead
    with _dead_lock:
        if parsed is not None:
            _dead = 0
            return
        _dead += 1
        if _dead >= DEAD_RUN and not LIMIT.is_set():
            LIMIT.set()
            with PRINT:
                print(f"  {_dead} unreadable replies in a row; stopping. "
                      f"Rerun to resume.", flush=True)


def parse_grid(text: str):
    """The grid in the reply, or None. Whatever is in the last fence wins."""
    if not text:
        return None
    blocks = FENCE.findall(text)
    body = blocks[-1] if blocks else text
    rows = []
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.replace(",", " ").split()
        if not all(t.isdigit() and len(t) == 1 for t in toks):
            return None
        rows.append([int(t) for t in toks])
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return rows


def past_deadline() -> bool:
    return DEADLINE is not None and datetime.datetime.now() >= DEADLINE


def ask(user: str, model: str, timeout: int) -> str | None:
    if LIMIT.is_set() or past_deadline():
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as sf:
        sf.write(SYSTEM)
        sys_file = sf.name
    cmd = [shutil.which("claude") or "claude", "-p",
           "--system-prompt-file", sys_file,
           "--model", model,
           "--output-format", "text",
           "--dangerously-skip-permissions"]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout or "").strip()
        if time.monotonic() - t0 < MIN_SECONDS:
            # Too fast to have been generated. Whatever came back, it is not an
            # answer, and scoring it as one is how the overnight run turned a
            # dead CLI into four hundred findings.
            note("too fast", p.returncode, out, (p.stderr or ""))
            LIMIT.set()
            return None
        err = (p.stderr or "").strip()
        # A spent budget has to stop the run, and it does not announce itself
        # in one place: the message has turned up on stdout with a zero exit
        # status and on stderr with a non-zero one. Whichever way it arrives,
        # every later call would be scored as an answer the solver never gave
        # -- a run of silent wrong data, which is worse than no run. So: the
        # message anywhere, or an empty reply with a non-zero exit, stops it.
        blob = (out + "\n" + err).lower()
        if any(k in blob for k in ("usage limit", "rate limit", "quota",
                                   "limit reached", "upgrade to")):
            LIMIT.set()
            return None
        if p.returncode != 0 and not out:
            note("empty, nonzero exit", p.returncode, out, err)
            LIMIT.set()
            return None
        return out
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(sys_file)


def episode(draw: Path, tid: str, which: int):
    """One published episode: its three demonstrations and its test pair."""
    hits = sorted(draw.glob(f"whole/test.{tid}.*"))
    if not hits:
        return None
    files = sorted(hits[0].glob("*.json"), key=lambda p: int(p.stem))
    if not files:
        return None
    e = json.loads(files[which % len(files)].read_text())
    pairs = [(unpad(g, d), unpad(go, do)) for g, d, go, do in
             zip(e["ex_in"], e["ex_in_grid_dim"], e["ex_out"], e["ex_out_grid_dim"])]
    return pairs, unpad(e["in_grid"], e["grid_dim"][0]), unpad(e["out_grid"], e["grid_dim"][-1])


def original(tid: str):
    """The ARC task's own train pairs and its first test pair."""
    f = ARC_DIR / f"{tid}.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    pairs = [(p["input"], p["output"]) for p in d["train"]]
    t = d["test"][0]
    return pairs, t["input"], t["output"]


NUDGE = ("\n\nReply with the grid only, inside one ``` block. "
         "No sentences, no reasoning, no label.")


def judge(pairs, ti, to, model: str, timeout: int) -> str:
    """pass / fail / unparsed for one (demonstrations, test input) question.

    An unparseable reply is asked once more with the format restated. Five of
    the first hundred and fifty came back as prose, and scoring those as
    misses would put the solver's habits into a number that is supposed to be
    about the episodes.
    """
    u = user_prompt(pairs, ti)
    got = parse_grid(ask(u, model, timeout))
    if got is None and not LIMIT.is_set():
        got = parse_grid(ask(u + NUDGE, model, timeout))
    saw(got)
    return "pass" if got == to else ("unparsed" if got is None else "fail")


def score(draw: Path, tid: str, which: int, model: str, timeout: int) -> str:
    made = episode(draw, tid, which)
    if not made:
        return "no episode"
    pairs, ti, to = made
    return judge(pairs, ti, to, model, timeout)


def run_one(tid: str, draw: Path, which: int, model: str, timeout: int,
            control: bool, retries: int, paired: bool,
            originals_only: bool = False) -> dict:
    """One task: ours, and the controls the design calls for.

    `paired` scores the original on every task rather than only where ours
    failed. Scoring it only on failures cannot say what the original's own
    rate is, and without that there is nothing to compare ours against -- the
    first run could report thirteen tasks where ours failed and the original
    passed without anyone knowing the solver misses that many ARC tasks
    anyway. Naming individual tasks needs more repeats than the budget allows;
    a paired rate over the same tasks, same solver, does not.
    """
    rec = {"task_id": tid}
    if LIMIT.is_set() or past_deadline():
        return rec
    if not originals_only:
        rec["ours"] = score(draw, tid, which, model, timeout)
    orig = original(tid)
    if originals_only or paired or (control and rec["ours"] not in ("pass", "no episode")):
        if not orig:
            rec["original"] = "not found"
        else:
            opairs, oi, oo = orig
            rec["original"] = judge(opairs, oi, oo, model, timeout)
    if originals_only or rec["ours"] in ("pass", "no episode") or not control:
        return rec
    rec["others"] = [score(draw, tid, which + 1 + k, model, timeout)
                     for k in range(retries)]
    return rec


WEEK_RE = re.compile(r"week \(all models\):\s*(\d+)%", re.I)


def weekly_pct(timeout: int = 120) -> int | None:
    """The subscription's weekly figure, or None if it could not be read.

    A run of several hundred calls left overnight has to be able to stop
    itself. None is not zero: if the figure cannot be read the gate does not
    fire, because guessing a number here would either halt a run that had
    budget or spend one that did not.
    """
    try:
        p = subprocess.run([shutil.which("claude") or "claude", "-p", "/usage"],
                           capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    m = WEEK_RE.search(p.stdout or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=0,
                    help="score this many tasks chosen at random instead of all")
    ap.add_argument("--which", type=int, default=0,
                    help="which episode of each task to score")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--until", default=None, metavar="HHMM",
                    help="stop taking new tasks at this local time. The work "
                         "belongs to the night; the daytime budget is the "
                         "user's")
    ap.add_argument("--raw", default=None,
                    help="append the first few unreadable replies here")
    ap.add_argument("--budget", type=int, default=0,
                    help="stop once the subscription's weekly figure reaches "
                         "this percent; 0 leaves the run ungated")
    ap.add_argument("--budget_every", type=int, default=25,
                    help="tasks between budget checks (each check is itself a "
                         "call, so this is not free)")
    ap.add_argument("--no_control", action="store_true",
                    help="skip both controls on the ones that failed")
    ap.add_argument("--originals_only", action="store_true",
                    help="score only the original ARC tasks, into their own "
                         "file. The comparison needs the original's rate over "
                         "a sample of the same population; it does not need "
                         "the original of every task, and each one is a call")
    ap.add_argument("--paired", action="store_true",
                    help="score the original ARC task on every task, not only "
                         "where ours failed, so the two rates can be compared")
    ap.add_argument("--retries", type=int, default=2,
                    help="further episodes of the same task to score when the "
                         "first one failed")
    args = ap.parse_args()

    global RAW, DEADLINE
    RAW = args.raw
    if args.until:
        now = datetime.datetime.now()
        hh, mm = int(args.until[:2]), int(args.until[2:])
        stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop <= now:
            stop += datetime.timedelta(days=1)
        DEADLINE = stop
        print(f"stopping at {stop:%m-%d %H:%M}", flush=True)
    draw = Path(args.draw).expanduser().resolve()
    tasks = args.tasks or sorted(p.name.split(".")[1] for p in (draw / "whole").iterdir()
                                 if p.is_dir())
    if args.sample:
        tasks = sorted(random.Random(0).sample(tasks, min(args.sample, len(tasks))))
    out = Path(args.out).expanduser()
    # Resumable: a run of several hundred LLM calls will be interrupted, and
    # re-asking a task already answered spends budget to learn nothing.
    done = {r["task_id"]: r for r in json.loads(out.read_text())} if out.is_file() else {}
    if args.originals_only:
        done = {k: v for k, v in done.items() if "original" in v}
    todo = [t for t in tasks if t not in done]
    print(f"{len(tasks)} tasks, {len(done)} already scored, {len(todo)} to go",
          flush=True)

    if args.budget:
        w = weekly_pct()
        print(f"weekly {w}% (gate {args.budget}%)", flush=True)
        if w is not None and w >= args.budget:
            print("already at the gate; nothing run", flush=True)
            return

    recs = list(done.values())
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_one, t, draw, args.which, args.model, args.timeout,
                          not args.no_control, args.retries, args.paired,
                          args.originals_only): t for t in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if "ours" not in rec and "original" not in rec:
                continue          # 창이 닫혀 시작도 못 한 것: 기록하지 않는다
            recs.append(rec)
            with PRINT:
                print(f"[{n}/{len(todo)}] {rec['task_id']} ours={rec.get('ours', '-')}"
                      + (f" others={','.join(rec['others'])}" if "others" in rec else "")
                      + (f" original={rec['original']}" if "original" in rec else ""),
                      flush=True)
                save(out, recs)
            if args.budget and n % args.budget_every == 0 and not LIMIT.is_set():
                w = weekly_pct()
                if w is not None and w >= args.budget:
                    LIMIT.set()
                    with PRINT:
                        print(f"weekly {w}% reached the {args.budget}% gate; "
                              f"stopping. Rerun to resume.", flush=True)
    save(out, recs)
    if LIMIT.is_set():
        print("\nstopped early; rerun to resume", file=sys.stderr)
    if past_deadline():
        print("\nstopped at the deadline; rerun to resume", file=sys.stderr)

    report(recs)


def report(recs) -> None:
    """What the run can and cannot say.

    The paired rate is the finding. Per-task verdicts are printed after it and
    hedged, because three episodes cannot separate an unreadable draw from a
    solver that misses the task one time in three: the first run's thirteen
    "only this draw failed" tasks were six of that second kind.
    """
    recs = [r for r in recs if "ours" in r or "original" in r]

    def rate(key):
        v = [r[key] for r in recs if key in r and r[key] in ("pass", "fail", "unparsed")]
        return sum(x == "pass" for x in v), len(v)

    recs_ours = [r for r in recs if "ours" in r]
    o_p, o_n = rate("ours")
    c_p, c_n = rate("original")
    print(f"\n생성본  {o_p}/{o_n}" + (f"  ({o_p / o_n:.0%})" if o_n else ""))
    if c_n:
        print(f"원본    {c_p}/{c_n}  ({c_p / c_n:.0%})"
              "   <- 공개 학습셋이라 외웠을 수 있음: 상한으로만 읽을 것")
    both = [r for r in recs if "original" in r and "ours" in r]
    _ = recs_ours
    if both:
        only_ours = sum(1 for r in both if r["ours"] != "pass" and r["original"] == "pass")
        only_orig = sum(1 for r in both if r["ours"] == "pass" and r["original"] != "pass")
        print(f"짝지어 본 {len(both)}개 중  우리만 실패 {only_ours}, 원본만 실패 {only_orig}")
    tail = [r for r in recs_ours if r["ours"] != "pass" and r.get("others")]
    clear = [r["task_id"] for r in tail if r["others"].count("pass") == len(r["others"])]
    noisy = [r["task_id"] for r in tail if 0 < r["others"].count("pass") < len(r["others"])]
    never = [r["task_id"] for r in tail if r["others"].count("pass") == 0]
    print(f"\n그 드로우만 실패, 다른 두 에피소드는 모두 통과 {len(clear)}: "
          + " ".join(clear[:30]))
    print(f"세 에피소드 중 일부만 통과 {len(noisy)} (solver 잡음과 구분 안 됨): "
          + " ".join(noisy[:30]))
    print(f"세 에피소드 모두 실패 {len(never)}: " + " ".join(never[:30]))


if __name__ == "__main__":
    main()
