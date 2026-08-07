#!/usr/bin/env python3
"""Pack generated ARCLE trajectories into HF-ready parquet shards.

The rollouts land on disk as one JSON per trajectory with every grid padded to
30x30 and serialized as int64 lists. That is ~112 KB per trajectory and 1.4 GB
for the ARC-AGI-1 draw, which is not a shape anyone wants to download. Every
value fits in a byte, so each array becomes a raw uint8 blob plus its shape, and
the shards come out ~50x smaller with no loss: `--verify` round-trips rows back
to the original JSON and compares element by element.

Provenance the records did not carry is injected here: `maker_version` per task
(from best_manifest.json, i.e. which arm the bestof judge picked), the subset
name, and the source folder date. `rand_seed` is deliberately absent — see
release_manifest.json:caveats.

    python export_release.py --out /hdd_data/yunho/release --subsets arc_1d --verify
    python export_release.py --out /hdd_data/yunho/release --verify
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPEC_VERSION = "1.0"
DATA_ROOT = Path("/hdd_data/yunho")
SOLAR_ROOT = Path(__file__).parent.resolve()

# folder name: test.<task_id>.s30.<YY.MM.DD>
FOLDER_RE = re.compile(r"^test\.(?P<task>[^.]+)\.s(?P<dim>\d+)\.(?P<date>[\d.]+)$")

# Arrays are stored as raw bytes of these dtypes. Ranges are asserted per file,
# so a value that outgrows its column aborts the export instead of wrapping.
# uint8 is reserved for the pixel planes (colors 0-9, pad 10, masks 0/1) which
# carry essentially all the bytes. Everything else is int16: object_pos goes
# negative once an object is dragged off the top-left edge, and the small
# columns are too cheap for the extra byte to matter.
COL_DTYPE = {
    "grid": "u1", "clip": "u1", "selection_mask": "u1", "selected": "u1",
    "object": "u1", "object_sel": "u1", "background": "u1",
    "in_grid": "u1", "out_grid": "u1", "ex_in": "u1", "ex_out": "u1",
    "grid_dim": "i2", "clip_dim": "i2", "ex_in_grid_dim": "i2",
    "ex_out_grid_dim": "i2", "object_dim": "i2", "object_pos": "i2",
    "selection": "i2", "operation": "i2", "object_active": "i2", "reward": "i2",
    "step": "i2", "terminated": "b1",
}
# kept as a native parquet list<string>, not a blob
STR_LIST_COLS = ("operation_name",)

SUBSETS = {
    "arc_agi1": dict(
        # ARC_best10 (the 26.07.27 draw) predates the terminal-filler fix in
        # gen_rearc_trajectories_v2.py and ends every episode in Submit,Submit
        # with states and actions the same length. This is the re-roll.
        root=DATA_ROOT / "ARC_best10_r5" / "whole",
        makers="maker/arc-best",
        episodes=10,
        # what the maker set is called in the release; the working tree keeps its
        # own name, so renaming for publication does not disturb generation
        label="arc-agi-1",
        manifest=DATA_ROOT / "best_manifest.json",
        note="400 ARC-AGI-1 training tasks, one maker per task",
        rollout="python gen_rearc_trajectories_v2.py --subfolder arc-agi-1 "
                "--num_samples 25 --rand_seed 0 --max_grid_dim 30 30 "
                "--data_folder <out>",
    ),
    "handcraft": dict(
        # A control, not new coverage: all 10 base task ids are already in
        # arc_agi1, so hand-written and LLM-written makers meet on the same tasks.
        # The `half` variants only. Rolling out every expert/half variant put ~28%
        # duplicates in the set: across the two, 282 duplicate signature groups
        # spanned different variants, because the variants differ in how many
        # demonstrations they also solve, not in the route to the problem. Within
        # half alone no two variants share a trajectory.
        root=DATA_ROOT / "ARC_handcraft_h15" / "whole",
        makers="maker/arc-handcraft",
        episodes=10,
        # 74dd1130-half is a transpose: FlipV, Rotate90, Submit, the same three
        # actions on all 25 rollouts. There is no route to read off it, which is
        # what this subset is for.
        exclude={"74dd1130-half"},
        label="handcraft",
        manifest=None,
        note="11 hand-written half-variant makers over 10 ARC-AGI-1 training tasks",
        # --force_grid_size is not optional here: these makers unpack
        # max_grid_dim from kwargs and KeyError without it.
        # 15x15, not the 30x30 the release uses. These makers scale their work
        # with the grid: the move-an-object one spent 1828 actions on a 30x30
        # sample. At 15 the worst episode is 174 and the grids still look like
        # ARC tasks, which 10x10 does not — it squeezes several makers into 5x5.
        rollout="python pipeline/gen_rearc_trajectories_v2.py --subfolder handcraft "
                "--num_samples 25 --rand_seed 0 --max_grid_dim 15 15 "
                "--force_grid_size --data_folder <out>",
    ),
    "arc_1d": dict(
        root=DATA_ROOT / "ARC_1d" / "whole",
        makers="maker/arc-1d",
        manifest=None,
        note="18 1D-ARC task families",
        rollout="python gen_rearc_trajectories_v2.py --subfolder arc-1d "
                "--num_samples 10 --max_grid_dim 30 30 --data_folder <out>",
    ),
    "arc_agi2_solve": dict(
        root=DATA_ROOT / "ARC_agi2_pilot" / "solve" / "whole",
        makers="maker/arc-agi2-solve",
        manifest=None,
        note="20 ARC-AGI-2 eval tasks, maker sees I only (no output leakage)",
        rollout="python gen_agi2_llm.py --mode solve",
    ),
    "arc_agi2_construction": dict(
        root=DATA_ROOT / "ARC_agi2_pilot" / "construction" / "whole",
        makers="maker/arc-agi2-construction",
        manifest=None,
        note="20 ARC-AGI-2 eval tasks, maker may consult O",
        rollout="python gen_agi2_llm.py --mode construction",
    ),
}


def repo_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(SOLAR_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def pack(name: str, value, src: Path) -> tuple[bytes, list[int]]:
    """Array -> (raw bytes in the column's dtype, shape). Aborts on overflow."""
    dt = np.dtype(COL_DTYPE[name])
    arr = np.asarray(value)
    if dt.kind in "iu":
        lo, hi = np.iinfo(dt).min, np.iinfo(dt).max
        if arr.size and (arr.min() < lo or arr.max() > hi):
            raise ValueError(
                f"{src}: column '{name}' range [{arr.min()},{arr.max()}] "
                f"does not fit {dt} — widen COL_DTYPE and re-export"
            )
    return arr.astype(dt).tobytes(), list(arr.shape)


def unpack(name: str, blob: bytes, shape: list[int]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.dtype(COL_DTYPE[name])).reshape(shape)


def row_of(path: Path, subset: str, task_id: str, date: str, version: str) -> dict:
    d = json.loads(path.read_text())
    desc = d.get("desc") or {}
    row = {
        "id": desc.get("id", path.stem),
        "task_id": task_id,
        "subset": subset,
        "concept": desc.get("concept", ""),
        "maker_version": version,
        "source_date": date,
        "n_steps": len(d["operation"]),
        "operation_name": list(d["operation_name"]),
    }
    for col in COL_DTYPE:
        blob, shape = pack(col, d[col], path)
        row[col] = blob
        row[f"{col}__shape"] = shape

    # Standard RL alignment: N actions, N+1 states. The 26.07.27 draw shipped
    # N states and a duplicated terminal Submit, which silently mis-pairs (s,a)
    # downstream, so the exporter refuses to pack anything that looks like it.
    n_act = row["n_steps"]
    n_state = row["grid__shape"][0]
    if n_state != n_act + 1:
        raise AssertionError(
            f"{path}: {n_act} actions but {n_state} states (want {n_act + 1}) — "
            "source predates the terminal-filler fix; re-run the rollout"
        )
    if row["operation_name"][-2:] == ["Submit", "Submit"]:
        raise AssertionError(f"{path}: trajectory ends in a duplicated Submit")
    return row


def rebuild(row: dict) -> dict:
    """Inverse of row_of, back to the on-disk JSON structure."""
    out = {
        "desc": {"id": row["id"], "concept": row["concept"]},
        "operation_name": list(row["operation_name"]),
    }
    for col in COL_DTYPE:
        out[col] = unpack(col, row[col], list(row[f"{col}__shape"])).tolist()
    return out


def arrow_schema() -> pa.Schema:
    fields = [
        pa.field("id", pa.string()), pa.field("task_id", pa.string()),
        pa.field("subset", pa.string()), pa.field("concept", pa.string()),
        pa.field("maker_version", pa.string()), pa.field("source_date", pa.string()),
        pa.field("n_steps", pa.int32()),
        pa.field("operation_name", pa.list_(pa.string())),
    ]
    for col in COL_DTYPE:
        fields.append(pa.field(col, pa.binary()))
        fields.append(pa.field(f"{col}__shape", pa.list_(pa.int32())))
    meta = {
        b"spec_version": SPEC_VERSION.encode(),
        b"column_dtypes": json.dumps(COL_DTYPE).encode(),
        b"note": (b"array columns are raw little-endian bytes of the dtype in "
                  b"column_dtypes; reshape with the matching <col>__shape"),
    }
    return pa.schema(fields, metadata=meta)


def export_subset(name: str, cfg: dict, out_root: Path, shard_rows: int,
                  verify: int, maker_version: bool = False) -> dict:
    root: Path = cfg["root"]
    if not root.is_dir():
        print(f"  skip {name}: {root} missing")
        return {}

    # Which generation round a task's maker came from is internal history, so by
    # default every row is stamped with the maker set instead. best_manifest.json
    # keeps the per-task mapping on disk; --maker_version puts it back in.
    versions = {}
    if maker_version and cfg["manifest"] and Path(cfg["manifest"]).is_file():
        versions = json.loads(Path(cfg["manifest"]).read_text())

    jobs = []
    for folder in sorted(root.iterdir()):
        m = FOLDER_RE.match(folder.name)
        if not m:
            print(f"  skip unparseable folder: {folder.name}")
            continue
        task, date = m["task"], m["date"]
        if task in cfg.get("exclude", ()):
            continue
        version = versions.get(task, cfg.get("label") or Path(cfg["makers"]).name)
        files = sorted(folder.glob("*.json"))
        # Every task carries the same number of episodes, so a consumer can address
        # a row by task index * episodes + episode — which the web viewer does,
        # because the Hub's /filter endpoint is unreliable. A short task silently
        # shifts every task after it onto the wrong data, so refuse to pack it.
        # Roll out with more samples than this and the surplus is dropped here.
        want = cfg.get("episodes")
        if want:
            if len(files) < want:
                raise AssertionError(
                    f"{folder.name}: {len(files)} episodes, need {want}. Re-run the "
                    f"rollout with a larger --num_samples; do not ship a short task."
                )
            files = files[:want]
        for f in files:
            jobs.append((f, task, date, version))

    out_dir = out_root / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.parquet"):
        stale.unlink()

    schema = arrow_schema()
    n_shards = max(1, (len(jobs) + shard_rows - 1) // shard_rows)
    written, tasks_seen, ver_count, raw_bytes = 0, set(), {}, 0
    checked = 0

    for si in range(n_shards):
        batch = jobs[si * shard_rows:(si + 1) * shard_rows]
        if not batch:
            continue
        rows = []
        for f, task, date, version in batch:
            raw_bytes += f.stat().st_size
            r = row_of(f, name, task, date, version)
            rows.append(r)
            tasks_seen.add(task)
            ver_count[version] = ver_count.get(version, 0) + 1
            if checked < verify:
                original = json.loads(f.read_text())
                back = rebuild(r)
                for k, v in original.items():
                    if k == "desc":
                        continue
                    if back[k] != v:
                        raise AssertionError(f"round-trip mismatch {f} column {k}")
                if set(back) != set(original):
                    raise AssertionError(f"round-trip key mismatch {f}")
                checked += 1
        table = pa.Table.from_pylist(rows, schema=schema)
        path = out_dir / f"train-{si:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table, path, compression="zstd", compression_level=9)
        written += len(rows)
        print(f"  {path.name}: {len(rows)} rows, {path.stat().st_size / 1e6:.1f} MB")

    packed = sum(p.stat().st_size for p in out_dir.glob("*.parquet"))
    print(f"  {name}: {written} traj / {len(tasks_seen)} tasks | "
          f"{raw_bytes/1e6:.0f} MB json -> {packed/1e6:.1f} MB parquet "
          f"({raw_bytes/max(packed,1):.0f}x) | round-trip checked {checked}")
    return {
        "trajectories": written,
        "tasks": len(tasks_seen),
        "maker_versions": dict(sorted(ver_count.items())),
        "maker_set": cfg.get("label") or Path(cfg["makers"]).name,
        "source": str(root),
        "rollout_cmd": cfg.get("rollout", ""),
        "note": cfg["note"],
        "raw_json_bytes": raw_bytes,
        "parquet_bytes": packed,
        "round_trip_checked": checked,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT / "release"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS),
                    choices=list(SUBSETS))
    ap.add_argument("--shard_rows", type=int, default=500)
    ap.add_argument("--verify", type=int, default=0,
                    help="round-trip this many rows per subset against the source JSON")
    ap.add_argument("--maker_version", action="store_true",
                    help="stamp rows with the per-task generation round from "
                         "best_manifest.json instead of the maker set name")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in args.subsets:
        print(f"[{name}]")
        s = export_subset(name, SUBSETS[name], out_root, args.shard_rows,
                          args.verify, maker_version=args.maker_version)
        if s:
            stats[name] = s

    manifest_path = out_root / "release_manifest.json"
    # Carry over subsets exported in an earlier partial run, but only those whose
    # shards are still in the tree — otherwise a subset pulled out of the release
    # lingers in the manifest describing files nobody can download.
    prev = {}
    if manifest_path.is_file():
        prev = json.loads(manifest_path.read_text()).get("subsets", {})
    prev = {k: v for k, v in prev.items()
            if any((out_root / "data" / k).glob("*.parquet"))}
    prev.update(stats)
    manifest_path.write_text(json.dumps({
        "spec_version": SPEC_VERSION,
        "repo_commit": repo_commit(),
        "generator": "SOLAR-Generator/gen_rearc_trajectories_v2.py",
        "exporter": "SOLAR-Generator/export_release.py",
        "column_dtypes": COL_DTYPE,
        "subsets": prev,
        "caveats": [
            "Rollouts whose final grid did not match the target are dropped at "
            "generation time. Subsets with an `episodes` count are rolled out "
            "with a surplus and packed at exactly that many per task, so the "
            "row order is task index * episodes + episode.",
            "The per-sample RNG seed was not recorded at generation time. These "
            "shards are the reference draw; re-running the makers yields a "
            "different, equally valid draw.",
            "Grids are padded to 30x30 with the value 10; the real extent of "
            "step i is grid_dim[i].",
        ],
    }, indent=2))
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
