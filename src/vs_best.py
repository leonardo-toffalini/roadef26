"""Compare local search load vectors with the published best solutions.

Downloads the qualification and sprint ``loads_vector.csv`` files from
https://gitlab.com/Orange-OpenSource/network-optimization-tools/challenge-roadef-2026
Each row is one instance: the team with the best solution, then every link
utilization sorted worst-first. That order is the lex order used for ranking.

The qualification CSV labels its rows setA-01 .. setA-12. Those vectors are the
setX instances: the qualification readme calls them setX, and each vector has
one entry per arc and time slot, which matches setX and not setA. Rows are
renamed setX before looking for ``data/setX``.

Local search solutions are ``out/ls/<name>-srpaths.json``. A missing file is
computed with ``ls.search`` before the comparison. The checker produces the
same sorted vector. A solution is better when its vector is smaller at the
first position where the two differ.
"""

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LS_DIR = ROOT / "out" / "ls"
CHECKER = ROOT / "check"
CACHE = ROOT / "results"
REPO = "https://gitlab.com/Orange-OpenSource/network-optimization-tools/challenge-roadef-2026/-/raw/main"
# the CSVs drop everything past 6 decimal places
PLACES = 1_000_000

PHASES = {
    "sprint": "sprint_results/loads_vector.csv",
    "qualification": "qualification_results/loads_vector.csv",
}


def published_name(phase: str, raw: str) -> str:
    """Qualification rows are setX, stored in the CSV under setA names."""
    if phase == "qualification" and raw.startswith("setA-"):
        return "setX-" + raw.removeprefix("setA-")
    return raw


def fetch(phase: str, refresh: bool) -> Path:
    path = CACHE / phase / "loads_vector.csv"
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO}/{PHASES[phase]}"
    print(f"downloading {url}", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_vectors(path: Path, phase: str) -> list[tuple[str, str, list[float]]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            loads = []
            index = 1
            while True:
                cell = row.get(str(index))
                if cell is None or cell == "":
                    break
                loads.append(float(cell))
                index += 1
            rows.append((published_name(phase, row["Instance"]), row["Best team"], loads))
    return rows


def located(name: str) -> str:
    """setA-01 lives at data/setA/setA-01."""
    return f"{name.split('-', 1)[0]}/{name}"


def arc_slots(instance: str) -> int | None:
    base = DATA / located(instance)
    net_path, tm_path = Path(f"{base}-net.json"), Path(f"{base}-tm.json")
    if not net_path.exists() or not tm_path.exists():
        return None
    links = json.loads(net_path.read_text())["links"]
    demands = json.loads(tm_path.read_text())["demands"]
    return len(links) * len(demands[0]["v"])


def solve(instance: str, path: Path) -> None:
    """Run local search and write ``path``. Used when no solution is saved yet."""
    print(f"running local search for {instance}", file=sys.stderr, flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    from ls import load, search, write

    n, links, demands, scenario = load(located(instance))
    routes, _, _ = search(n, links, demands, scenario)
    write(routes, path)


def check(instance: str, srpaths: Path) -> dict:
    base = DATA / located(instance)
    proc = subprocess.run(
        [
            str(CHECKER),
            "--net",
            f"{base}-net.json",
            "--tm",
            f"{base}-tm.json",
            "--scenario",
            f"{base}-scenario.json",
            "--srpaths",
            str(srpaths),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise SystemExit(
            f"checker failed for {instance} ({srpaths})\n{proc.stderr}{proc.stdout}"
        )


def load_vector(result: dict) -> list[float] | None:
    if not result.get("valid"):
        return None
    return [entry["sat"] for entry in result["saturations"]]


def trunc6(value: float) -> float:
    """Match the CSV, which keeps 6 digits after the decimal and drops the rest."""
    return int(value * PLACES + 1e-8) / PLACES


def first_diff(local: list[float], best: list[float]) -> tuple[int, float] | None:
    """Index and (local - best) at the first load that differs. None on a tie."""
    for index, (got, want) in enumerate(zip(local, best, strict=True)):
        gap = trunc6(got) - trunc6(want)
        if gap:
            return index, gap
    return None


def headers() -> list[str]:
    return ["phase", "instance", "team", "best MLU", "LS MLU", "first diff", "LS-best", "verdict"]


def cells(
    phase: str,
    instance: str,
    team: str,
    best: list[float],
    local: list[float] | None,
    note: str | None = None,
) -> list[str]:
    row = [phase, instance, team, f"{trunc6(best[0]):.6f}"]
    if note:
        return row + ["-", "-", "-", note]
    assert local is not None
    diff = first_diff(local, best)
    if diff is None:
        verdict, where, gap = "tie", "-", "0"
    else:
        index, gap_value = diff
        verdict = "LS better" if gap_value < 0 else "best better"
        where, gap = str(index + 1), f"{gap_value:+.6f}"
    return row + [f"{trunc6(local[0]):.6f}", where, gap, verdict]


def render(rows: list[list[str]]) -> str:
    numeric = [False, False, False, True, True, True, True, False]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        parts = [
            cell.rjust(width) if num else cell.ljust(width)
            for cell, width, num in zip(row, widths, numeric)
        ]
        lines.append("  ".join(parts).rstrip())
    rule = "-" * len(lines[0])
    return "\n".join([lines[0], rule, *lines[1:]])


def _self_check() -> None:
    assert published_name("qualification", "setA-01") == "setX-01"
    assert published_name("qualification", "setX-01") == "setX-01"
    assert published_name("sprint", "setA-01") == "setA-01"
    assert first_diff([0.5, 0.2], [0.5, 0.2]) is None
    assert first_diff([0.929383899821], [0.929383]) is None
    assert first_diff([0.5, 0.1], [0.5, 0.2]) == (1, -0.1)
    assert first_diff([0.9], [0.5]) == (0, 0.4)
    text = "Instance,Best team,1,2\nsetA-01,S8,0.5,0.25\n"
    path = CACHE / "_self_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    got = load_vectors(path, "sprint")
    path.unlink()
    assert got == [("setA-01", "S8", [0.5, 0.25])]


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instances",
        nargs="*",
        help="instance names to compare (default: every row in both CSVs)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download the CSVs again",
    )
    args = parser.parse_args()
    if not CHECKER.exists():
        raise SystemExit(f"checker not found at {CHECKER}")

    wanted = set(args.instances)
    rows = [headers()]
    tally = {phase: {"LS better": 0, "best better": 0, "tie": 0, "missing": 0} for phase in PHASES}
    for phase in PHASES:
        for instance, team, best in load_vectors(fetch(phase, args.refresh), phase):
            if wanted and instance not in wanted:
                continue
            size = arc_slots(instance)
            solution = LS_DIR / f"{Path(instance).name}-srpaths.json"
            if size is None:
                rows.append(cells(phase, instance, team, best, None, "no instance"))
                tally[phase]["missing"] += 1
                continue
            if size != len(best):
                raise SystemExit(
                    f"{phase} {instance}: vector has {len(best)} loads, "
                    f"instance has {size} arc-slots"
                )
            if not solution.exists():
                solve(instance, solution)
            local = load_vector(check(instance, solution))
            if local is None:
                rows.append(cells(phase, instance, team, best, None, "invalid"))
                tally[phase]["missing"] += 1
                continue
            row = cells(phase, instance, team, best, local)
            rows.append(row)
            tally[phase][row[-1]] += 1

    if len(rows) == 1:
        raise SystemExit("no matching rows in the published CSVs")
    print("Lower loads are better. Compared at 6 decimal places, which is what the CSVs store.")
    print("first diff is the 1-based position in the worst-first vector.")
    print("Qualification CSV rows are renamed setA-* to setX-* (see the module docstring).")
    print()
    print(render(rows))
    print()
    for phase, counts in tally.items():
        print(
            f"{phase}: LS better {counts['LS better']}, "
            f"best better {counts['best better']}, tie {counts['tie']}, "
            f"missing {counts['missing']}"
        )


if __name__ == "__main__":
    main()
