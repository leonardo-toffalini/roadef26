"""Compare checker scores of LP coordinate descent, local search, and closure search.

LP solutions are the ``*-srpaths.json`` files written next to each instance.
Local search solutions live in ``out/ls``, closure search in ``out/ls_closure``.
Every instance with at least one of those files is listed. A solver with no
file for that instance is shown as "-". ``--recompute`` reruns local search.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LS_DIR = ROOT / "out" / "ls"
CL_DIR = ROOT / "out" / "ls_closure"
CHECKER = ROOT / "check"
# label, and where its srpaths file lives; LP is resolved per instance
SOLVERS = ("LP", "LS", "CL")


def solution_names() -> list[str]:
    """Instance tails that have a saved solution from at least one solver."""
    names = set()
    for path in DATA.glob("*/*-srpaths.json"):
        if path.parent.name == "toy":
            continue
        names.add(path.name.removesuffix("-srpaths.json"))
    for folder in (LS_DIR, CL_DIR):
        for path in folder.glob("*-srpaths.json"):
            names.add(path.name.removesuffix("-srpaths.json"))
    return sorted(names)


def located(name: str) -> str:
    """setA-01 lives at data/setA/setA-01."""
    return f"{name.split('-', 1)[0]}/{name}"


def solver_path(solver: str, name: str) -> Path | None:
    if solver == "LP":
        path = DATA / f"{located(name)}-srpaths.json"
    elif solver == "LS":
        path = LS_DIR / f"{name}-srpaths.json"
    else:
        path = CL_DIR / f"{name}-srpaths.json"
    return path if path.exists() else None


def ls_solution(instance: str, recompute: bool) -> Path | None:
    name = Path(instance).name
    path = LS_DIR / f"{name}-srpaths.json"
    if path.exists() and not recompute:
        return path
    if not recompute:
        return None
    print(f"running local search for {instance}", file=sys.stderr, flush=True)
    from ls import load, search, write

    n, links, demands, scenario = load(instance)
    routes, _, _ = search(n, links, demands, scenario)
    write(routes, path)
    return path


def check(instance: str, srpaths: Path) -> dict:
    base = DATA / instance
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


def objectives(result: dict) -> list[dict] | None:
    if not result.get("valid"):
        return None
    return sorted(result["objectives"], key=lambda row: row["t"])


def slots_of(results: list[dict | None]) -> list[int]:
    found = set()
    for result in results:
        got = objectives(result) if result is not None else None
        if got:
            found.update(int(row["t"]) for row in got)
    return sorted(found)


def missing_row(instance: str, src: str, slots: list[int]) -> list[str]:
    return [instance, src, *["-"] * (4 + 3 * len(slots))]


def metric_row(instance: str, src: str, result: dict | None, slots: list[int]) -> list[str]:
    if result is None:
        return missing_row(instance, src, slots)
    cells = [instance, src]
    got = objectives(result)
    if got is None:
        cells += ["no", "-", "-", "-"]
        cells += ["-", "-", "-"] * len(slots)
        return cells
    by_t = {int(row["t"]): row for row in got}
    cells += [
        "yes",
        str(result["total_cost"]),
        str(result["total_srpaths"]),
        str(result["total_segments"]),
    ]
    for t in slots:
        row = by_t.get(t)
        if row is None:
            cells += ["-", "-", "-"]
        else:
            cells += [
                f"{row['mlu']:.6f}",
                f"{row['inv_load_cost']:.4f}",
                f"{row['jain_index']:.4f}",
            ]
    return cells


def delta_row(label: str, left_result: dict | None, right_result: dict | None, slots: list[int]) -> list[str]:
    left = objectives(left_result) if left_result is not None else None
    right = objectives(right_result) if right_result is not None else None
    cells = ["", label, ""]
    if left is None or right is None:
        cells += ["-"] * (3 + 3 * len(slots))
        return cells
    cells += [
        f"{right_result['total_cost'] - left_result['total_cost']:+d}",
        f"{right_result['total_srpaths'] - left_result['total_srpaths']:+d}",
        f"{right_result['total_segments'] - left_result['total_segments']:+d}",
    ]
    a = {int(row["t"]): row for row in left}
    b = {int(row["t"]): row for row in right}
    for t in slots:
        if t not in a or t not in b:
            cells += ["-", "-", "-"]
            continue
        cells += [
            f"{b[t]['mlu'] - a[t]['mlu']:+.6f}",
            f"{b[t]['inv_load_cost'] - a[t]['inv_load_cost']:+.4f}",
            f"{b[t]['jain_index'] - a[t]['jain_index']:+.4f}",
        ]
    return cells


def headers(slots: list[int]) -> list[str]:
    names = ["instance", "src", "ok", "cost", "paths", "segs"]
    for t in slots:
        names += [f"MLU t{t}", f"inv t{t}", f"jain t{t}"]
    return names


def render(rows: list[list[str]], numeric: list[bool]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        parts = [
            cell.rjust(width) if num else cell.ljust(width)
            for cell, width, num in zip(row, widths, numeric)
        ]
        lines.append("  ".join(parts).rstrip())
    assert len({len(line) for line in lines}) == 1
    rule = "-" * len(lines[0])
    return "\n".join([lines[0], rule, *lines[1:]])


def mlu_profile(result: dict) -> tuple[float, ...] | None:
    """MLUs sorted worst-first, which is the lex order for link utilization."""
    got = objectives(result)
    if got is None:
        return None
    return tuple(sorted((row["mlu"] for row in got), reverse=True))


def inv_cost(result: dict) -> float | None:
    got = objectives(result)
    if got is None:
        return None
    return sum(row["inv_load_cost"] for row in got)


def smaller(left, right) -> int:
    """-1 when left is smaller, 1 when right is, 0 on a tie. Inf equals inf."""
    if isinstance(left, tuple):
        for a, b in zip(left, right, strict=True):
            sign = smaller(a, b)
            if sign:
                return sign
        return 0
    if left == right:
        return 0
    if abs(left - right) <= 1e-9:
        return 0
    return -1 if left < right else 1


def tally(pairs: list[tuple[dict | None, dict | None]], key) -> tuple[int, int, int]:
    """How often the right result is lower, the left is lower, or they tie."""
    right_lower = left_lower = tie = 0
    for left, right in pairs:
        if left is None or right is None:
            continue
        a, b = key(left), key(right)
        if a is None or b is None:
            continue
        sign = smaller(b, a)
        if sign == 0:
            tie += 1
        elif sign < 0:
            right_lower += 1
        else:
            left_lower += 1
    return right_lower, left_lower, tie


def _self_check() -> None:
    assert smaller(1.0, 1.0) == 0
    assert smaller(float("inf"), float("inf")) == 0
    assert smaller(0.5, float("inf")) == -1
    assert smaller((0.9, 0.2), (0.9, 0.8)) == -1
    assert smaller((0.9, 0.8), (0.9, 0.8)) == 0
    slots = [0, 1]
    present = {
        "valid": True,
        "total_cost": 1,
        "total_srpaths": 2,
        "total_segments": 3,
        "objectives": [{"t": 0, "mlu": 0.5, "inv_load_cost": 1.0, "jain_index": 0.5}],
    }
    width = len(headers(slots))
    assert len(metric_row("setA-01", "LP", present, slots)) == width
    assert len(missing_row("", "CL", slots)) == width
    assert len(delta_row("CL-LS", None, present, slots)) == width
    assert metric_row("", "CL", None, slots)[2:] == ["-"] * (width - 2)


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instances",
        nargs="*",
        help="instance names to compare (default: every saved solution)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rerun local search even when out/ls already has a solution",
    )
    args = parser.parse_args()
    if not CHECKER.exists():
        raise SystemExit(f"checker not found at {CHECKER}")

    known = set(solution_names())
    chosen = []
    for name in solution_names():
        if args.instances and name not in args.instances and located(name) not in args.instances:
            continue
        chosen.append(name)
    if args.instances:
        missing = [name for name in args.instances if name not in known and name.split("/")[-1] not in known]
        if missing:
            raise SystemExit(f"no saved solution for {', '.join(missing)}")
    if not chosen:
        raise SystemExit("no saved solutions for LP, local search, or closure search")

    compared = []
    for name in chosen:
        instance = located(name)
        results = {}
        for solver in SOLVERS:
            path = solver_path(solver, name)
            if solver == "LS":
                path = ls_solution(instance, args.recompute)
            results[solver] = check(instance, path) if path is not None else None
        compared.append((name, results))

    slots = slots_of([result for _, results in compared for result in results.values()])
    numeric = [False, False, False, True, True, True] + [True, True, True] * len(slots)
    rows = [headers(slots)]
    pairs = (("LS-LP", "LP", "LS"), ("CL-LP", "LP", "CL"), ("CL-LS", "LS", "CL"))
    for name, results in compared:
        rows.append(metric_row(name, "LP", results["LP"], slots))
        for solver in ("LS", "CL"):
            rows.append(metric_row("", solver, results[solver], slots))
        for label, left, right in pairs:
            rows.append(delta_row(label, results[left], results[right], slots))
    block = len(SOLVERS) + len(pairs)
    aligned = render(rows, numeric).splitlines()
    groups = ["\n".join(aligned[i : i + block]) for i in range(2, len(aligned), block)]
    table = aligned[0] + "\n" + aligned[1] + "\n\n" + "\n\n".join(groups)

    cached = "recomputed" if args.recompute else "saved"
    print("LP  data/<set>/<name>-srpaths.json")
    print(f"LS  out/ls/<name>-srpaths.json ({cached})")
    print("CL  out/ls_closure/<name>-srpaths.json")
    print("A-B is A minus B. Lower MLU and inv are better; higher jain is better.")
    print()
    print(table)
    print()
    for key, label in (
        (mlu_profile, "MLU, worst slot first"),
        (inv_cost, "inverse load cost"),
    ):
        print(label)
        for title, left, right in pairs:
            low_right, low_left, tie = tally(
                [(results[left], results[right]) for _, results in compared], key
            )
            print(f"  {title}: {right} lower {low_right}, {left} lower {low_left}, tie {tie}")


if __name__ == "__main__":
    main()
