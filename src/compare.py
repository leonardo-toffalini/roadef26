"""Compare checker scores of LP coordinate descent and local search.

LP solutions are the ``*-srpaths.json`` files already written next to each
instance. Local search solutions live in ``out/ls`` and are computed only
when that file is missing, or when ``--recompute`` is passed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LS_DIR = ROOT / "out" / "ls"
CHECKER = ROOT / "check"


def lp_instances() -> list[tuple[str, Path]]:
    found = []
    for path in sorted(DATA.glob("*/*-srpaths.json")):
        if path.parent.name == "toy":
            continue
        name = path.name.removesuffix("-srpaths.json")
        found.append((f"{path.parent.name}/{name}", path))
    return found


def ls_solution(instance: str, recompute: bool) -> Path:
    path = LS_DIR / f"{Path(instance).name}-srpaths.json"
    if path.exists() and not recompute:
        return path
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


def slots_of(results: list[dict]) -> list[int]:
    found = set()
    for result in results:
        got = objectives(result)
        if got:
            found.update(int(row["t"]) for row in got)
    return sorted(found)


def metric_row(instance: str, src: str, result: dict, slots: list[int]) -> list[str]:
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


def delta_row(lp: dict, ls: dict, slots: list[int]) -> list[str]:
    left, right = objectives(lp), objectives(ls)
    cells = ["", "LS-LP", ""]
    if left is None or right is None:
        cells += ["-"] * (3 + 3 * len(slots))
        return cells
    cells += [
        f"{ls['total_cost'] - lp['total_cost']:+d}",
        f"{ls['total_srpaths'] - lp['total_srpaths']:+d}",
        f"{ls['total_segments'] - lp['total_segments']:+d}",
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


def tally(pairs: list[tuple[dict, dict]], key) -> tuple[int, int, int]:
    ls_lower = lp_lower = tie = 0
    for lp, ls in pairs:
        a, b = key(lp), key(ls)
        if a is None or b is None:
            continue
        sign = smaller(b, a)
        if sign == 0:
            tie += 1
        elif sign < 0:
            ls_lower += 1
        else:
            lp_lower += 1
    return ls_lower, lp_lower, tie


def _self_check() -> None:
    assert smaller(1.0, 1.0) == 0
    assert smaller(float("inf"), float("inf")) == 0
    assert smaller(0.5, float("inf")) == -1
    assert smaller((0.9, 0.2), (0.9, 0.8)) == -1
    assert smaller((0.9, 0.8), (0.9, 0.8)) == 0


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instances",
        nargs="*",
        help="instance names to compare (default: every saved LP solution)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rerun local search even when out/ls already has a solution",
    )
    args = parser.parse_args()
    if not CHECKER.exists():
        raise SystemExit(f"checker not found at {CHECKER}")

    chosen = []
    for instance, lp_path in lp_instances():
        tail = instance.split("/")[-1]
        if args.instances and instance not in args.instances and tail not in args.instances:
            continue
        chosen.append((instance, lp_path))
    if args.instances and len(chosen) != len(set(args.instances)):
        known = {name for name, _ in lp_instances()} | {
            name.split("/")[-1] for name, _ in lp_instances()
        }
        missing = [name for name in args.instances if name not in known]
        if missing:
            raise SystemExit(f"no saved LP solution for {', '.join(missing)}")
    if not chosen:
        raise SystemExit(f"no saved LP solutions under {DATA}")

    compared = []
    for instance, lp_path in chosen:
        ls_path = ls_solution(instance, args.recompute)
        compared.append((instance, check(instance, lp_path), check(instance, ls_path)))

    slots = slots_of([result for _, lp, ls in compared for result in (lp, ls)])
    numeric = [False, False, False, True, True, True] + [True, True, True] * len(slots)
    rows = [headers(slots)]
    for instance, lp, ls in compared:
        label = instance.split("/")[-1]
        rows.append(metric_row(label, "LP", lp, slots))
        rows.append(metric_row("", "LS", ls, slots))
        rows.append(delta_row(lp, ls, slots))
    aligned = render(rows, numeric).splitlines()
    groups = ["\n".join(aligned[i : i + 3]) for i in range(2, len(aligned), 3)]
    table = aligned[0] + "\n" + aligned[1] + "\n\n" + "\n\n".join(groups)

    cached = "recomputed" if args.recompute else "cached"
    print(f"LP  data/<instance>-srpaths.json")
    print(f"LS  out/ls/<instance>-srpaths.json ({cached})")
    print("LS-LP is local search minus LP. Lower MLU and inv are better; higher jain is better.")
    print()
    print(table)
    print()
    pairs = [(lp, ls) for _, lp, ls in compared]
    for key, label in (
        (mlu_profile, "MLU, worst slot first"),
        (inv_cost, "inverse load cost"),
    ):
        ls_lower, lp_lower, tie = tally(pairs, key)
        print(f"{label}: LS lower {ls_lower}, LP lower {lp_lower}, tie {tie}")


if __name__ == "__main__":
    main()
