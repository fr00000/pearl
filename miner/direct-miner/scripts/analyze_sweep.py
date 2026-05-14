#!/usr/bin/env python3
"""Analyze an H200 sweep summary.csv and pick the winner.

Usage:
    python analyze_sweep.py /workspace/sweeps/h200-YYYYMMDD-HHMMSS
"""

import csv
import sys
from pathlib import Path


H100_BASELINE_TILES_PER_S = 1_935_876  # H100 PCIe winner reference

def main(sweep_dir: Path) -> int:
    csv_path = sweep_dir / "summary.csv"
    if not csv_path.exists():
        print(f"FATAL: {csv_path} not found", file=sys.stderr)
        return 1

    rows = []
    skipped = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                r["mif"] = int(r["max_in_flight"])
                r["mm_s"] = float(r["completion_rate_mm_s"]) if r["completion_rate_mm_s"] else 0.0
                r["tile_s"] = float(r["tile_rate_per_s"]) if r["tile_rate_per_s"] else 0.0
                r["errors"] = int(r["errors"]) if r["errors"] else 0
                r["m"] = int(r["m"])
                r["n"] = int(r["n"])
                r["k"] = int(r["k"])
                r["gpu"] = int(r.get("gpu", -1)) if r.get("gpu") not in (None, "") else -1
                if r["tile_s"] > 0:
                    rows.append(r)
                else:
                    skipped.append(r)
            except (ValueError, KeyError) as e:
                skipped.append({**r, "_err": str(e)})

    if not rows:
        print("FATAL: no completed cells", file=sys.stderr)
        for r in skipped:
            print(f"  SKIP {r.get('shape_id', '?')} mif={r.get('max_in_flight', '?')}: "
                  f"notes={r.get('notes', '')} errors={r.get('errors', '')}")
        return 2

    rows.sort(key=lambda r: r["tile_s"], reverse=True)

    print(f"{'Shape':<18} {'m':>5} {'n':>6} {'k':>6} {'mif':>3} {'gpu':>3} "
          f"{'mm/s':>9} {'tile/s':>14} {'err':>4}")
    print("-" * 84)
    for r in rows:
        print(
            f"{r['shape_id']:<18} {r['m']:>5} {r['n']:>6} {r['k']:>6} "
            f"{r['mif']:>3} {r['gpu']:>3} {r['mm_s']:>9.1f} {r['tile_s']:>14,.0f} "
            f"{r['errors']:>4}"
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} cells:")
        for r in skipped:
            print(f"  {r.get('shape_id', '?')} mif={r.get('max_in_flight', '?')} "
                  f"notes={r.get('notes', '')} errors={r.get('errors', '')}")

    # Best mif per shape
    print("\n=== Best mif per shape (errors == 0) ===")
    by_shape = {}
    for r in rows:
        if r["errors"] > 0:
            continue
        sid = r["shape_id"]
        if sid not in by_shape or r["tile_s"] > by_shape[sid]["tile_s"]:
            by_shape[sid] = r
    for sid, r in sorted(by_shape.items(), key=lambda x: -x[1]["tile_s"]):
        print(f"  {sid:<18} m={r['m']:>5} n={r['n']:>6} k={r['k']:>6}  "
              f"mif={r['mif']}  -> {r['tile_s']:>14,.0f} tiles/s")

    # Overall winner
    winner = next((r for r in rows if r["errors"] == 0), None)
    if not winner:
        print("\nFATAL: no error-free cell to pick as winner", file=sys.stderr)
        return 3

    print(f"\n=== WINNER ===")
    print(f"  shape:  m={winner['m']} n={winner['n']} k={winner['k']}")
    print(f"  mif:    {winner['mif']}")
    print(f"  rate:   {winner['mm_s']:.1f} mm/s, {winner['tile_s']:,.0f} tiles/s")
    print(f"  vs H100 winner ({H100_BASELINE_TILES_PER_S:,} tiles/s): "
          f"{winner['tile_s'] / H100_BASELINE_TILES_PER_S:.2f}x")
    print(f"  4-GPU projected: {4 * winner['tile_s']:,.0f} tiles/s "
          f"({4 * winner['mm_s']:.1f} mm/s aggregate)")

    # Emit a machine-readable line for the launcher script
    print("\n# Production launcher variables (paste into launch_h200_production.sh):")
    print(f"SHAPE_M={winner['m']}")
    print(f"SHAPE_N={winner['n']}")
    print(f"SHAPE_K={winner['k']}")
    print(f"MAX_IN_FLIGHT={winner['mif']}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: analyze_sweep.py <sweep_dir>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
