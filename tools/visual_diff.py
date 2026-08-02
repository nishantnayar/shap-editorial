"""Pixel-diff two trees of PNGs, for visual regression checks on the examples.

    python tools/visual_diff.py <baseline_dir> <current_dir>

Exits non-zero if anything differs, so it can gate a release. See
docs/TESTING.md for the full baseline-vs-current procedure.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def compare(baseline: Path, current: Path):
    rows = []
    for b in sorted(baseline.rglob("*.png")):
        rel = b.relative_to(baseline)
        a = current / rel
        if not a.exists():
            rows.append((rel, "MISSING", ""))
            continue
        ib = Image.open(b).convert("RGBA")
        ia = Image.open(a).convert("RGBA")
        if ib.size != ia.size:
            rows.append((rel, "SIZE", f"{ib.size} -> {ia.size}"))
            continue
        delta = np.abs(np.asarray(ib, int) - np.asarray(ia, int)).sum(axis=2)
        changed = int((delta > 0).sum())
        if changed:
            pct = 100 * changed / delta.size
            detail = f"{changed} px ({pct:.3f}%) maxdelta={int(delta.max())}"
            rows.append((rel, "DIFFERS", detail))
        else:
            rows.append((rel, "IDENTICAL", ""))
    return rows


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    baseline, current = Path(argv[1]), Path(argv[2])
    if not baseline.is_dir() or not current.is_dir():
        sys.exit(f"Both paths must be directories: {baseline}, {current}")

    rows = compare(baseline, current)
    if not rows:
        sys.exit(f"No PNGs found under {baseline}")

    width = max(len(str(r[0])) for r in rows)
    for name, status, detail in rows:
        print(f"{str(name):<{width}}  {status:<10} {detail}")

    changed = [r for r in rows if r[1] != "IDENTICAL"]
    print(f"\n{len(rows) - len(changed)}/{len(rows)} identical")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
