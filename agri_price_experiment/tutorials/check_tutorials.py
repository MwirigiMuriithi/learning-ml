#!/usr/bin/env python
"""Run every tutorial notebook top to bottom and report problems (drift check).

    python tutorials/check_tutorials.py            # executes copies; the shipped notebooks are NOT modified
    python tutorials/check_tutorials.py --save     # re-execute and overwrite the notebooks with fresh outputs
    python tutorials/check_tutorials.py 05 08      # only notebooks whose file name starts with these prefixes

A notebook "passes" if every cell runs without an exception. This does NOT check that the written explanation still
matches the printed numbers - after a re-run with another seed or changed config, re-read the text against the outputs.
"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent / "notebooks"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv
    files = sorted(NB_DIR.glob("*.ipynb"))
    if args:
        files = [f for f in files if any(f.name.startswith(a) for a in args)]
    failed = 0
    for f in files:
        nb = nbformat.read(f, as_version=4)
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
            status = "ok"
            if save:
                nbformat.write(nb, f)
        except Exception as e:                      # noqa
            status = "FAILED: " + str(e).strip().splitlines()[-1][:200]
            failed += 1
        print(f"{f.name:42s} {time.time() - t0:5.0f}s  {status}")
    print("\nall notebooks ran cleanly" if not failed else f"\n{failed} notebook(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
