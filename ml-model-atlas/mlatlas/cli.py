"""Tiny CLI wrapper so every lesson can be run on its own:  python lessons/03_linear_regression.py"""
import argparse, os, pathlib, time


def run(main, file):
    stem = pathlib.Path(file).stem
    ap = argparse.ArgumentParser(description=f"ML Atlas lesson {stem}")
    ap.add_argument("--out", default=str(pathlib.Path(file).resolve().parents[1] / "outputs"),
                    help="folder for the generated GIF/PNG files")
    ap.add_argument("--quick", action="store_true", help="fewer frames/iterations (fast smoke test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    main(os.path.join(a.out, stem + ".gif"), a.quick)
    print(f"[{stem}] finished in {time.time() - t0:.1f}s")
