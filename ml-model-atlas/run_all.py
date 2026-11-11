"""Render every lesson's animations, then rebuild docs + gallery.

    python run_all.py                 # everything (takes a while: ~37 lessons)
    python run_all.py --quick         # fast smoke test with fewer frames/iterations
    python run_all.py --only 03,12    # just those lesson numbers
"""
import argparse, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args(); only = {x.strip().zfill(2) for x in a.only.split(",") if x.strip()}
    lessons = [p for p in sorted((ROOT / "lessons").glob("[0-9][0-9]_*.py")) if not only or p.name[:2] in only]; fails = []; t0 = time.time()
    for p in lessons:
        print(f"==> {p.name}", flush=True); r = subprocess.run([sys.executable, str(p)] + (["--quick"] if a.quick else []), capture_output=True, text=True); print(r.stdout, end="")
        if r.returncode: fails.append(p.name); print(r.stderr[-800:])
    subprocess.run([sys.executable, str(ROOT / "build_docs.py")])
    print(f"\nDone in {time.time() - t0:.0f}s. Failures: {fails or 'none'}. Open gallery.html in a browser.")


if __name__ == "__main__":
    main()
