#!/usr/bin/env python3
"""Run this first:  python verify_setup.py
Checks your Python, libraries and the bundled datasets, and tells you exactly what (if anything) to fix."""
import importlib, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ok = True
def line(good, msg, fix=""):
    global ok
    ok &= good
    print(("✅ " if good else "❌ ") + msg + ("" if good else f"\n     → {fix}"))

print("== Python ==")
line(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "install Python 3.9 or newer (3.10–3.12 recommended)")

print("\n== Required libraries ==")
for mod, pip in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
                 ("sklearn", "scikit-learn"), ("PIL", "pillow"), ("torch", "torch"), ("ipykernel", "ipykernel")]:
    try:
        m = importlib.import_module(mod); line(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        line(False, f"{mod} cannot be imported ({type(e).__name__})", f"pip install {pip}")

print("\n== Optional (Lesson 15) ==")
for mod in ["keras", "tensorflow"]:
    try:
        m = importlib.import_module(mod); print(f"✅ {mod} {getattr(m, '__version__', '')}")
    except Exception:
        print(f"⚪ {mod} not installed: fine for Lessons 00–14 (pip install {mod} for Lesson 15's TensorFlow cells)")

print("\n== Datasets ==")
try:
    sys.path.insert(0, str(ROOT))
    from utils import dlcourse as dl
    xtr, ytr, xte, yte = dl.load_mnist();    line(xtr.shape == (20000, 28, 28) and xte.shape == (5000, 28, 28), f"MNIST subset {xtr.shape} / {xte.shape}", "re-extract the zip: data/mnist_subset.npz is missing or damaged")
    xtr, ytr, xte, yte = dl.load_fashion();  line(xtr.shape == (15000, 28, 28), f"Fashion-MNIST subset {xtr.shape} / {xte.shape}", "data/fashion_subset.npz missing")
    d = dl.load_leaves();                    line(d["x_train"].shape[1:] == (48, 48, 3), f"Maize leaves train {d['x_train'].shape}, val {d['x_val'].shape}, test {d['x_test'].shape}", "run: python data/build_datasets.py")
    p = dl.load_prices(None);                line(len(p) == 2349, f"Maize prices {p.shape} (3 markets)", "run: python data/build_datasets.py")
except Exception as e:
    line(False, f"could not load datasets ({type(e).__name__}: {e})", "make sure you run this script from the course folder")

print("\n== Quick compute test ==")
try:
    import time, torch
    t = time.time(); a = torch.randn(512, 512); [a @ a for _ in range(20)]
    line(True, f"20 matrix multiplies took {time.time() - t:.2f}s on {'GPU' if torch.cuda.is_available() else 'CPU'} (CPU is fine for this course)")
except Exception:
    pass

print("\n🎉 All set; open notebooks/00_welcome_and_setup.ipynb" if ok else "\n⚠️ Fix the ❌ items above, then run this script again.")
sys.exit(0 if ok else 1)
