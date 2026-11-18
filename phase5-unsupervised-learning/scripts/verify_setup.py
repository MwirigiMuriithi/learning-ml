"""
verify_setup.py - run this ONCE before you start the course.

    python scripts/verify_setup.py

It checks your Python version, every required library, and that the datasets exist
(it regenerates them if they are missing).
"""
import importlib, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True
print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 9) else "-> NEED 3.9 OR NEWER")
ok &= sys.version_info >= (3, 9)

for mod, pip_name in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                      ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("mlxtend", "mlxtend"), ("jupyterlab", "jupyterlab")]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {pip_name:14s} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  MISSING {pip_name:11s} -> pip install {pip_name}")
        ok = False

try:
    from sklearn.cluster import HDBSCAN  # noqa: F401
    print("  OK   HDBSCAN available in scikit-learn")
except ImportError:
    print("  NOTE: HDBSCAN needs scikit-learn >= 1.3 (only Lesson 3's last demo uses it)")

need = ["retail_customers", "farmers", "markets", "products", "grocery_baskets", "agro_input_baskets"]
if not all((ROOT / "data" / f"{n}.csv").exists() for n in need):
    print("Datasets missing -> regenerating...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_datasets.py")], check=True)
print("\nAll set! Open notebooks/00_welcome_and_setup.ipynb" if ok else "\nFix the items above, then run this script again.")
