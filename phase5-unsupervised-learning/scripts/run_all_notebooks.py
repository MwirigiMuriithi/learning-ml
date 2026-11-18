"""
run_all_notebooks.py - (optional) execute every notebook top-to-bottom to confirm the course works on your machine.

    python scripts/run_all_notebooks.py            # runs the notebooks/ folder
    python scripts/run_all_notebooks.py solutions  # runs the solutions/ folder (every self-check should show a green tick)

Needs:  pip install nbformat nbclient
Note: this OVERWRITES outputs saved inside the notebooks. Work on a copy if you have written answers!
"""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
folder = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "notebooks")
failed = 0
for path in sorted(folder.glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(folder)}}, allow_errors=True).execute()
    errors = [o["ename"] for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(f"{'PASS' if not errors else 'FAIL'}  {path.name}  {errors if errors else ''}")
    failed += bool(errors)
sys.exit(1 if failed else 0)
