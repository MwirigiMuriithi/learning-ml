"""
Rebuilds every notebook of the course from the Python sources in source/lessons/.
Requires: pip install nbformat nbconvert nbclient ipykernel   (and the course requirements)

Usage:
  python make_all.py                 # build all notebooks (student: clean, solutions: executed)
  python make_all.py L01 L02         # build only some lessons
  python make_all.py --no-exec       # don't execute (fast structural build)
  python make_all.py --verify-student  # also execute student copies in memory (must not crash)
"""
import importlib
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # the course root (parent of source/)
sys.path.insert(0, os.path.join(HERE, "lessons"))
sys.path.insert(0, HERE)

MODULES = [
    "L00", "L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08",
    "L09", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17",
    "P1", "P2", "P3", "P4",
]


def execute(nb, cwd, timeout=1200):
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": cwd}})
    client.execute()
    return nb


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    do_exec = "--no-exec" not in flags
    verify_student = "--verify-student" in flags
    mods = args or MODULES
    failures = []
    for m in mods:
        t0 = time.time()
        mod = importlib.import_module(m)
        importlib.reload(mod)
        lesson = mod.make()
        is_project = lesson.kind == "Project"
        s_dir = os.path.join(ROOT, "projects" if is_project else "notebooks")
        o_dir = os.path.join(ROOT, "project_solutions" if is_project else "solutions")
        os.makedirs(s_dir, exist_ok=True)
        os.makedirs(o_dir, exist_ok=True)

        student = lesson.build(solutions=False)
        sol = lesson.build(solutions=True)
        status = "built"
        try:
            if do_exec:
                execute(sol, s_dir)           # run from a sibling folder so '..' works
                bad = [o.text for c in sol.cells if c.cell_type == "code" for o in c.get("outputs", [])
                       if o.output_type == "stream" and "❌" in o.text]
                if bad:
                    raise RuntimeError("Self-check failed in SOLUTIONS: " + " | ".join(t.strip()[:120] for t in bad))
                status = "solutions executed OK, all checks ✅"
            if verify_student:
                import copy
                execute(copy.deepcopy(student), s_dir)
                status += " + student runs OK"
        except Exception as exc:  # noqa
            failures.append(m)
            import re as _re
            status = "❌ FAILED: " + _re.sub(r"\x1b\[[0-9;]*m", "", str(exc))[-1800:]
        nbformat.write(student, os.path.join(s_dir, lesson.filename + ".ipynb"))
        nbformat.write(sol, os.path.join(o_dir, lesson.filename + "_SOLUTIONS.ipynb"))
        print(f"[{m}] {lesson.filename:<48} {time.time() - t0:6.1f}s  {status}")
    if failures:
        print("\nFAILED:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
