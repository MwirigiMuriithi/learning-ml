# source/: the notebook generator

Every notebook in this course is *generated* from one Python module per lesson (`lessons/L00.py` … `L17.py`, `P1.py` … `P4.py`).
A module builds a `nbkit.Lesson` with `md()`, `code()`, `header()`, `table()`, `quiz()` and `problem(title, statement, hints, starter, solution, check_code, explain)`;
`make_all.py` writes the **student** notebook (starter code, collapsible hints, no outputs) and the **solutions** notebook (full solutions), executes the solutions notebook and **fails the build if any self-check prints ❌**.

```bash
pip install nbformat nbconvert nbclient ipykernel
python make_all.py L05                  # rebuild one lesson
python make_all.py --verify-student     # rebuild everything; also run every student notebook top-to-bottom
python make_all.py L05 --no-exec        # structure only (does not refresh solution outputs!)
```
Edit a lesson's text or code, rebuild, and read the new outputs: if numbers in the narrative no longer match, the self-checks or your own eyes should tell you.
