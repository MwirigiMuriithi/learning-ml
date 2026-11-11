"""
course_utils.py  --  small helpers used by every notebook.

    from course_utils import DATA_DIR, check, check_plot

* DATA_DIR      : Path to the data/ folder (works no matter where you launch Jupyter).
* check(id, x)  : compares YOUR answer with the reference answer and gives friendly feedback.
* check_plot(ax): checks that your chart has the basics (title, axis labels, enough marks).

The reference answers live in course_answers.json. Peeking defeats the purpose --
you are here to build the skill, not to pass the check.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
_ANSWER_FILE = ROOT / "course_answers.json"
_ANSWERS = json.loads(_ANSWER_FILE.read_text()) if _ANSWER_FILE.exists() else {}


# --------------------------------------------------------------------------
# Normalising Python / NumPy / pandas objects into plain JSON-able values
# --------------------------------------------------------------------------
def to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (pd.Timestamp, pd.Period)):
        return str(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, pd.Series):
        return to_jsonable(x.tolist())
    if isinstance(x, pd.DataFrame):
        return to_jsonable(x.to_numpy().tolist())
    if isinstance(x, pd.Index):
        return to_jsonable(x.tolist())
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in (sorted(x) if isinstance(x, set) else x)]
    return str(x)


def _same(a, b, rtol=1e-3, atol=1e-2):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(u, v, rtol, atol) for u, v in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def _shape(x):
    arr = np.asarray(x, dtype=object) if isinstance(x, list) else None
    try:
        return np.array(x).shape
    except Exception:  # ragged
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def check(problem_id, answer):
    """Compare `answer` with the reference answer for `problem_id`."""
    if answer is None:
        print(f"⏳ {problem_id}: you haven't set `answer` yet. Read the problem, try it, then run this cell again.")
        return False
    if problem_id not in _ANSWERS:
        print(f"⚠️  {problem_id}: no reference answer found (is course_answers.json next to course_utils.py?)")
        return False
    mine, ref = to_jsonable(answer), _ANSWERS[problem_id]
    if _same(mine, ref):
        print(f"✅ {problem_id}: correct! Nicely done.")
        return True
    # friendly, non-spoiling diagnostics
    msg = "❌ " + problem_id + ": not quite."
    if type(mine) is not type(ref) and not (isinstance(mine, (int, float)) and isinstance(ref, (int, float))):
        msg += f" Your answer is a {type(mine).__name__}, but the problem expects a {type(ref).__name__}."
    elif isinstance(ref, list):
        if len(mine) != len(ref):
            msg += f" Your answer has {len(mine)} items, but {len(ref)} are expected."
        else:
            msg += " The length is right, but some values differ - check order, rounding and axes."
    elif isinstance(ref, dict):
        if mine.keys() != ref.keys():
            msg += " The keys of your dictionary are different from what's expected."
        else:
            msg += " The keys are right, but some values differ."
    elif isinstance(ref, (int, float)):
        msg += " The number is different from the reference - re-read the rounding instructions and try a hint."
    else:
        msg += " Compare spelling, capitalisation and spacing."
    print(msg)
    return False


def check_plot(ax, title=True, xlabel=True, ylabel=True, n_lines=None, min_lines=None,
               min_bars=None, min_points=None, legend=None, name="plot"):
    """Check that a Matplotlib Axes has the essentials of a good chart."""
    problems = []
    if ax is None:
        print(f"⏳ {name}: no axes found - create your chart first.")
        return False
    # count only real data lines (Seaborn adds empty legend-proxy lines)
    real_lines = [ln for ln in ax.lines if len(ln.get_xdata()) >= 2]
    if title and not ax.get_title():
        problems.append("add a title (ax.set_title)")
    if xlabel and not ax.get_xlabel():
        problems.append("label the x-axis (ax.set_xlabel)")
    if ylabel and not ax.get_ylabel():
        problems.append("label the y-axis (ax.set_ylabel)")
    if n_lines is not None and len(real_lines) != n_lines:
        problems.append(f"expected {n_lines} line(s) but found {len(real_lines)}")
    if min_lines is not None and len(real_lines) < min_lines:
        problems.append(f"expected at least {min_lines} line(s) but found {len(real_lines)}")
    if min_bars is not None and len(ax.patches) < min_bars:
        problems.append(f"expected at least {min_bars} bar/patch objects but found {len(ax.patches)}")
    if min_points is not None:
        pts = sum(len(c.get_offsets()) for c in ax.collections)
        if pts < min_points:
            problems.append(f"expected at least {min_points} scatter points but found {pts}")
    if legend is True and ax.get_legend() is None:
        problems.append("add a legend (ax.legend or a hue/label argument)")
    if problems:
        print(f"❌ {name}: " + "; ".join(problems) + ".")
        return False
    print(f"✅ {name}: your chart has the essentials. Now ask yourself: would a stranger understand it in 10 seconds?")
    return True


def first_look(df, n=5):
    """The 30-second health check we run on every new dataset."""
    from IPython.display import display
    print(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🔁 Fully duplicated rows: {int(df.duplicated().sum()):,}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    display(summary)
    display(df.head(n))
