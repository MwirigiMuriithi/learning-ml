"""
nbkit -- a tiny DSL to author course notebooks once and emit TWO versions:

  * student   : problems have starter code, hints are collapsible, no solutions
  * solutions : same notebook with full solutions + explanations (executed, with outputs)
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

SETUP_CODE = '''import sys, os
sys.path.insert(0, os.path.abspath(".."))      # so we can import course_utils.py
from course_utils import *                      # np, pd, plt, check(), data_path(), ...
import numpy as np, pandas as pd, matplotlib.pyplot as plt
%matplotlib inline
RANDOM_STATE = 42
print("Setup complete ✅")'''


class Lesson:
    def __init__(self, filename, number, title, subtitle, minutes, level, kind="Lesson", setup_extra=""):
        self.filename = filename
        self.number = number
        self.title = title
        self.items = []          # list of (kind, payload)
        self.problem_count = 0
        self.kind = kind
        self.subtitle = subtitle
        self.minutes = minutes
        self.level = level
        self.setup_extra = setup_extra

    # ---- plain cells ------------------------------------------------------------
    def md(self, text):
        self.items.append(("md", text.strip("\n")))

    def code(self, text):
        self.items.append(("code", text.strip("\n")))

    # ---- structured helpers -----------------------------------------------------
    def header(self, objectives, prereqs, roadmap=None, hook=None):
        icon = "📘" if self.kind == "Lesson" else "🏗️"
        tag = f"{self.kind} {self.number}" if self.number is not None else self.kind
        lines = [
            f"# {icon} {tag}: {self.title}",
            f"*{self.subtitle}*",
            "",
            f"| ⏱️ Time | 🎚️ Level | 🧰 Prerequisites |",
            f"|---|---|---|",
            f"| ~{self.minutes} min | {self.level} | {prereqs} |",
            "",
        ]
        if hook:
            lines += ["---", "## 🎬 The Hook", hook, ""]
        lines += ["---", "## 🎯 Learning Objectives", "By the end of this notebook you will be able to:", ""]
        lines += [f"{i}. {o}" for i, o in enumerate(objectives, 1)]
        if roadmap:
            lines += ["", "## 🗺️ Roadmap", ""] + [f"- {r}" for r in roadmap]
        lines += ["", "> 💡 **How to use this notebook:** run cells top to bottom. Read the text *before* you "
                  "run the code. When you reach a 🧩 **Problem**, try it yourself first, open the hints one at "
                  "a time, and only then peek at the solutions notebook."]
        self.md("\n".join(lines))
        self.code(SETUP_CODE + ("\n" + self.setup_extra if self.setup_extra else ""))

    def section(self, emoji, title, text=""):
        self.md(f"---\n## {emoji} {title}\n\n{text}".rstrip())

    def callout(self, kind, text):
        icons = {"think": "🤔 **Pause and think.**", "tip": "💡 **Pro tip.**", "warn": "⚠️ **Watch out.**",
                 "math": "📐 **Math note.**", "story": "📖 **Story time.**", "trap": "🪤 **Common trap.**",
                 "key": "🔑 **Key idea.**"}
        body = "\n".join("> " + ln if ln else ">" for ln in text.strip().split("\n"))
        self.md(f"> {icons[kind]}\n>\n{body}")

    def table(self, rows, header):
        out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        self.md("\n".join(out))

    def quiz(self, qa, title="🧠 Check Your Understanding"):
        parts = [f"---\n## {title}", "Answer each question in your head (or on paper) *before* opening the answer.", ""]
        for i, (q, a) in enumerate(qa, 1):
            parts.append(f"**Q{i}.** {q}\n\n<details><summary>👀 Show answer</summary>\n\n{a}\n\n</details>\n")
        self.md("\n".join(parts))

    def problem(self, title, statement, hints, starter, solution, check_code=None, explain=None, level="⭐"):
        self.problem_count += 1
        self.items.append(("problem", dict(
            n=self.problem_count, title=title, statement=statement, hints=hints, starter=starter,
            solution=solution, check=check_code, explain=explain, level=level)))

    def footer(self, recap, next_title=None, next_file=None):
        lines = ["---", "## 🔁 Recap", ""] + [f"- {r}" for r in recap]
        if next_title:
            lines += ["", f"### ➡️ Up next: **{next_title}**"]
            if next_file:
                lines += [f"Open `{next_file}` when you're ready."]
        lines += ["", "*You did the hard part: you thought about it, not just ran it. See you in the next one!* 🚀"]
        self.md("\n".join(lines))

    # ---- build ----------------------------------------------------------------------
    def build(self, solutions=False):
        cells = []
        for kind, p in self.items:
            if kind == "md":
                cells.append(new_markdown_cell(p))
            elif kind == "code":
                cells.append(new_code_cell(p))
            elif kind == "problem":
                head = f"---\n### 🧩 Problem {p['n']}: {p['title']}  {p['level']}\n\n{p['statement']}"
                cells.append(new_markdown_cell(head))
                if not solutions:
                    hints = "\n\n".join(
                        f"<details><summary>💡 Hint {i}</summary>\n\n{h}\n\n</details>" for i, h in enumerate(p["hints"], 1))
                    cells.append(new_markdown_cell("**Need a nudge?** Open one hint at a time.\n\n" + hints))
                    cells.append(new_code_cell(p["starter"].strip("\n")))
                else:
                    cells.append(new_code_cell("# ✅ SOLUTION\n" + p["solution"].strip("\n")))
                    if p["explain"]:
                        cells.append(new_markdown_cell("**Why this works.** " + p["explain"]))
                if p["check"]:
                    cells.append(new_code_cell(p["check"].strip("\n")))
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        if solutions:
            banner = new_markdown_cell(
                "> 🔓 **SOLUTIONS EDITION**: this is the instructor copy with worked solutions and executed outputs. "
                "Please attempt every problem in the student notebook first; struggling is where learning happens.")
            nb.cells.insert(0, banner)
        return nb
