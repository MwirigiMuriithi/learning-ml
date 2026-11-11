"""
Build self-contained interactive Jupyter notebooks for all 37 lessons in ml-model-atlas.
Each notebook contains:
1. Title, family badge, definition, and intuition
2. Mathematical formulation
3. Environment setup & imports
4. Model architecture & algorithm implementation (from scratch, pure NumPy)
5. Execution and training simulation (main function)
6. Inline animation and visual learning display
7. Assumptions, pitfalls, and when to use
8. Interactive 'Try this' exercises
"""
import ast
import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
LESSONS_DIR = ROOT / "lessons"
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUTS_DIR = ROOT / "outputs"

NOTEBOOKS_DIR.mkdir(exist_ok=True)

PARTS = [
    ("Part 0 · Foundations", 0, 2),
    ("Part 1 · Classical supervised learning", 3, 14),
    ("Part 2 · Neural networks & deep learning", 15, 19),
    ("Part 3 · Generative models & representation learning", 20, 22),
    ("Part 4 · Unsupervised learning", 23, 31),
    ("Part 5 · Sequences, probability, recommenders, forecasting & RL", 32, 36)
]

def get_part_name(num):
    for name, a, b in PARTS:
        if a <= num <= b:
            return name
    return "Machine Learning Reference"

def parse_docstring(doc):
    lines = doc.split("\n")
    title = lines[0].lstrip("# ").strip()
    fam_match = re.search(r"\*\*Family:\*\*\s*(.+)", doc)
    family = fam_match.group(1).strip() if fam_match else ""
    
    sections = {}
    current_sec = "intro"
    current_lines = []
    
    for line in lines[1:]:
        if line.strip().startswith("**Family:**"):
            continue
        m = re.match(r"^##\s+(.+)$", line.strip())
        if m:
            if current_lines:
                sections[current_sec] = "\n".join(current_lines).strip()
            current_sec = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
            
    if current_lines:
        sections[current_sec] = "\n".join(current_lines).strip()
        
    return title, family, sections

def extract_code_blocks(filepath):
    text = filepath.read_text(encoding="utf-8")
    mod = ast.parse(text)
    lines = text.splitlines(keepends=True)
    
    classes_and_helpers = []
    main_func_lines = []
    
    for node in mod.body:
        if isinstance(node, ast.Expr):
            # Could be docstring or sys.path.insert
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            # Check if it's sys.path.insert
            expr_str = ast.unparse(node)
            if "sys.path" in expr_str:
                continue
            classes_and_helpers.extend(lines[node.lineno - 1:node.end_lineno])
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func_lines = lines[node.lineno - 1:node.end_lineno]
        elif isinstance(node, ast.If):
            # if __name__ == '__main__': ignore
            continue
        else:
            classes_and_helpers.extend(lines[node.lineno - 1:node.end_lineno])
            
    return "".join(classes_and_helpers).strip(), "".join(main_func_lines).strip()

def to_cell_source(text):
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    # Ensure every line except possibly empty ends with \n
    res = []
    for l in lines:
        if not l.endswith("\n"):
            res.append(l + "\n")
        else:
            res.append(l)
    return res

def make_cell(cell_type, text):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": to_cell_source(text),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {})
    }

def build_notebook(lesson_path):
    stem = lesson_path.stem
    num = int(stem[:2])
    part_name = get_part_name(num)
    
    text = lesson_path.read_text(encoding="utf-8")
    mod = ast.parse(text)
    doc = ast.get_docstring(mod) or ""
    title, family, sections = parse_docstring(doc)
    helpers_code, main_code = extract_code_blocks(lesson_path)
    
    media = sorted(f.name for f in OUTPUTS_DIR.glob(stem + "*") if f.suffix in (".gif", ".png"))
    
    cells = []
    
    # 1. Header cell
    header_text = f"# {title}\n"
    header_text += f"> **Part:** {part_name}  \n"
    if family:
        header_text += f"> **Family:** `{family}`\n\n"
    
    if "Definition" in sections:
        header_text += f"## Definition\n{sections['Definition']}\n\n"
    if "Intuition" in sections:
        header_text += f"## Intuition\n{sections['Intuition']}\n"
        
    cells.append(make_cell("markdown", header_text.strip()))
    
    # 2. Math cell
    math_key = next((k for k in sections if k.lower().startswith("the math")), None)
    if math_key:
        math_text = f"## The Math\n\n```text\n{sections[math_key]}\n```"
        cells.append(make_cell("markdown", math_text))
        
    # 3. Setup & imports cell
    setup_code = f"""# Setup & Imports
import os
import pathlib
import sys
from IPython.display import Image, display

# Ensure parent folder is in path to import mlatlas
parent_dir = pathlib.Path.cwd().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
from mlatlas import data, viz, cli
"""
    cells.append(make_cell("code", setup_code.strip()))
    
    # 4. Helper classes/functions cell (if any)
    if helpers_code:
        helper_cell_text = f"# Model Implementation / Helper Classes\n{helpers_code}"
        cells.append(make_cell("code", helper_cell_text))
        
    # 5. Main training and execution function cell
    if main_code:
        main_cell_text = f"# Demonstration & Training Pipeline\n{main_code}"
        cells.append(make_cell("code", main_cell_text))
        
    # 6. Interactive Execution cell
    exec_code = f"""# Run the model demonstration (quick=True for fast interactive execution)
gif_path = f"../outputs/{stem}.gif"
main(gif=gif_path, quick=True)
"""
    cells.append(make_cell("code", exec_code.strip()))
    
    # 7. Animation display cell
    anim_key = next((k for k in sections if "animation" in k.lower()), None)
    anim_text = "## Visualizing the Learning Process\n\n"
    if anim_key:
        anim_text += f"{sections[anim_key]}\n\n"
        
    cells.append(make_cell("markdown", anim_text.strip()))
    
    # Display code cell for animations/media
    media_display_code = "# Display the learning animation inline\n"
    for m in media:
        media_display_code += f"""media_file = f"../outputs/{m}"
if os.path.exists(media_file):
    display(Image(filename=media_file))
elif os.path.exists(f"outputs/{m}"):
    display(Image(filename=f"outputs/{m}"))
"""
    cells.append(make_cell("code", media_display_code.strip()))
    
    # 8. Practical tips / Assumptions / Weaknesses
    practical_keys = [
        "Assumptions & pitfalls", "When to use what", "When to use",
        "Practical tips", "Strengths & weaknesses", "Limitations (and why they matter)",
        "How to fight each problem"
    ]
    for pk in practical_keys:
        if pk in sections:
            cells.append(make_cell("markdown", f"## {pk}\n\n{sections[pk]}"))
            
    # 9. Try this / Experiments
    if "Try this" in sections:
        cells.append(make_cell("markdown", f"## 🧪 Try This & Experiments\n\n{sections['Try this']}"))
        
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    nb_path = NOTEBOOKS_DIR / f"{stem}.ipynb"
    with open(nb_path, "w", encoding="utf-8") as fp:
        json.dump(nb, fp, indent=1, ensure_ascii=False)
        fp.write("\n")
        
    return nb_path

def main():
    lessons = sorted(LESSONS_DIR.glob("[0-9][0-9]_*.py"))
    print(f"Generating notebooks for {len(lessons)} lessons...")
    for l in lessons:
        p = build_notebook(l)
        print(f"  ✓ {p.name}")
    print(f"Successfully generated {len(lessons)} notebooks in {NOTEBOOKS_DIR}/")

if __name__ == "__main__":
    main()
