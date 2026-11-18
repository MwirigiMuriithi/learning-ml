"""Regenerates README.md from the lesson folders (title + one-line summary come from each lesson.md).
Run from anywhere:  python tools/make_readme.py"""
import pathlib, re

root = pathlib.Path(__file__).resolve().parents[1]
groups = [
    ("Part 0 - Foundations: how models learn and how we judge them", ["00", "01", "02"]),
    ("Part 1 - Regression: predicting numbers", ["03", "04", "05", "06", "07"]),
    ("Part 2 - Classification: predicting classes", ["08", "09", "10", "11", "12", "13"]),
    ("Part 3 - Trees and ensembles", ["14", "15", "16", "17"]),
    ("Part 4 - Neural networks", ["18", "19", "20", "21"]),
    ("Part 5 - Unsupervised learning: structure without labels", ["22", "23", "24", "25", "26", "27", "28", "29", "30"]),
    ("Part 6 - Generative models and representation learning", ["31", "32", "33", "34"]),
    ("Part 7 - Reinforcement learning: learning from rewards", ["35", "36", "37", "38"]),
]
lessons = {p.name[:2]: p for p in sorted((root / "lessons").iterdir()) if p.is_dir()}

def info(folder):
    md = folder / "lesson.md"
    if not md.exists():
        return None
    text = md.read_text().splitlines()
    title = next((l[2:].strip() for l in text if l.startswith("# ")), folder.name)
    one = next((l[1:].strip().replace("**One line:**", "").strip() for l in text if l.startswith(">")), "")
    gifs = len(list(folder.glob("*.gif")))
    return title, one, gifs

out = ["# Machine Learning in Depth", "",
       "A hands-on course on how machine-learning models work: **definition, intuition, maths, from-scratch NumPy code and an animation of each model learning on synthetic but realistic data.**",
       "", "Every lesson folder contains:", "",
       "| file | what it is |", "|---|---|",
       "| `lesson.md` | the written lesson: definition, intuition, maths, algorithm, how to read the animation, hyper-parameters, variants, strengths and weaknesses, metrics, code tour, experiments, pitfalls |",
       "| `<topic>.py` | the model implemented **from scratch** with NumPy (no ML libraries in the main path); running it regenerates the GIFs and prints verification numbers |",
       "| `*.gif` | animations showing the model *learning* step by step |", "",
       "## Quick start", "", "```bash", "pip install -r requirements.txt     # numpy, matplotlib, pillow (scikit-learn only for optional cross-checks)",
       "python lessons/03_linear_regression/linear_regression.py     # regenerate one lesson's animation",
       "bash run_all.sh                                              # regenerate everything (10-20 minutes)", "```", "",
       "Open any `lesson.md` in a Markdown viewer that shows images (GitHub, VS Code preview, Obsidian...) to see the GIFs inline; otherwise open the `.gif` files directly.", "",
       "## How to study a lesson", "",
       "1. Read sections 1-2 (definition and intuition) and watch the GIF with the caption text in view.",
       "2. Read section 5 *Reading the animation* with the GIF open: it tells you what to look at in each panel.",
       "3. Read the maths (section 3), then read the script: every formula appears next to the line that implements it.",
       "4. Run the script, then do the *Experiments* in section 11: change one number and predict what the animation will do before you re-run it.",
       "", "## Course map", ""]
count = 0
for title, ids in groups:
    out += [f"### {title}", "", "| # | lesson | what it teaches | animations |", "|---|---|---|---|"]
    for i in ids:
        f = lessons.get(i)
        inf = info(f) if f else None
        if inf:
            t, one, g = inf
            t = re.sub(r"^\d+\s*·\s*", "", t)
            out.append(f"| {i} | [{t}](lessons/{f.name}/lesson.md) | {one} | {g} |")
            count += 1
        else:
            out.append(f"| {i} | *(not yet available)* | | |")
    out.append("")
out += ["## What is not covered", "",
        "\"Every machine-learning model\" is an open-ended set, so this course covers the *families and their core algorithms* in depth rather than every named model. Each lesson's section 7 (variants) lists relatives that are described but **not** implemented. Families that have no lesson of their own here:", "",
        "- **Other classical models:** Bayesian networks and graphical models in general, factor analysis, ICA, NMF, spectral clustering, Kalman and particle filters, generalised additive models, Bayesian linear models and MCMC, kernel ridge regression, Gaussian-mixture variants beyond EM, ordinal and survival models.",
        "- **Modern deep learning at scale:** graph neural networks, normalising flows, energy-based models, restricted Boltzmann machines, large language-model pre-training and fine-tuning (the attention lesson shows the core mechanism only), vision transformers, contrastive and self-supervised learning, mixture-of-experts.",
        "- **Further reinforcement learning:** actor-critic and PPO (policy gradient with a critic is only touched on in lesson 37), model-based RL, multi-agent RL, offline RL.",
        "- **Practice topics:** feature engineering, hyper-parameter search, deployment, fairness and interpretability tools (SHAP, LIME), causal inference.", "",
        "## Shared code", "",
        "- `common/utils.py` - plotting style, the GIF writer (`Animator`), synthetic data generators, small helpers.",
        "- `common/envs.py` - the vectorised CartPole simulator used by the policy-gradient and DQN lessons.",
        "- `common/nn.py` - a tiny NumPy MLP with manual backpropagation (gradient-checked) used by the autoencoder, VAE, GAN, diffusion and DQN lessons.",
        "- `tools/make_readme.py` - regenerates this file.", "",
        "## Conventions", "",
        "- Every dataset is synthetic and seeded, so results are reproducible; each has a real-world story (house prices, churn, sensors, fraud...).",
        "- Models are written from scratch. Optional scikit-learn cross-checks (inside `try/except`) print whether our implementation agrees.",
        "- Colours follow one fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red); animations carry a caption describing what happens *now*.",
        f"- Lessons available in this archive: **{count}**.", ""]
(root / "README.md").write_text("\n".join(out))
print("README written,", count, "lessons listed")
