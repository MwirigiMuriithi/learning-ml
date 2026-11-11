# Applied Machine Learning: From Data Analysis to Machine Learning Engineering

A comprehensive, hands-on machine learning curriculum designed to take learners from foundational data manipulation to end-to-end model development and algorithmic mastery.

The course pairs conceptual rigor with real-world datasets, leak-proof engineering discipline, and from-scratch algorithmic implementations.

---

## 🗺️ Curriculum Overview

```text
learn-ml/
├── phase1-data-analysis/          # Weeks 1–2: Exploratory Data Analysis & Manipulation
├── phase2-preprocessing-course/   # Weeks 3–4: Data Cleaning, Feature Engineering & Pipelines
├── phase3-ml-fundamentals/        # Weeks 5–7: Core Machine Learning Concepts & Evaluation
├── ml-model-atlas/                # 37 Algorithms: From-Scratch Code, Visuals & Notebooks
├── machine_learning_model_comparison_lab.ipynb # Multi-model benchmarking lab
├── ROADMAP.md                     # Comprehensive 20-week engineering roadmap
└── models.pdf                     # Visual architecture and model reference sheet
```

---

## 📚 Phase Structure

| Phase / Module | Duration | Core Focus | Key Stack & Artifacts |
|---|---|---|---|
| **[Phase 1: Data Analysis & Exploration](phase1-data-analysis/)** | Weeks 1–2 | Data inspection, vectorization, aggregation, statistical visualization, and structured EDA playbooks. | NumPy, pandas, Matplotlib, Seaborn · 10 lesson notebooks + capstone |
| **[Phase 2: Data Preprocessing](phase2-preprocessing-course/)** | Weeks 3–4 | Missing values, outliers, scaling, encoding, feature engineering, selection, data leakage prevention, and production pipelines. | scikit-learn, joblib · 8 lesson notebooks + AgriReady reference pipeline |
| **[Phase 3: ML Fundamentals](phase3-ml-fundamentals/)** | Weeks 5–7 | Supervised vs unsupervised learning, bias-variance tradeoff, cross-validation, regression/classification metrics, optimization, and capstone project. | scikit-learn · 11 lesson notebooks + practical assignment + capstone |
| **[ML Model Atlas](ml-model-atlas/)** | Reference | 37 machine learning algorithms implemented from scratch in pure NumPy, accompanied by animated learning dynamics and interactive notebooks. | NumPy, Matplotlib · 37 interactive notebooks + 37 Python scripts + HTML gallery |

---

## 🚀 Quick Start

### 1. Clone & Environment Setup

#### Using Anaconda / Conda (Recommended)

```bash
# Create and activate a conda environment
conda create -n learn-ml python=3.10 -y
conda activate learn-ml

# Install required packages
pip install -r phase1-data-analysis/requirements.txt
pip install -r phase2-preprocessing-course/requirements.txt
pip install -r phase3-ml-fundamentals/requirements.txt
pip install -r ml-model-atlas/requirements.txt
```

#### Using Python venv

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r phase1-data-analysis/requirements.txt
```

### 2. Launching the Materials

Launch JupyterLab from the project root to navigate all phases and notebooks:

```bash
jupyter lab
```

### 3. Recommended Learning Path

1. **Start Here**: Open `phase1-data-analysis/notebooks/00_Start_Here.ipynb` to verify your environment and review the learning methodology.
2. **Progress Through Phases**: Follow Phase 1, Phase 2, and Phase 3 sequentially. Each lesson notebook includes self-checks, graduated hints, and reference solutions.
3. **Deepen with the Model Atlas**: Use `ml-model-atlas/gallery.html` or `ml-model-atlas/notebooks/` whenever exploring how an algorithm works under the hood mathematically and computationally.

---

## 🎯 Educational Methodology

Each lesson follows a structured five-step progression:

1. **Intuition First**: Concepts are introduced through relatable physical analogies before formal abstractions.
2. **Visual & Geometric Thinking**: Decision surfaces, error contours, and probability distributions are plotted and animated.
3. **Mathematical Honesty**: Loss functions, objective formulations, and update gradients are clearly stated and connected directly to code.
4. **Leak-Proof Implementation**: Strict discipline around train/validation/test splits, preventing target leakage and overoptimistic evaluation.
5. **Practical Application**: Work on realistic datasets (agricultural yields, commodity prices, customer churn, housing valuation) rather than toy examples.

---

## 📖 Key Documentation & Reference Guides

* **Master Study Plan**: [`ROADMAP.md`](ROADMAP.md) (Complete 20-week roadmap to Machine Learning Engineering)
* **Visual Model Reference**: [`models.pdf`](models.pdf)
* **Model Comparison Benchmark**: [`machine_learning_model_comparison_lab.ipynb`](machine_learning_model_comparison_lab.ipynb)
* **Single-Page Visual Atlas**: [`ml-model-atlas/gallery.html`](ml-model-atlas/gallery.html)
