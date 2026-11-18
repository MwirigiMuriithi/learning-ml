# 🧠 Deep Learning & Neural Networks: A Hands-On Course
### Phase 8 · Weeks 16–17 · From a single neuron to Transformers, with maize leaves and market prices

> *"The best way to understand a neural network is to watch one learn: then build one yourself."*

Welcome! This is a complete, self-contained course: **17 notebooks** (each with explanations, worked examples, runnable code,
graded practice problems with collapsible hints, and a matching solutions notebook), **4 projects**, a **capstone**, and the datasets
to run everything **offline on an ordinary laptop CPU**: no GPU needed.

You will learn how neural networks work by **building them from scratch in NumPy first**, then using **PyTorch**
(and reading **Keras/TensorFlow**), and finally applying them to agriculture: reading handwriting, classifying clothing,
**diagnosing maize leaf disease** and **forecasting maize prices**.

---

## 🚀 Quick start (5 minutes)

```bash
# 1. unzip, then from inside the course folder:
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # CPU-only PyTorch is much smaller:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
python verify_setup.py                                   # checks libraries + datasets, tells you what to fix
jupyter lab                                              # then open notebooks/00_welcome_and_setup.ipynb
```
*Using Google Colab?* Upload the **whole folder** (keep `data/`, `utils/` and `notebooks/` side by side) to Drive, mount it, and open the notebooks from there.
Each notebook starts with `sys.path.append("..")` so it can find `utils/dlcourse.py` and the data.

---

## 🗂 What's inside

```
dl_course/
├── README.md              ← you are here
├── SYLLABUS.md            ← day-by-day plan, learning outcomes, how to study
├── CHEATSHEET.md          ← PyTorch / shapes / losses / debugging / Keras on one page
├── GLOSSARY.md            ← every term in the course, in plain English
├── requirements.txt, verify_setup.py
├── notebooks/             ← the 17 lessons (start at 00)
├── solutions/             ← worked solutions for every practice problem (try first!)
├── data/                  ← datasets + DATA_CARDS.md + the generator for the synthetic ones
├── utils/dlcourse.py      ← small helper library (loaders, `fit`, plots, self-checks): read it!
└── capstone/              ← brief, rubric, model-card template, ethics checklist, presentation guide
```

## 🗺 Course map

| # | Notebook | You will… | ~Time |
|:-:|:--|:--|:-:|
| 00 | Welcome & setup | See a network learn in 25 lines; refresh NumPy | 1 h |
| 01 | Neuron & perceptron | Build neurons, logic gates, the perceptron rule; meet the XOR wall | 1.25 h |
| 02 | Layers & activations | Shapes, parameter counting, ReLU/sigmoid/softmax, initialisation | 1.5 h |
| 03 | Forward prop & loss | Worked forward pass; MSE, cross-entropy, "initial loss ≈ ln k" | 1.3 h |
| 04 | Gradient descent & backprop | Chain rule by hand; train a net **from scratch**; gradient checking; build an autograd engine | 2 h |
| 05 | PyTorch fundamentals | Tensors, autograd, `nn.Module`, `DataLoader`, the 5-step training loop | 1.7 h |
| 06 | Optimizers, LR, epochs, batches | SGD/momentum/RMSProp/Adam from scratch; LR sweeps; batch-size experiments | 1.7 h |
| 07 | **Project 1** · Handwritten digits | MNIST ANN, baselines, error analysis, reject option, break it with shifts | 2.5 h |
| 08 | CNNs | Convolution by hand, size formulas, pooling, build & inspect a CNN | 2 h |
| 09 | **Project 2** · Image classification | Fashion-MNIST: fair model comparison, BatchNorm, dropout, augmentation | 2.5 h |
| 10 | **Project 3** · Crop disease | Maize-leaf CNN, severity analysis, occlusion maps, triage policy, transfer learning | 2.8 h |
| 11 | RNN / LSTM / GRU | Vanishing gradients through time; LSTM cell verified against PyTorch; adding problem | 2.2 h |
| 12 | **Project 4** · Price forecasting | Chronological splits, baselines, LSTM/GRU, prediction intervals, hold/sell back-test | 3 h |
| 13 | Transformers | Attention from scratch, multi-head, positional encoding, mini Vision Transformer | 2.3 h |
| 14 | Optimisation & regularisation | Dropout/BatchNorm from scratch, weight decay, early stopping, LR schedules, ablation | 2.3 h |
| 15 | Keras & TensorFlow bridge | Rosetta stone, Keras models/callbacks, `tf.GradientTape`, TFLite + quantisation | 1.7 h |
| 16 | 🏆 **Capstone** · FieldAdvisor | Integrate leaf-disease + price models into an uncertainty-aware advisor | 7 h |

**Phase-8 syllabus coverage:** artificial neurons ✔ perceptrons ✔ layers ✔ weights & biases ✔ activations ✔ forward prop ✔ backprop ✔ loss functions ✔
gradient descent ✔ optimizers ✔ learning rates ✔ epochs ✔ batch sizes ✔ ANN ✔ CNN ✔ RNN ✔ LSTM ✔ GRU ✔ Transformers ✔ dropout ✔ batch-norm ✔
early stopping ✔ regularisation ✔ LR scheduling ✔ PyTorch ✔ TensorFlow/Keras ✔, and the four practical projects (digits, image classification, crop disease, LSTM price forecasting).

---

## 🎓 How to get the most out of it

1. **Type, don't paste.** After each cell ask *"what if I changed this?"*: then change it.
2. **Work the problems.** Each has a self-check (✅/❌) and up to three hints. Struggle ~10 minutes before opening Hint 1; look at `solutions/` last.
3. **Predict before you run.** Write down what you expect a plot or number to be. Surprises are where learning happens.
4. **Keep an experiment log** in the projects (what changed → result). It is the habit that separates practitioners from tinkerers.
5. **Explain it aloud**: each lesson ends with key takeaways and a quiz. If you can teach it, you own it.

## 🧪 About the data: please read

| Dataset | Real or synthetic? |
|:--|:--|
| MNIST (20k train / 5k test subset) | ✅ Real: LeCun, Cortes & Burges |
| Fashion-MNIST (15k train / 5k test subset) | ✅ Real: Zalando Research (MIT licence) |
| Maize leaf images (3,000, 5 classes) | 🧪 **Synthetic**: generated by `data/build_datasets.py` |
| Maize prices (3 Kenyan markets, 2011–2025) | 🧪 **Synthetic**: seasonality + rainfall + fuel + droughts |

The agricultural datasets are simulations so the course works offline with known ground truth. **They teach the workflow and the thinking:
not agronomy or real market behaviour.** Lesson 10 and the capstone show how to swap in real photos (PlantVillage etc.) and real prices;
see `data/DATA_CARDS.md` for details and limitations.

## 🔧 Troubleshooting

| Problem | Fix |
|:--|:--|
| `ModuleNotFoundError: utils` | Start Jupyter from the course folder (or keep `notebooks/` next to `utils/`); notebooks add `..` to the path |
| `import torch` fails | `pip install torch` (see CPU-only command above) and restart the kernel |
| Data files not found | You must unzip the **whole** folder; run `python verify_setup.py` |
| Training feels slow | Everything is sized for CPUs. Close other apps; the longest runs (Projects 3–4, Lesson 14) take 1–3 minutes |
| Lesson 15 TensorFlow cells skip | Install `tensorflow` (or `tensorflow-cpu`); without it the notebook runs Keras on PyTorch and skips TF-only cells |
| Results differ slightly from the text | Normal: different CPUs/library versions change floating-point details. Patterns should match; exact digits may not |

*All notebooks were executed end-to-end (student and solution versions) while building this course; solution self-checks run in strict mode.*

Happy learning: you can do this. 🌽
