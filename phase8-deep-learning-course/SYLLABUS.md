# 📅 Syllabus: Phase 8: Deep Learning & Neural Networks (Weeks 16–17)

**Goal.** Understand how neural networks work and be able to train and evaluate basic deep-learning models.

**Total effort:** ≈ 39 hours of focused work (lessons + problems + projects). Two calendar weeks at ~2.5–3 h/day for 14 days:
or 15 "sessions" of 2.5–3 h each. Prefer a gentler pace? Stretch it over 3 weeks: one lesson per day, projects on weekends.

## Learning outcomes: by the end you can…
1. Explain neurons, layers, weights, biases, activation functions, forward propagation, loss and backpropagation, and **implement them from scratch**.
2. Train, debug and evaluate models in **PyTorch** (and read/modify **Keras/TensorFlow**).
3. Choose and tune **optimisers, learning rates, batch sizes, epochs**; diagnose training from learning curves.
4. Build **ANNs, CNNs, RNN/LSTM/GRU models and Transformers** and explain what each architecture assumes.
5. Apply **dropout, batch-norm, early stopping, weight decay and LR schedules** and justify them with an ablation.
6. Run honest ML projects: baselines, validation/test discipline, error analysis, uncertainty, limitations.
7. Deliver four practical projects (digits, image classification, crop-disease classification, LSTM price forecasting) and an integrated capstone.

## Day-by-day plan (15 sessions)

| Session | Notebook(s) | Focus | Output |
|:-:|:--|:--|:--|
| 1 | **00** Welcome & setup · **01** Neuron & perceptron | Orientation; neurons as weighted votes; perceptron rule; XOR wall | Environment verified; first network trained; problems 00 + 01 |
| 2 | **02** Layers & activations | Shapes, parameter counts, non-linearity, activation zoo, init | Hand-built XOR net; problems 02 |
| 3 | **03** Forward prop & loss · start **04** | Worked forward pass; MSE/CE; initial-loss check | Loss functions from scratch; problems 03 |
| 4 | **04** Gradient descent & backprop | Chain rule, backprop, training a net in NumPy, gradient check, autograd engine | A from-scratch MLP solving spirals; problems 04 |
| 5 | **05** PyTorch fundamentals | Tensors, autograd, `nn.Module`, `DataLoader`, 5-step loop, debugging | Moons classifier & sine regressor; problems 05 |
| 6 | **06** Optimizers, LR, epochs, batches | Optimiser race; LR sweep & range test; batch-size & epoch experiments | Experiment log; problems 06 |
| 7 | **07** 🧪 Project 1: Digits | Full ML workflow on MNIST; error analysis; reject option; shift test | Project-1 report; ≥ 96 % test accuracy |
| 8 | **08** CNNs | Convolution, pooling, sizes & params; CNN vs ANN under shifts | A trained CNN; problems 08 |
| 9 | **09** 🧪 Project 2: Image classification | Model ladder (ANN → CNN → BN+dropout → augmentation) on Fashion-MNIST | Project-2 report; experiment table |
| 10 | **10** 🧪 Project 3: Crop disease | Maize-leaf CNN; severity analysis; occlusion; triage policy | Project-3 report incl. limitations |
| 11 | **11** RNN / LSTM / GRU | Recurrence, BPTT, gates, adding problem, windows | LSTM cell verified vs PyTorch; problems 11 |
| 12 | **12** 🧪 Project 4: Price forecasting | Chronological splits, baselines, LSTM/GRU, intervals, back-test | Project-4 report; honest comparison table |
| 13 | **13** Transformers · **14** Optimisation & regularisation (start) | Attention, multi-head, ViT · overfitting & noisy labels | Mini-ViT; first ablation runs |
| 14 | **14** (finish) · **15** Keras & TensorFlow | Dropout/BN/ES/schedules ablation · framework translation, TFLite | Ablation table; Keras models; problems 14–15 |
| 15 | **16** 🏆 Capstone; FieldAdvisor | Integrate both models into an advice engine; evaluate; model card; demo | Notebook + model card + 5-min talk |

> 🕐 *Squeezed for time?* **Core path (~26 h):** 00–10, 11–12, 14, 16 (skip 13 and 15, do problems marked ⭐/⭐⭐ only). 13 and 15 are valuable but optional for the Phase-8 outcome.

## How each lesson is built
Header (time, objectives, roadmap) → **intuition & analogy** → **worked example by hand** → **code** → *Pause and think* prompts → **practice problems** (⭐ warm-up · ⭐⭐ standard · ⭐⭐⭐ challenge, hints, self-checks) → **key takeaways** → **quiz** → *go deeper* reading.

## Assessment (self-assessed or instructor-led)
| Component | Weight | Evidence |
|:--|:-:|:--|
| Practice problems (all lessons) | 20 % | Self-checks ✅ in the notebooks (compare with `solutions/`) |
| Project 1–4 reports | 40 % | Report tables at the end of Lessons 07, 09, 10, 12 |
| Capstone (notebook, model card, talk) | 30 % | `capstone/RUBRIC.md` |
| Participation / explanations | 10 % | Quizzes answered aloud; peer teaching |

## Study habits that work
*Predict → run → explain.* · *One change at a time.* · *Read your error messages (shapes first).* · *Look at your data and your mistakes.* · *Test set once.* · *Teach someone.*

## Reading list (optional)
- Michael Nielsen (*Neural Networks and Deep Learning* (free online) · Goodfellow, Bengio & Courville) *Deep Learning* (free online)
- 3Blue1Brown (neural-network series · Andrej Karpathy) *Zero to Hero* (micrograd, makemore, nanoGPT) and *A Recipe for Training Neural Networks*
- Stanford CS231n (CNNs) · fast.ai *Practical Deep Learning* · Hyndman & Athanasopoulos: *Forecasting: Principles and Practice*
