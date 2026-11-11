# ML Model Atlas: Every Core Machine-Learning Model, Explained, Animated, and Interactive

37 lessons across 6 core machine learning areas. Each model is available both as an **interactive Jupyter notebook** (`notebooks/`) and as a **standalone Python script** (`lessons/`), structured into four key components:

1. **Definition & intuition**: plain-language conceptual explanations with practical analogies.
2. **The math**: the exact equations and loss formulations implemented in code.
3. **From-scratch code**: pure NumPy implementations (plus a concise autograd engine for deep models), avoiding black boxes.
4. **An animation of the learning process**: visual demonstrations on synthetic-but-practical datasets (house prices, loan defaults, spam classification, mall customer clusters, sensor anomalies, supermarket baskets, sales forecasts).

## Quick start

### Option A: Interactive Jupyter Notebooks

Open the `notebooks/` directory in JupyterLab or VS Code:

    jupyter lab notebooks/

Every notebook is self-contained: it includes the theoretical background, KaTeX math formulations, the complete from-scratch model implementation, an execution cell, and inline animations.

### Option B: HTML Gallery & Pre-rendered Visualizations

The animations are **already pre-rendered in `outputs/`**, so you can open the complete interactive gallery immediately:

    open gallery.html                      # theory + animation + notebook links, all on one page

### Option C: Run Lessons via Python CLI

Install the minimal dependencies (numpy, matplotlib, pillow):

    pip install -r requirements.txt
    python lessons/12_random_forest.py              # runs lesson and updates animation
    python lessons/12_random_forest.py --quick      # fast smoke test with fewer iterations

To rebuild all documentation and notebooks:

    python build_docs.py                   # rebuilds docs/ and gallery.html
    python build_notebooks.py              # regenerates all 37 Jupyter notebooks

---

## The curriculum

| # | Lesson | Formats | What you will watch |
|---|--------|---------|---------------------|
| **Part 0 · Foundations** | | | |
| 00 | Gradient descent & optimizers | [Notebook](notebooks/00_gradient_descent_optimizers.ipynb) · [Script](lessons/00_gradient_descent_optimizers.py) | SGD, Momentum, RMSprop, Adam racing down a valley |
| 01 | Bias, variance & overfitting | [Notebook](notebooks/01_bias_variance_overfitting.ipynb) · [Script](lessons/01_bias_variance_overfitting.py) | Polynomial fits on many training sets; bias²/variance curves |
| 02 | Classification metrics, ROC | [Notebook](notebooks/02_classification_metrics_roc.ipynb) · [Script](lessons/02_classification_metrics_roc.py) | Threshold sweep → confusion matrix, precision/recall, ROC point |
| **Part 1 · Classical supervised** | | | |
| 03 | Linear regression | [Notebook](notebooks/03_linear_regression.ipynb) · [Script](lessons/03_linear_regression.py) | Line fitting + loss surface descent; house-price regression |
| 04 | Ridge / Lasso / Elastic Net | [Notebook](notebooks/04_regularization_ridge_lasso.ipynb) · [Script](lessons/04_regularization_ridge_lasso.py) | Coefficient paths as λ grows; feature selection |
| 05 | Logistic regression | [Notebook](notebooks/05_logistic_regression.ipynb) · [Script](lessons/05_logistic_regression.py) | Probability map & decision line rotating into place |
| 06 | Perceptron | [Notebook](notebooks/06_perceptron.ipynb) · [Script](lessons/06_perceptron.py) | Mistake-driven updates; failure on XOR |
| 07 | k-Nearest Neighbors | [Notebook](notebooks/07_k_nearest_neighbors.ipynb) · [Script](lessons/07_k_nearest_neighbors.py) | Boundaries for k = 1…61; a query voting with its neighbours |
| 08 | Naive Bayes | [Notebook](notebooks/08_naive_bayes.ipynb) · [Script](lessons/08_naive_bayes.py) | Gaussians sharpening as the spam filter sees more email |
| 09 | LDA / QDA | [Notebook](notebooks/09_discriminant_analysis_lda_qda.ipynb) · [Script](lessons/09_discriminant_analysis_lda_qda.py) | Linear vs curved boundary as class covariances diverge |
| 10 | Support Vector Machines | [Notebook](notebooks/10_support_vector_machine.ipynb) · [Script](lessons/10_support_vector_machine.py) | Margin, support vectors, kernel trick, C/γ sweeps |
| 11 | Decision trees | [Notebook](notebooks/11_decision_tree.ipynb) · [Script](lessons/11_decision_tree.py) | Splits growing level by level; regression staircase |
| 12 | Random forest | [Notebook](notebooks/12_random_forest.ipynb) · [Script](lessons/12_random_forest.py) | Averaging trees smooths the boundary; OOB score |
| 13 | AdaBoost | [Notebook](notebooks/13_adaboost.ipynb) · [Script](lessons/13_adaboost.py) | Point weights growing on hard cases; stumps voting |
| 14 | Gradient boosting | [Notebook](notebooks/14_gradient_boosting.ipynb) · [Script](lessons/14_gradient_boosting.py) | Trees fitting residuals; classification via log-loss |
| **Part 2 · Deep learning** | | | |
| 15 | MLP & backpropagation | [Notebook](notebooks/15_neural_network_backprop.ipynb) · [Script](lessons/15_neural_network_backprop.py) | A network untangling 3 spirals; gradient check |
| 16 | CNN (conv + pool + backprop) | [Notebook](notebooks/16_convolutional_neural_network.ipynb) · [Script](lessons/16_convolutional_neural_network.py) | Filters becoming edge detectors; feature maps |
| 17 | RNN & BPTT | [Notebook](notebooks/17_recurrent_neural_network.ipynb) · [Script](lessons/17_recurrent_neural_network.py) | Forecasting a seasonal signal; hidden-state memory |
| 18 | LSTM & GRU | [Notebook](notebooks/18_lstm_gru_gated_memory.ipynb) · [Script](lessons/18_lstm_gru_gated_memory.py) | Gates and cell state storing a signal across 30 noisy steps |
| 19 | Attention & Transformer | [Notebook](notebooks/19_attention_transformer.ipynb) · [Script](lessons/19_attention_transformer.py) | Attention locking onto the right token; PE & QKᵀ figure |
| **Part 3 · Generative & representation learning** | | | |
| 20 | Autoencoder | [Notebook](notebooks/20_autoencoder.ipynb) · [Script](lessons/20_autoencoder.py) | Decoder curve unfolding along a learned manifold |
| 21 | Variational autoencoder | [Notebook](notebooks/21_variational_autoencoder.ipynb) · [Script](lessons/21_variational_autoencoder.py) | Latent space organizing; generating from noise |
| 22 | GAN | [Notebook](notebooks/22_generative_adversarial_network.ipynb) · [Script](lessons/22_generative_adversarial_network.py) | Generator vs discriminator on bimodal data |
| **Part 4 · Unsupervised** | | | |
| 23 | K-means | [Notebook](notebooks/23_k_means.ipynb) · [Script](lessons/23_k_means.py) | Assign/update steps, bad init vs k-means++, elbow |
| 24 | Hierarchical clustering | [Notebook](notebooks/24_hierarchical_clustering.ipynb) · [Script](lessons/24_hierarchical_clustering.py) | Merges + dendrogram growing |
| 25 | DBSCAN | [Notebook](notebooks/25_dbscan.ipynb) · [Script](lessons/25_dbscan.py) | Density expansion, noise points, k-distance plot |
| 26 | Gaussian mixture / EM | [Notebook](notebooks/26_gaussian_mixture_em.ipynb) · [Script](lessons/26_gaussian_mixture_em.py) | Ellipses fitting; soft assignments; BIC |
| 27 | PCA | [Notebook](notebooks/27_pca.ipynb) · [Script](lessons/27_pca.py) | Rotating projection (variance vs error); 3-D plane |
| 28 | t-SNE | [Notebook](notebooks/28_tsne.ipynb) · [Script](lessons/28_tsne.py) | 64-D images unfolding into clusters |
| 29 | Self-organizing map | [Notebook](notebooks/29_self_organizing_map.ipynb) · [Script](lessons/29_self_organizing_map.py) | A colour atlas organizing itself |
| 30 | Isolation forest | [Notebook](notebooks/30_isolation_forest_anomaly_detection.ipynb) · [Script](lessons/30_isolation_forest_anomaly_detection.py) | Random cuts: outlier vs normal; score map |
| 31 | Apriori / association rules | [Notebook](notebooks/31_apriori_association_rules.ipynb) · [Script](lessons/31_apriori_association_rules.py) | Candidate pruning; rules by lift |
| **Part 5 · Sequences, probability, recommenders, forecasting, RL** | | | |
| 32 | Hidden Markov models | [Notebook](notebooks/32_hidden_markov_model.ipynb) · [Script](lessons/32_hidden_markov_model.py) | Viterbi trellis; Baum–Welch learning |
| 33 | Gaussian processes | [Notebook](notebooks/33_gaussian_process.ipynb) · [Script](lessons/33_gaussian_process.py) | Uncertainty tightening as data arrives; length-scale sweep |
| 34 | Matrix factorization | [Notebook](notebooks/34_matrix_factorization_recommender.ipynb) · [Script](lessons/34_matrix_factorization_recommender.py) | Ratings matrix filling in; taste space |
| 35 | Time-series forecasting | [Notebook](notebooks/35_time_series_forecasting.ipynb) · [Script](lessons/35_time_series_forecasting.py) | Seasonal naive vs Holt-Winters vs AR, walk-forward |
| 36 | Reinforcement learning | [Notebook](notebooks/36_reinforcement_learning.ipynb) · [Script](lessons/36_reinforcement_learning.py) | Q-learning cliff-walking; bandits (ε-greedy / UCB / Thompson) |

See `ROADMAP.md` for study sequence recommendations and future expansion areas.

---

## How to study with this

For each lesson:
1. **Explore the Notebook**: Open the lesson notebook in `notebooks/` to read the theory, run the code step-by-step, and inspect intermediate variables.
2. **Watch the Animation**: Observe how the loss surfaces, decision boundaries, or clusters evolve over iterations.
3. **Inspect the from-scratch implementation**: Every implementation uses clean, explicit NumPy calculations without opaque wrappers.
4. **Complete the "Try this" experiments**: Modify learning rates, add noise, alter hyperparameters, and predict the outcome.

---

## Repository Layout

```text
ml-model-atlas/
├── notebooks/       # Self-contained interactive Jupyter notebooks (00 through 36)
├── lessons/         # Standalone Python scripts for each algorithm
├── mlatlas/         # Shared helpers: data.py, viz.py, nn.py (autograd), trees.py (CART)
├── outputs/         # Pre-rendered animated GIFs and reference diagrams
├── docs/            # Markdown documentation for all lessons + master INDEX.md
├── gallery.html     # Single-page visual gallery with embedded theory & animations
├── build_notebooks.py # Script to regenerate all 37 Jupyter notebooks
└── build_docs.py    # Script to regenerate docs/ and gallery.html
```
