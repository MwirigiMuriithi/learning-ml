# Machine Learning in Depth

A hands-on course on how machine-learning models work: **definition, intuition, maths, from-scratch NumPy code and an animation of each model learning on synthetic but realistic data.**

Every lesson folder contains:

| file | what it is |
|---|---|
| `lesson.md` | the written lesson: definition, intuition, maths, algorithm, how to read the animation, hyper-parameters, variants, strengths and weaknesses, metrics, code tour, experiments, pitfalls |
| `<topic>.py` | the model implemented **from scratch** with NumPy (no ML libraries in the main path); running it regenerates the GIFs and prints verification numbers |
| `*.gif` | animations showing the model *learning* step by step |

## Quick start

```bash
pip install -r requirements.txt     # numpy, matplotlib, pillow (scikit-learn only for optional cross-checks)
python lessons/03_linear_regression/linear_regression.py     # regenerate one lesson's animation
bash run_all.sh                                              # regenerate everything (10-20 minutes)
```

Open any `lesson.md` in a Markdown viewer that shows images (GitHub, VS Code preview, Obsidian...) to see the GIFs inline; otherwise open the `.gif` files directly.

## How to study a lesson

1. Read sections 1-2 (definition and intuition) and watch the GIF with the caption text in view.
2. Read section 5 *Reading the animation* with the GIF open: it tells you what to look at in each panel.
3. Read the maths (section 3), then read the script: every formula appears next to the line that implements it.
4. Run the script, then do the *Experiments* in section 11: change one number and predict what the animation will do before you re-run it.

## Course map

### Part 0 - Foundations: how models learn and how we judge them

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 00 | [Gradient Descent & Optimisers](lessons/00_gradient_descent_optimizers/lesson.md) | almost every model in this course is "a loss function plus a way to walk downhill on it" - this lesson studies the walking. | 2 |
| 01 | [Bias, Variance, Underfitting and Overfitting](lessons/01_bias_variance_overfitting/lesson.md) | a model's error on new data is the sum of three parts: a **systematic miss** (bias), **how much the fit wobbles when the training data change** (variance), and **noise nobody can predict**; every model in this course is a position on that trade-off. | 2 |
| 02 | [Evaluating Models: Metrics, Thresholds and Cross-Validation](lessons/02_evaluation_metrics_crossval/lesson.md) | a model is only as good as your ability to measure it: choose a metric that matches the cost of mistakes, choose a threshold that matches the business, and estimate the score on data the model has never seen, more than once. | 2 |

### Part 1 - Regression: predicting numbers

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 03 | [Linear Regression](lessons/03_linear_regression/lesson.md) | fit a straight line (or plane) through data by choosing the slope and intercept that make the squared prediction errors as small as possible. | 1 |
| 04 | [Polynomial Regression](lessons/04_polynomial_regression/lesson.md) | keep linear regression, but feed it powers of the input (`x, x², x³, …`) so the model can bend; the degree controls the flexibility, and too much flexibility fits the noise. | 1 |
| 05 | [Regularisation: Ridge, Lasso & Elastic Net](lessons/05_regularization_ridge_lasso/lesson.md) | add a penalty on the size of the weights so the model cannot chase noise; the shape of the penalty decides whether weights merely shrink (ridge) or switch off completely (lasso). | 2 |
| 06 | [Gaussian-Process Regression](lessons/06_gaussian_process_regression/lesson.md) | instead of choosing one function, keep a probability distribution over *all* smooth functions; observing data simply removes the functions that disagree with it, leaving a prediction with an honest error bar at every point. | 3 |
| 07 | [Time Series: Autocorrelation, AR, ARMA and ARIMA](lessons/07_time_series_ar_arima/lesson.md) | in a time series, today's value tells you about tomorrow's; model that memory (autoregression), remove trend and seasonality first (differencing or regression), and forecast with bands that widen the further you look. | 3 |

### Part 2 - Classification: predicting classes

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 08 | [Logistic Regression](lessons/08_logistic_regression/lesson.md) | take a linear score, squash it through a sigmoid to get a probability, and train it with log-loss so that confident mistakes are punished hard. | 2 |
| 09 | [k-Nearest Neighbours (kNN)](lessons/09_knn/lesson.md) | to classify a new point, find the k most similar stored points and let them vote; there is no training, the data *is* the model. | 3 |
| 10 | [Naive Bayes](lessons/10_naive_bayes/lesson.md) | apply Bayes' rule, and assume features are independent *given the class*, so the model reduces to counting and multiplying simple per-feature probabilities. | 2 |
| 11 | [Linear & Quadratic Discriminant Analysis (LDA / QDA)](lessons/11_lda_qda/lesson.md) | model each class as a Gaussian, then let Bayes' rule decide; sharing one covariance gives a straight boundary (LDA), giving each class its own gives a curved one (QDA). LDA is also Fisher's "best separating direction". | 2 |
| 12 | [The Perceptron](lessons/12_perceptron/lesson.md) | the oldest learning machine: look at one example, and only if you get it wrong, add (or subtract) that example to the weights. On separable data it provably stops. | 2 |
| 13 | [Support Vector Machines (SVM)](lessons/13_svm/lesson.md) | choose the boundary with the widest empty "street" between the classes; only the points on the street's edges (the support vectors) decide it, and kernels let the same idea draw curved boundaries. | 4 |

### Part 3 - Trees and ensembles

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 14 | [Decision Trees (CART)](lessons/14_decision_tree/lesson.md) | repeatedly split the data with the single yes/no question on one feature that makes the groups purest; the result is a flowchart of if-then rules. | 2 |
| 15 | [Random Forests](lessons/15_random_forest/lesson.md) | train many deep, deliberately different decision trees, each on a bootstrap sample and with random feature choices, then let them vote; the noise each tree memorises cancels out. | 2 |
| 16 | [AdaBoost](lessons/16_adaboost/lesson.md) | build a strong classifier from many weak ones, one after another, each trained to fix the mistakes of the ones before by up-weighting the examples they got wrong. | 1 |
| 17 | [Gradient Boosting](lessons/17_gradient_boosting/lesson.md) | add small trees one at a time, each fitted to the direction (the negative gradient of the loss) in which the current ensemble's predictions should move: gradient descent, but in the space of functions. | 3 |

### Part 4 - Neural networks

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 18 | [Multi-Layer Perceptron & Backpropagation](lessons/18_mlp_backprop/lesson.md) | stack layers of "weighted sum then bend" units, and use the chain rule (backpropagation) to compute how every single weight should change to reduce the loss. | 2 |
| 19 | [Convolutional Neural Networks (CNN)](lessons/19_cnn/lesson.md) | slide a small learned filter over an image so the same pattern detector is reused at every position; stack detectors to build features from edges to objects. | 1 |
| 20 | [Recurrent Neural Networks and LSTMs](lessons/20_rnn_lstm/lesson.md) | a network that reads a sequence one element at a time while carrying a hidden "memory" vector from step to step; the plain version forgets quickly because learning signals shrink as they travel back in time, and the LSTM fixes that with gated, additive memory. | 3 |
| 21 | [Attention and the Transformer](lessons/21_attention_transformer/lesson.md) | let every element of a sequence look at every other element directly: compare a *query* with all *keys*, turn the comparison scores into weights with a softmax, and return the weighted mix of the *values*; stack that with feed-forward layers and you have the transformer. | 3 |

### Part 5 - Unsupervised learning: structure without labels

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 22 | [K-Means Clustering](lessons/22_kmeans/lesson.md) | split unlabeled data into `k` groups by alternating two simple steps - give every point to its nearest centre, then move every centre to the mean of its points - until nothing changes. | 2 |
| 23 | [Hierarchical (Agglomerative) Clustering](lessons/23_hierarchical_clustering/lesson.md) | start with every point as its own cluster, repeatedly merge the two closest clusters, and record the order and distances of the merges as a tree (dendrogram); cut the tree wherever you like to get any number of clusters. | 2 |
| 24 | [DBSCAN (Density-Based Clustering)](lessons/24_dbscan/lesson.md) | a cluster is a region where points are packed densely; start from any dense point and let the cluster spread to every point it can reach through chains of close neighbours, and call whatever is left over *noise*. | 2 |
| 25 | [Gaussian Mixture Models and the EM Algorithm](lessons/25_gmm_em/lesson.md) | model the data as a blend of Gaussian blobs, and fit it by alternating "how responsible is each blob for each point?" (E-step) with "refit each blob to the points it is responsible for" (M-step): soft, probabilistic k-means with elliptical clusters. | 2 |
| 26 | [Principal Component Analysis (PCA)](lessons/26_pca/lesson.md) | rotate the coordinate system to line up with the directions of greatest variance, then keep only the first few; you lose the least information possible for that number of dimensions. | 3 |
| 27 | [t-SNE (t-distributed Stochastic Neighbour Embedding)](lessons/27_tsne/lesson.md) | turn "who is whose neighbour" in high dimensions into a probability distribution, then move points around in 2-D until their neighbour probabilities match; a map that keeps neighbourhoods together, with a heavy-tailed kernel that lets clusters separate. | 2 |
| 28 | [Self-Organising Map (Kohonen Map)](lessons/28_self_organizing_map/lesson.md) | a grid of neurons is stretched over the data by competitive learning, so that neighbouring neurons end up representing similar data: a topology-preserving 2-D map of high-dimensional data. | 2 |
| 29 | [Anomaly Detection: Isolation Forest (and friends)](lessons/29_anomaly_isolation_forest/lesson.md) | anomalies are few and different, so random cuts of the feature space isolate them in far fewer steps than normal points. | 2 |
| 30 | [Hidden Markov Models](lessons/30_hidden_markov_model/lesson.md) | a system hops between hidden states that you cannot see (with a rule that depends only on the current state), and each state emits noisy observations you can see; the HMM algorithms infer the hidden path and learn the rules from the observations alone. | 2 |

### Part 6 - Generative models and representation learning

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 31 | [Autoencoders](lessons/31_autoencoder/lesson.md) | train a network to copy its input through a narrow bottleneck; to succeed it must discover the compact hidden structure of the data. | 2 |
| 32 | [Variational Autoencoder (VAE)](lessons/32_vae/lesson.md) | an autoencoder whose code is a *probability distribution* pushed toward a simple prior, so that the latent space becomes a smooth, gap-free map you can sample from to generate new data. | 2 |
| 33 | [Generative Adversarial Networks (GAN and WGAN)](lessons/33_gan/lesson.md) | train a *generator* to forge data and a *discriminator* to catch forgeries; each improves by competing with the other, and at the end the generator's output is hard to tell from the real thing. | 2 |
| 34 | [Diffusion Models (DDPM and DDIM)](lessons/34_diffusion/lesson.md) | destroy data by adding a little Gaussian noise many times, train a network to undo one noising step, then generate new data by starting from pure noise and denoising it step by step. | 3 |

### Part 7 - Reinforcement learning: learning from rewards

| # | lesson | what it teaches | animations |
|---|---|---|---|
| 35 | [Multi-Armed Bandits: Exploration versus Exploitation](lessons/35_multi_armed_bandits/lesson.md) | you must repeatedly choose among options with unknown payoffs; every choice both earns reward and teaches you something, so the art is balancing *trying new things* against *using what already works*. | 3 |
| 36 | [Q-learning and SARSA: Learning from Rewards and Delayed Consequences](lessons/36_q_learning/lesson.md) | an agent in a world with *states* learns, purely by trying actions and seeing rewards, a table `Q(s, a)` of how good each action is in each state, by repeatedly nudging each estimate toward "reward now + value of where I end up". | 2 |
| 37 | [Policy Gradient (REINFORCE and Baselines)](lessons/37_policy_gradient/lesson.md) | instead of learning how good each action is and then picking the best, learn the *policy itself*: make actions that were followed by high return more probable and actions followed by low return less probable. | 2 |
| 38 | [Deep Q-Network (DQN)](lessons/38_deep_q_network/lesson.md) | Q-learning (lesson 36) with a neural network instead of a table, made stable by two tricks: learn from a *shuffled memory* of past experience (replay) and compute the learning target with a *slowly changing copy* of the network (target network). | 2 |

## What is not covered

"Every machine-learning model" is an open-ended set, so this course covers the *families and their core algorithms* in depth rather than every named model. Each lesson's section 7 (variants) lists relatives that are described but **not** implemented. Families that have no lesson of their own here:

- **Other classical models:** Bayesian networks and graphical models in general, factor analysis, ICA, NMF, spectral clustering, Kalman and particle filters, generalised additive models, Bayesian linear models and MCMC, kernel ridge regression, Gaussian-mixture variants beyond EM, ordinal and survival models.
- **Modern deep learning at scale:** graph neural networks, normalising flows, energy-based models, restricted Boltzmann machines, large language-model pre-training and fine-tuning (the attention lesson shows the core mechanism only), vision transformers, contrastive and self-supervised learning, mixture-of-experts.
- **Further reinforcement learning:** actor-critic and PPO (policy gradient with a critic is only touched on in lesson 37), model-based RL, multi-agent RL, offline RL.
- **Practice topics:** feature engineering, hyper-parameter search, deployment, fairness and interpretability tools (SHAP, LIME), causal inference.

## Shared code

- `common/utils.py` - plotting style, the GIF writer (`Animator`), synthetic data generators, small helpers.
- `common/envs.py` - the vectorised CartPole simulator used by the policy-gradient and DQN lessons.
- `common/nn.py` - a tiny NumPy MLP with manual backpropagation (gradient-checked) used by the autoencoder, VAE, GAN, diffusion and DQN lessons.
- `tools/make_readme.py` - regenerates this file.

## Conventions

- Every dataset is synthetic and seeded, so results are reproducible; each has a real-world story (house prices, churn, sensors, fraud...).
- Models are written from scratch. Optional scikit-learn cross-checks (inside `try/except`) print whether our implementation agrees.
- Colours follow one fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red); animations carry a caption describing what happens *now*.
- Lessons available in this archive: **39**.
