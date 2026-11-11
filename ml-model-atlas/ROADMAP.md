# Coverage, honest limits and study order

"Every machine-learning model" has no finite end: the field adds models every month. This course covers the **core canon** that almost everything else
is built from, with working from-scratch code and an animation for each. Below is exactly what is and isn't included.

## Covered (37 lessons)
Optimization · bias/variance · evaluation · linear & polynomial-style regression · ridge/lasso/elastic net · logistic regression · perceptron · k-NN · naive Bayes · LDA/QDA ·
SVM (linear, RBF kernels) · decision trees · random forests/bagging · AdaBoost · gradient boosting (regression & classification) · MLP/backprop · CNN · RNN · LSTM/GRU ·
attention/transformer · autoencoder · VAE · GAN · k-means · hierarchical clustering · DBSCAN · GMM/EM · PCA · t-SNE · SOM · isolation forest · Apriori ·
HMM · Gaussian processes · matrix factorization · time-series (Holt-Winters, AR) · Q-learning & bandits.

## Not yet included (natural next lessons)
- **Supervised:** support-vector *regression*, Poisson/GLM family, ordinal regression, quantile regression, multi-output/structured prediction (CRF), stacking/voting ensembles, XGBoost-style second-order boosting details, CatBoost categorical handling, learning-to-rank.
- **Deep learning:** ResNets/normalization/dropout ablations, seq2seq + beam search, full GPT-style language model, Vision Transformer, graph neural networks, diffusion models, normalizing flows, contrastive/self-supervised learning, mixture-of-experts.
- **Unsupervised:** spectral clustering, mean-shift, ICA, kernel PCA, UMAP, NMF/topic models (LDA topic modelling), word2vec/embeddings, one-class SVM, local outlier factor.
- **Probabilistic:** Bayesian linear regression, Bayesian networks, MCMC/variational inference, Kalman & particle filters, Bayesian optimization.
- **Time series:** full ARIMA/SARIMA, state-space models, Prophet-style decomposition, DeepAR/N-BEATS.
- **RL:** SARSA, DQN, policy gradients/PPO, actor-critic, model-based RL, multi-agent RL.
- **Practice:** feature engineering, cross-validation/hyper-parameter search, calibration, fairness, interpretability (SHAP, LIME), MLOps.

## Suggested order
1. **Foundations:** 00 → 01 → 02, then 03 → 05 (a linear model end to end).
2. **Understand non-linearity:** 07 → 11 → 12 → 14 (trees & ensembles), 10 (SVM), then 15 (neural nets).
3. **Unsupervised:** 23 → 26 → 27 → 28.
4. **Sequences & modern deep learning:** 17 → 18 → 19, then 20 → 22.
5. **Specialities:** 25, 30, 31, 32, 33, 34, 35, 36 in any order.

## Ideas for extending the project
Add a lesson by copying any file in `lessons/`: keep the docstring headings (Definition · Intuition · The math · … · What the animation shows · Try this),
write `main(gif, quick)`, and end with `run(main, __file__)`; `build_docs.py` will pick it up (add its number to `PARTS`).
