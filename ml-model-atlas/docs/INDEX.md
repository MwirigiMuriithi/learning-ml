# ML Model Atlas · lesson index


## Part 0 · Foundations

- [00 · Gradient Descent & Optimizers](00_gradient_descent_optimizers.md): Foundations · Optimization ([Notebook](../notebooks/00_gradient_descent_optimizers.ipynb))
- [01 · Bias, Variance & Overfitting](01_bias_variance_overfitting.md): Foundations · Generalization ([Notebook](../notebooks/01_bias_variance_overfitting.ipynb))
- [02 · Classification Metrics, Thresholds & ROC](02_classification_metrics_roc.md): Foundations · Evaluation ([Notebook](../notebooks/02_classification_metrics_roc.ipynb))

## Part 1 · Classical supervised learning

- [03 · Linear Regression](03_linear_regression.md): Supervised · Regression ([Notebook](../notebooks/03_linear_regression.ipynb))
- [04 · Regularization: Ridge, Lasso & Elastic Net](04_regularization_ridge_lasso.md): Supervised · Regression (penalized linear models) ([Notebook](../notebooks/04_regularization_ridge_lasso.ipynb))
- [05 · Logistic Regression](05_logistic_regression.md): Supervised · Classification (linear) ([Notebook](../notebooks/05_logistic_regression.ipynb))
- [06 · The Perceptron](06_perceptron.md): Supervised · Classification (linear), the ancestor of neural networks ([Notebook](../notebooks/06_perceptron.ipynb))
- [07 · k-Nearest Neighbors (k-NN)](07_k_nearest_neighbors.md): Supervised · Classification / Regression (instance-based, non-parametric) ([Notebook](../notebooks/07_k_nearest_neighbors.ipynb))
- [08 · Naive Bayes](08_naive_bayes.md): Supervised · Classification (probabilistic, generative) ([Notebook](../notebooks/08_naive_bayes.ipynb))
- [09 · Linear & Quadratic Discriminant Analysis (LDA / QDA)](09_discriminant_analysis_lda_qda.md): Supervised · Classification (generative, Gaussian class models) ([Notebook](../notebooks/09_discriminant_analysis_lda_qda.ipynb))
- [10 · Support Vector Machines (SVM) & the Kernel Trick](10_support_vector_machine.md): Supervised · Classification (margin-based) ([Notebook](../notebooks/10_support_vector_machine.ipynb))
- [11 · Decision Trees (CART)](11_decision_tree.md): Supervised · Classification & Regression (non-linear, rule-based) ([Notebook](../notebooks/11_decision_tree.ipynb))
- [12 · Random Forest (Bagging)](12_random_forest.md): Supervised · Classification & Regression (ensemble, variance reduction) ([Notebook](../notebooks/12_random_forest.ipynb))
- [13 · AdaBoost](13_adaboost.md): Supervised · Classification (ensemble, boosting) ([Notebook](../notebooks/13_adaboost.ipynb))
- [14 · Gradient Boosting (GBM / XGBoost / LightGBM idea)](14_gradient_boosting.md): Supervised · Regression & Classification (ensemble, boosting) ([Notebook](../notebooks/14_gradient_boosting.ipynb))

## Part 2 · Neural networks & deep learning

- [15 · Neural Networks (MLP) & Backpropagation](15_neural_network_backprop.md): Supervised · Deep learning (multi-layer perceptron) ([Notebook](../notebooks/15_neural_network_backprop.ipynb))
- [16 · Convolutional Neural Networks (CNN)](16_convolutional_neural_network.md): Supervised · Deep learning for images / grids ([Notebook](../notebooks/16_convolutional_neural_network.ipynb))
- [17 · Recurrent Neural Networks (RNN) & Backprop-Through-Time](17_recurrent_neural_network.md): Supervised · Deep learning for sequences ([Notebook](../notebooks/17_recurrent_neural_network.ipynb))
- [18 · LSTM & GRU: Gated Memory](18_lstm_gru_gated_memory.md): Supervised · Deep learning for sequences ([Notebook](../notebooks/18_lstm_gru_gated_memory.ipynb))
- [19 · Attention & the Transformer](19_attention_transformer.md): Supervised · Deep learning (sequence → sequence, language, vision) ([Notebook](../notebooks/19_attention_transformer.ipynb))

## Part 3 · Generative models & representation learning

- [20 · Autoencoders](20_autoencoder.md): Unsupervised · Representation learning / dimensionality reduction ([Notebook](../notebooks/20_autoencoder.ipynb))
- [21 · Variational Autoencoder (VAE)](21_variational_autoencoder.md): Unsupervised · Generative model ([Notebook](../notebooks/21_variational_autoencoder.ipynb))
- [22 · Generative Adversarial Networks (GAN)](22_generative_adversarial_network.md): Unsupervised · Generative model (adversarial training) ([Notebook](../notebooks/22_generative_adversarial_network.ipynb))

## Part 4 · Unsupervised learning

- [23 · K-Means Clustering](23_k_means.md): Unsupervised · Clustering (centroid-based) ([Notebook](../notebooks/23_k_means.ipynb))
- [24 · Hierarchical (Agglomerative) Clustering](24_hierarchical_clustering.md): Unsupervised · Clustering (tree-based) ([Notebook](../notebooks/24_hierarchical_clustering.ipynb))
- [25 · DBSCAN (Density-Based Clustering)](25_dbscan.md): Unsupervised · Clustering (density-based) + outlier detection ([Notebook](../notebooks/25_dbscan.ipynb))
- [26 · Gaussian Mixture Models & the EM Algorithm](26_gaussian_mixture_em.md): Unsupervised · Clustering / density estimation (probabilistic) ([Notebook](../notebooks/26_gaussian_mixture_em.ipynb))
- [27 · Principal Component Analysis (PCA)](27_pca.md): Unsupervised · Dimensionality reduction / feature extraction ([Notebook](../notebooks/27_pca.ipynb))
- [28 · t-SNE (Non-linear Embedding for Visualization)](28_tsne.md): Unsupervised · Dimensionality reduction (visualization) ([Notebook](../notebooks/28_tsne.ipynb))
- [29 · Self-Organizing Map (SOM / Kohonen map)](29_self_organizing_map.md): Unsupervised · Neural clustering & topology-preserving visualization ([Notebook](../notebooks/29_self_organizing_map.ipynb))
- [30 · Isolation Forest (Anomaly Detection)](30_isolation_forest_anomaly_detection.md): Unsupervised · Anomaly / outlier detection ([Notebook](../notebooks/30_isolation_forest_anomaly_detection.ipynb))
- [31 · Association Rules & the Apriori Algorithm](31_apriori_association_rules.md): Unsupervised · Pattern mining (market-basket analysis) ([Notebook](../notebooks/31_apriori_association_rules.ipynb))

## Part 5 · Sequences, probability, recommenders, forecasting & RL

- [32 · Hidden Markov Models (HMM): Viterbi & Baum-Welch](32_hidden_markov_model.md): Probabilistic sequence models (supervised or unsupervised) ([Notebook](../notebooks/32_hidden_markov_model.ipynb))
- [33 · Gaussian Processes (GP) Regression](33_gaussian_process.md): Supervised · Bayesian non-parametric regression ([Notebook](../notebooks/33_gaussian_process.ipynb))
- [34 · Matrix Factorization & Recommender Systems](34_matrix_factorization_recommender.md): Unsupervised/supervised hybrid · Collaborative filtering ([Notebook](../notebooks/34_matrix_factorization_recommender.ipynb))
- [35 · Time-Series Forecasting: Smoothing, Holt-Winters & Autoregression](35_time_series_forecasting.md): Supervised · Forecasting ([Notebook](../notebooks/35_time_series_forecasting.ipynb))
- [36 · Reinforcement Learning: Q-Learning & Bandits](36_reinforcement_learning.md): Reinforcement learning (learning from rewards by interaction) ([Notebook](../notebooks/36_reinforcement_learning.ipynb))
