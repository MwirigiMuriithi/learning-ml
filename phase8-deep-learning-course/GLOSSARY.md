# 📖 Glossary (plain-English definitions)

**Activation function**: a non-linear function applied to a neuron's weighted sum (ReLU, sigmoid, tanh, GELU, softmax). Without it, many layers collapse into one linear map.
**Adam / AdamW**: optimiser combining momentum with per-parameter adaptive step sizes; AdamW applies weight decay separately.
**Attention**: mechanism where each token takes a weighted average of all tokens' *values*, with weights from query–key similarity.
**Augmentation (data)**: creating modified training examples (flips, shifts, rotations) that keep the label; applied to training data only.
**Backpropagation**: applying the chain rule layer by layer from the loss back to every weight, reusing cached forward-pass values.
**Batch / mini-batch**: the group of examples used for one gradient estimate and weight update.
**Batch normalisation**: normalises each feature over the batch to mean 0 / variance 1, then rescales with learned γ and β; uses running averages at eval time.
**Bias (parameter)**: a learned offset added to a neuron's weighted sum; controls how easily it fires.
**Bias (statistical) / inductive bias**: assumptions built into a model (e.g. CNNs assume local, shift-invariant patterns).
**BPTT**: backpropagation through time: unrolling a recurrent network across time steps and backpropagating.
**Callback**: a hook run during training (early stopping, LR reduction, checkpointing).
**Calibration**: whether predicted confidences match real frequencies (80 % confident ⇒ right ~80 % of the time).
**Channel**: one 2-D layer of an image or feature map (RGB = 3 channels).
**CNN**: convolutional neural network: layers of small shared filters sliding over the input.
**Confusion matrix**: table of true class (rows) vs predicted class (columns).
**Convolution (cross-correlation)**: sliding a small kernel over an input, multiplying and summing at each position.
**Coverage (prediction interval)**: fraction of true values that actually fall inside the predicted interval.
**Cross-entropy**: loss = −log(probability assigned to the true class); the standard classification loss.
**Data leakage**: information from validation/test/future data sneaking into training or preprocessing, inflating scores.
**Dropout**: randomly zeroing activations during training (scaled by 1/(1−p)) to prevent co-adaptation and overfitting.
**Early stopping**: stopping when validation loss stops improving and restoring the best weights.
**Embedding**: a learned vector representing a discrete item (word, patch, token).
**Ensemble**: averaging the predictions of several models to reduce variance.
**Epoch**: one full pass over the training set.
**Exploding / vanishing gradients**: gradients that grow or shrink exponentially with depth or time steps, destabilising or stalling learning.
**F1 score**: harmonic mean of precision and recall.
**Feature map**: the output of a convolutional filter applied to an input (shows where a feature occurs).
**Forward propagation**: computing predictions by passing inputs through the network layer by layer.
**GRU**: gated recurrent unit: a recurrent cell with update and reset gates (lighter than an LSTM).
**Gradient**: vector of partial derivatives of the loss with respect to the parameters; points uphill.
**Gradient clipping**: rescaling gradients whose norm exceeds a threshold to prevent explosions.
**Gradient descent**: repeatedly stepping parameters against the gradient: w ← w − η∇L.
**Hidden layer / state**: a layer between input and output / a recurrent network's running memory vector.
**Huber loss**: quadratic for small errors, linear for large ones; robust to outliers.
**Hyperparameter**: a setting you choose before training (learning rate, batch size, layers); *not* learned.
**Inference**: using a trained model to make predictions.
**Kernel / filter**: the small grid of weights in a convolution.
**L1 / L2 regularisation**: penalties on |w| (promotes sparsity) or w² (shrinks weights; "weight decay").
**LayerNorm**: normalises across a token's features (used in Transformers).
**Learning rate (η)**: step size of gradient descent; the most important hyperparameter.
**Logit**: raw, un-normalised output score before softmax/sigmoid.
**Loss function**: one number measuring how wrong predictions are; what training minimises.
**LSTM**: long short-term memory: recurrent cell with forget/input/output gates and an additive cell state.
**MAE / RMSE / MAPE**: mean absolute error / root-mean-squared error / mean absolute percentage error.
**Momentum**: optimiser trick accumulating a velocity so updates keep moving along consistent directions.
**Multi-head attention**: several attention operations in parallel in lower-dimensional subspaces, concatenated.
**Naive forecast**: predicting that the future equals the present; the baseline any forecaster must beat.
**Neuron**: computes a weighted sum of inputs plus a bias, then applies an activation.
**Occlusion sensitivity**: measuring how predictions change when image regions are hidden; shows what the model uses.
**One-hot encoding**: representing class *k* as a vector with 1 at position *k* and 0 elsewhere.
**Overfitting**: memorising training specifics (or noise) so performance on new data is worse.
**Padding**: adding border pixels (usually zeros) so convolutions preserve size and see edges.
**Parameter**: a learned number (weight or bias).
**Perceptron**: a single neuron with a step activation and the perceptron learning rule.
**Pooling**: downsampling by taking the max or average in each window.
**Positional encoding**: position information added to token embeddings, since attention ignores order.
**Precision / recall**: of the items called class *c*, the fraction correct / of all true *c*, the fraction found.
**Prediction interval**: a range expected to contain the future value with stated probability.
**Query / key / value**: what a token looks for / offers for matching / contributes in attention.
**ReLU**: max(0, z); the default hidden-layer activation.
**Receptive field**: the region of the input that can influence a given neuron.
**Regularisation**: any technique that reduces overfitting.
**Residual (skip) connection**: adding a block's input to its output (x + f(x)) so gradients flow easily.
**RNN**: recurrent neural network: processes sequences step by step, carrying a hidden state.
**Softmax**: turns logits into probabilities that sum to 1.
**Stride**: step size of a convolution or pooling window.
**Test set**: data held back until the very end for one honest performance estimate.
**Transfer learning**: reusing a model pre-trained on a large dataset and fine-tuning it on yours.
**Transformer**: architecture built from attention + feed-forward blocks with residuals and LayerNorm.
**Underfitting**: model too simple or trained too little; poor on training *and* validation data.
**Validation set**: data used for choices (hyperparameters, early stopping, model selection).
**Weight**: a learned multiplier on an input; encodes importance and direction.
**Weight decay**: shrinking weights a little each step (equivalent to L2 for plain SGD).
**Weight sharing**: reusing the same weights at many positions (convolution) or time steps (RNN).
