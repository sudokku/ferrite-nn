# Architecture Analysis Notes

## Layer Forward Pass Detail
`Layer::feed_from(input: Vec<f64>) -> Vec<f64>`
- Creates a 1-row matrix from input: shape [1, input_size]
- Computes z = input_matrix * weights + biases  (shapes: [1,in] * [in,out] + [1,out] = [1,out])
- Applies activation element-wise via .map()
- Stores z in pre_neurons, a in neurons
- Returns a.data[0] (the single output row as Vec<f64>)

## Backprop Detail
Trainer propagates delta = ∂L/∂a backward through layers:
- For each layer i (reversed): compute_gradients(delta, inputs) -> (w_grad, b_grad)
- compute_gradients: layer_delta = delta ⊙ σ'(pre_neurons), w_grad = inputs.T * layer_delta, b_grad = layer_delta
- New delta for layer i-1: b_grad * weights[i].T (propagates through weight matrix)
- apply_gradients: weights -= lr * w_grad, biases -= lr * b_grad

## Softmax Cannot Use map()
Softmax(x_i) = exp(x_i) / sum(exp(x_j)) — depends on all elements of the row simultaneously.
The current .map(|x| activator.function(x)) is element-wise and cannot implement this.
Fix requires: either a row-wise method on ActivationFunction, or special-casing in Layer::feed_from.
The Softmax Jacobian in backprop is also non-trivial; the standard solution is to fuse it with
cross-entropy loss, giving gradient = softmax_output - one_hot_target (bypasses the Jacobian).

## Initialization Problem at Scale
Matrix::random produces uniform [-1,1]. For a layer with fan_in=784:
- Variance of each weight: 1/3 (uniform [-1,1])
- Variance of dot product z = sum(w_i * x_i): fan_in * Var(w) * Var(x)
- For normalized MNIST pixels (x in [0,1], Var(x) ~ 0.08): 784 * (1/3) * 0.08 ~ 20.9
- StdDev(z) ~ 4.6, meaning most z values will be in ±5 range -> sigmoid saturates badly
- He init would use stddev = sqrt(2/784) ~ 0.05, far smaller

## Trainer Hardcodes MseLoss
In train/trainer.rs lines 21 and 24, MseLoss is called directly — not via a trait or parameter.
To make loss injectable: either define a Loss trait and pass Box<dyn Loss>, or add a second
train_network_with_loss function. This is needed for cross-entropy.
