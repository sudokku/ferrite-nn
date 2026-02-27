# ferrite-nn 🧠

> A neural network library, hand-rolled in Rust — because the best way to understand
> the math is to write every gradient yourself.

No PyTorch. No TensorFlow. No magic black boxes. Just matrices, dot products, and
backpropagation, all the way down.

`ferrite-nn` is a from-scratch MLP library built for the joy of building things. It
is structured as a proper Rust library crate and comes with working examples that
train real models — including a digit recognizer that hits ~97% accuracy on MNIST.

---

## What you can do 🚀

| Example | What it does | Command |
|---------|-------------|---------|
| `xor` | Teaches a tiny network to learn XOR | `cargo run --example xor` |
| `mnist` | Trains a digit recognizer on MNIST (~97% accuracy) | `cargo run --example mnist --release` |
| `gui` | Launches a local web app for running inference | `cargo run --example gui --release` |

---

## Getting started

You will need Rust 1.75 or newer.

```sh
git clone https://github.com/your-username/ferrite-nn
cd ferrite-nn
```

### Run the XOR demo

No setup needed — just run it:

```sh
cargo run --example xor
```

The network converges after ~10 000 epochs. You will see outputs near `0.95` and
`0.05`, which is as close to true XOR as a sigmoid can get.

### Train on MNIST

Download the four IDX binary files from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
and place them in `examples/mnist_data/`:

```
examples/mnist_data/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

Then train:

```sh
cargo run --example mnist --release
```

Training prints a baseline accuracy (~10%, random chance), then per-epoch loss and
training accuracy. After 50 epochs the trained model is saved to
`examples/trained_models/mnist.json` and you will see ~97% test accuracy.

### Use ferrite-nn as a library

Add it as a local path dependency:

```toml
[dependencies]
ferrite-nn = { path = "../ferrite-nn" }
```

A minimal training loop:

```rust
use ferrite_nn::{Network, Sgd, ActivationFunction, train_network};

fn main() {
    let mut network = Network::new(vec![
        (4, 2, ActivationFunction::Sigmoid),  // hidden: 4 neurons, 2 inputs
        (1, 4, ActivationFunction::Sigmoid),  // output: 1 neuron, 4 inputs
    ]);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let expected = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let optimizer = Sgd::new(0.1);

    for epoch in 0..10_000 {
        let loss = train_network(&mut network, &inputs, &expected, &optimizer, 1);
        if epoch % 1000 == 0 {
            println!("Epoch {epoch}: loss = {loss:.6}");
        }
    }

    for input in &inputs {
        let output = network.forward(input.clone());
        println!("{input:?} -> {:.4}", output[0]);
    }
}
```

---

## Architecture 🗂️

```
src/
  lib.rs                 -- crate root; re-exports everything public
  math/
    matrix.rs            -- Matrix: zeros, he, xavier, random, transpose, map, +, -, *
  activation/
    activation.rs        -- ActivationFunction: Sigmoid, ReLU, Identity, Softmax
  layers/
    dense.rs             -- Layer: new(), feed_from(), compute_gradients(), apply_gradients()
  network/
    network.rs           -- Network: new(), forward(), save_json(), load_json()
  loss/
    mse.rs               -- MseLoss: loss(), derivative()
    cross_entropy.rs     -- CrossEntropyLoss: numerically-stable CE paired with Softmax
  optim/
    sgd.rs               -- Sgd: new(lr), step()
  train/
    trainer.rs           -- train_network(): mini-batch SGD training loop
  main.rs                -- thin binary entry point
examples/
  xor.rs                 -- XOR gate demo
  mnist.rs               -- MNIST digit classifier (saves model to JSON)
  gui.rs                 -- local web inference server
```

Weight initialization is automatic: He for ReLU layers, Xavier for everything else.

Dependencies:
- `rand 0.8` — weight initialization
- `serde` + `serde_json` — model serialization
- `tiny_http 0.12` — web GUI server (dev dependency, not compiled into the library)

---

## Web inference GUI 🌐

After you have trained a model, you can run inference on it through a browser:

```sh
cargo run --example gui --release
```

Then open `http://127.0.0.1:7878` in your browser. Pick a model from
`examples/trained_models/`, enter comma-separated input values, and click
"Run inference". The output is auto-formatted depending on the network's final
activation:

- **Softmax** — shows predicted class and a confidence bar for each output
- **Sigmoid (single output)** — shows a probability
- **Identity** — shows raw values

Train the MNIST model first to get the most out of the GUI.

---

## Training your own model

Construct a `Network` from a list of `(output_size, input_size, activation)` layer
specs, then call `train_network`:

```rust
use ferrite_nn::{Network, Sgd, ActivationFunction, train_network};

// A three-layer classifier: 784 inputs -> 256 -> 128 -> 10 classes
let mut net = Network::new(vec![
    (256, 784, ActivationFunction::ReLU),
    (128, 256, ActivationFunction::ReLU),
    (10,  128, ActivationFunction::Softmax),
]);

let optimizer = Sgd::new(0.01);

for epoch in 0..50 {
    let loss = train_network(&mut net, &inputs, &labels, &optimizer, 32);
    println!("Epoch {epoch}: CE loss = {loss:.4}");
}

// Save for later
net.save_json("examples/trained_models/my_model.json").unwrap();
```

Reload it at any time:

```rust
let mut net = Network::load_json("examples/trained_models/my_model.json").unwrap();
let output = net.forward(my_input);
```

---

## Future plans 🔭

ferrite-nn is intentionally simple right now — that is the point. But there is plenty
of room to grow:

- **More optimizers** — Adam, RMSProp, momentum SGD
- **Batch normalization** — stabilize deeper networks
- **Convolutional layers** — image-native feature extraction
- **More weight init strategies** — LeCun init, orthogonal init
- **WASM inference** — run saved models in the browser (no server needed)
- **Python bindings** — expose ferrite-nn via PyO3 for quick scripting
- **More example datasets** — FashionMNIST, CIFAR-10, XOR variants, toy regression
- **Visualization** — loss curves, weight histograms, maybe a training dashboard

Contributions welcome — see `.claude/agents/` for subagent descriptions if you want
to understand how this repo is maintained.

---

## License

MIT
