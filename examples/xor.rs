use ferrite_nn::{Network, Sgd, ActivationFunction, train_network};

fn main() {
    let mut network = Network::new(vec![
        (2, 2, ActivationFunction::Sigmoid),
        (1, 2, ActivationFunction::Sigmoid),
    ]);

    let inputs = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
    ];
    let expected_outputs = vec![
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
    ];

    let optimizer = Sgd::new(0.1);
    let epochs = 10000;

    for epoch in 0..epochs {
        let loss = train_network(&mut network, &inputs, &expected_outputs, &optimizer);
        if epoch % 1000 == 0 {
            println!("Epoch {epoch}: loss = {loss:.6}");
        }
    }

    for input in &inputs {
        println!("Input: {:?} -> Output: {:.4}", input, network.forward(input.clone())[0]);
    }
}
