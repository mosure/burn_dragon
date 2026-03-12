use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon::api::core::config::BDHConfig;
use burn_dragon::api::core::recurrent::BDH;
use burn_ndarray::NdArray;

fn main() {
    type Backend = NdArray<f32>;

    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let mut config = BDHConfig {
        n_layer: 2,
        n_embd: 32,
        n_head: 4,
        vocab_size: 128,
        ..Default::default()
    };
    config.set_rollout_fast_steps_per_slow_step(1);

    let model = BDH::<Backend>::new(config, &device);
    let token_ids = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
        &device,
    );

    let logits = model.forward(token_ids);
    println!("logits shape: {:?}", logits.shape().dims::<3>());
}
