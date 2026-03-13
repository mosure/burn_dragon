use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Clone)]
pub struct VlJepaLossBreakdown<B: Backend> {
    pub total: Tensor<B, 1>,
    pub predictor_to_target: Tensor<B, 1>,
    pub target_to_predictor: Tensor<B, 1>,
    pub similarities: Tensor<B, 2>,
}

fn gather_diagonal_log_probs<B: Backend>(log_probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, _] = log_probs.shape().dims::<2>();
    let device = log_probs.device();
    let labels = Tensor::<B, 1, Int>::arange(0..batch as i64, &device).reshape([batch, 1]);
    log_probs.gather(1, labels).neg().mean()
}

fn gather_label_log_probs<B: Backend>(log_probs: Tensor<B, 2>, labels: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let [batch] = labels.shape().dims::<1>();
    log_probs.gather(1, labels.reshape([batch, 1])).neg().mean()
}

fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let squared = x.clone().powf_scalar(2.0).sum_dim(1);
    let [batch, _dim] = x.shape().dims::<2>();
    let norm = squared.sqrt().reshape([batch, 1]).clamp_min(1e-6);
    x / norm
}

pub fn vl_jepa_bidirectional_info_nce_loss<B: Backend>(
    predicted_target_embedding: Tensor<B, 2>,
    target_embedding_y: Tensor<B, 2>,
    temperature: f32,
) -> VlJepaLossBreakdown<B> {
    let predicted = l2_normalize(predicted_target_embedding);
    let target = l2_normalize(target_embedding_y);
    let similarities = predicted.matmul(target.clone().swap_dims(0, 1)) / temperature.max(1e-6);
    let predictor_to_target = gather_diagonal_log_probs(activation::log_softmax(
        similarities.clone(),
        1,
    ));
    let target_to_predictor = gather_diagonal_log_probs(activation::log_softmax(
        similarities.clone().swap_dims(0, 1),
        1,
    ));
    let total = (predictor_to_target.clone() + target_to_predictor.clone()) / 2.0;
    VlJepaLossBreakdown {
        total,
        predictor_to_target,
        target_to_predictor,
        similarities,
    }
}

pub fn vl_jepa_teacher_student_info_nce_loss<B: Backend>(
    predicted_target_embedding: Tensor<B, 2>,
    student_target_embedding_y: Tensor<B, 2>,
    teacher_target_embedding_y: Tensor<B, 2>,
    temperature: f32,
) -> VlJepaLossBreakdown<B> {
    let predicted = l2_normalize(predicted_target_embedding);
    let student_target = l2_normalize(student_target_embedding_y);
    let teacher_target = l2_normalize(teacher_target_embedding_y);
    let similarities =
        predicted.clone().matmul(teacher_target.clone().swap_dims(0, 1)) / temperature.max(1e-6);
    let predictor_to_target = gather_diagonal_log_probs(activation::log_softmax(
        similarities.clone(),
        1,
    ));
    let target_to_predictor = gather_diagonal_log_probs(activation::log_softmax(
        student_target.matmul(predicted.detach().swap_dims(0, 1)) / temperature.max(1e-6),
        1,
    ));
    let total = (predictor_to_target.clone() + target_to_predictor.clone()) / 2.0;
    VlJepaLossBreakdown {
        total,
        predictor_to_target,
        target_to_predictor,
        similarities,
    }
}

pub fn vl_jepa_target_bank_loss<B: Backend>(
    predicted_target_embedding: Tensor<B, 2>,
    target_bank_embeddings: Tensor<B, 2>,
    target_indices: Tensor<B, 1, Int>,
    temperature: f32,
) -> VlJepaLossBreakdown<B> {
    let predicted = l2_normalize(predicted_target_embedding);
    let target_bank = l2_normalize(target_bank_embeddings);
    let similarities = predicted.matmul(target_bank.swap_dims(0, 1)) / temperature.max(1e-6);
    let predictor_to_target =
        gather_label_log_probs(activation::log_softmax(similarities.clone(), 1), target_indices);
    let target_to_predictor = predictor_to_target.clone();
    let total = predictor_to_target.clone();
    VlJepaLossBreakdown {
        total,
        predictor_to_target,
        target_to_predictor,
        similarities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn bidirectional_info_nce_returns_scalar_losses() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let predicted = Tensor::<Backend, 2>::ones([3, 4], &device);
        let target = Tensor::<Backend, 2>::ones([3, 4], &device);
        let loss = vl_jepa_bidirectional_info_nce_loss(predicted, target, 0.07);
        assert_eq!(loss.total.shape().dims(), [1]);
        assert_eq!(loss.similarities.shape().dims(), [3, 3]);
    }

    #[test]
    fn teacher_student_info_nce_returns_scalar_losses() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let predicted = Tensor::<Backend, 2>::ones([3, 4], &device);
        let student = Tensor::<Backend, 2>::ones([3, 4], &device);
        let teacher = Tensor::<Backend, 2>::ones([3, 4], &device);
        let loss = vl_jepa_teacher_student_info_nce_loss(predicted, student, teacher, 0.07);
        assert_eq!(loss.total.shape().dims(), [1]);
        assert_eq!(loss.similarities.shape().dims(), [3, 3]);
    }

    #[test]
    fn target_bank_loss_returns_scalar_losses() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let predicted = Tensor::<Backend, 2>::ones([3, 4], &device);
        let bank = Tensor::<Backend, 2>::ones([5, 4], &device);
        let labels = Tensor::<Backend, 1, Int>::from_data([0_i64, 1, 2], &device);
        let loss = vl_jepa_target_bank_loss(predicted, bank, labels, 0.07);
        assert_eq!(loss.total.shape().dims(), [1]);
        assert_eq!(loss.similarities.shape().dims(), [3, 5]);
    }
}
