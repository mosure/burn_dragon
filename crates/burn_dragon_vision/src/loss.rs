use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

const DISTILL_EPS: f32 = 1e-6;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct VisionDistillationLossConfig {
    pub patch_mse_weight: f32,
    pub cls_mse_weight: f32,
    pub cls_cosine_weight: f32,
    pub rel_weight: f32,
    pub rel_tau: f32,
    pub rel_sample_tokens: Option<usize>,
}

impl Default for VisionDistillationLossConfig {
    fn default() -> Self {
        Self {
            patch_mse_weight: 1.0,
            cls_mse_weight: 0.0,
            cls_cosine_weight: 1.0,
            rel_weight: 0.0,
            rel_tau: 0.07,
            rel_sample_tokens: None,
        }
    }
}

impl<B: Backend> Module<B> for VisionDistillationLossConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionDistillationLossConfig {
    type InnerModule = VisionDistillationLossConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for VisionDistillationLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("patch_mse_weight", &self.patch_mse_weight)
            .add("cls_mse_weight", &self.cls_mse_weight)
            .add("cls_cosine_weight", &self.cls_cosine_weight)
            .add("rel_weight", &self.rel_weight)
            .add("rel_tau", &self.rel_tau)
            .add("rel_sample_tokens", &self.rel_sample_tokens)
            .optional()
    }
}

impl ModuleDisplay for VisionDistillationLossConfig {}

pub fn vision_distillation_loss<B: Backend>(
    student_patch: Tensor<B, 3>,
    teacher_patch: Tensor<B, 3>,
    student_cls: Tensor<B, 2>,
    teacher_cls: Tensor<B, 2>,
    config: &VisionDistillationLossConfig,
) -> Tensor<B, 1> {
    let device = student_patch.device();
    let mut total = Tensor::<B, 1>::zeros([1], &device);

    if config.patch_mse_weight > 0.0 {
        let student = feature_layer_norm(student_patch.clone());
        let teacher = feature_layer_norm(teacher_patch.clone().detach());
        let mse = (student - teacher).powf_scalar(2.0).mean();
        total = total + mse.mul_scalar(config.patch_mse_weight);
    }

    if config.cls_mse_weight > 0.0 {
        let student = feature_layer_norm(student_cls.clone());
        let teacher = feature_layer_norm(teacher_cls.clone().detach());
        let mse = (student - teacher).powf_scalar(2.0).mean();
        total = total + mse.mul_scalar(config.cls_mse_weight);
    }

    if config.cls_cosine_weight > 0.0 {
        let student = l2_normalize(student_cls.clone());
        let teacher = l2_normalize(teacher_cls.clone().detach());
        let cosine = student.mul(teacher).sum_dim(1);
        let loss = cosine.mul_scalar(-1.0).add_scalar(1.0).mean();
        total = total + loss.mul_scalar(config.cls_cosine_weight);
    }

    if config.rel_weight > 0.0 {
        let student = maybe_sample_tokens(student_patch, config.rel_sample_tokens);
        let teacher = maybe_sample_tokens(teacher_patch.detach(), config.rel_sample_tokens);

        let student = feature_layer_norm(student);
        let teacher = feature_layer_norm(teacher);

        let sim_student = student
            .clone()
            .matmul(student.swap_dims(1, 2))
            .div_scalar(config.rel_tau);
        let sim_teacher = teacher
            .clone()
            .matmul(teacher.swap_dims(1, 2))
            .div_scalar(config.rel_tau);

        let log_student = activation::log_softmax(sim_student, 2);
        let teacher_prob = activation::softmax(sim_teacher, 2);
        let teacher_log = teacher_prob.clone().add_scalar(DISTILL_EPS).log();
        let kl = (teacher_prob * (teacher_log - log_student))
            .sum_dim(2)
            .mean();
        total = total + kl.mul_scalar(config.rel_weight);
    }

    total
}

fn feature_layer_norm<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let (var, mean) = tensor.clone().var_mean_bias(D - 1);
    tensor.sub(mean).div(var.add_scalar(1e-5).sqrt())
}

fn l2_normalize<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let norm = tensor
        .clone()
        .powf_scalar(2.0)
        .sum_dim(D - 1)
        .sqrt()
        .add_scalar(DISTILL_EPS);
    tensor / norm
}

fn maybe_sample_tokens<B: Backend>(tokens: Tensor<B, 3>, sample: Option<usize>) -> Tensor<B, 3> {
    if let Some(count) = sample {
        let time = tokens.shape().dims::<3>()[1];
        let limit = count.min(time).max(1);
        tokens.slice_dim(1, 0..limit)
    } else {
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn vision_distillation_loss_is_finite() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let student_patch =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let teacher_patch =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let student_cls =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);
        let teacher_cls =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);

        let config = VisionDistillationLossConfig::default();
        let loss = vision_distillation_loss(
            student_patch,
            teacher_patch,
            student_cls,
            teacher_cls,
            &config,
        );
        let value = loss
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss to vec")[0];
        assert!(value.is_finite());
    }
}
