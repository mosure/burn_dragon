use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
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

#[derive(Clone)]
pub struct VisionDistillationLossTerms<B: Backend> {
    pub total: Tensor<B, 1>,
    pub patch: Tensor<B, 1>,
    pub cls: Tensor<B, 1>,
    pub relational: Tensor<B, 1>,
}

#[derive(Clone)]
pub struct WeightedPatchDistillTarget<B: Backend> {
    pub student: Tensor<B, 3>,
    pub teacher: Tensor<B, 3>,
    pub weight: f32,
}

#[derive(Clone)]
pub struct WeightedClsDistillTarget<B: Backend> {
    pub student: Tensor<B, 2>,
    pub teacher: Tensor<B, 2>,
    pub weight: f32,
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

    fn from_inner(module: Self::InnerModule) -> Self {
        module
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
    vision_distillation_loss_terms(
        student_patch,
        teacher_patch,
        student_cls,
        teacher_cls,
        config,
    )
    .total
}

pub fn vision_distillation_loss_terms<B: Backend>(
    student_patch: Tensor<B, 3>,
    teacher_patch: Tensor<B, 3>,
    student_cls: Tensor<B, 2>,
    teacher_cls: Tensor<B, 2>,
    config: &VisionDistillationLossConfig,
) -> VisionDistillationLossTerms<B> {
    let device = student_patch.device();
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let mut patch_total = Tensor::<B, 1>::zeros([1], &device);
    let mut cls_total = Tensor::<B, 1>::zeros([1], &device);
    let mut relational_total = Tensor::<B, 1>::zeros([1], &device);

    if config.patch_mse_weight > 0.0 {
        let student = feature_layer_norm(student_patch.clone());
        let teacher = feature_layer_norm(teacher_patch.clone().detach());
        let mse = (student - teacher).powf_scalar(2.0).mean();
        let weighted = mse.mul_scalar(config.patch_mse_weight);
        patch_total = patch_total + weighted.clone();
        total = total + weighted;
    }

    if config.cls_mse_weight > 0.0 {
        let student = feature_layer_norm(student_cls.clone());
        let teacher = feature_layer_norm(teacher_cls.clone().detach());
        let mse = (student - teacher).powf_scalar(2.0).mean();
        let weighted = mse.mul_scalar(config.cls_mse_weight);
        cls_total = cls_total + weighted.clone();
        total = total + weighted;
    }

    if config.cls_cosine_weight > 0.0 {
        let student = l2_normalize(student_cls.clone());
        let teacher = l2_normalize(teacher_cls.clone().detach());
        let cosine = student.mul(teacher).sum_dim(1);
        let loss = cosine.mul_scalar(-1.0).add_scalar(1.0).mean();
        let weighted = loss.mul_scalar(config.cls_cosine_weight);
        cls_total = cls_total + weighted.clone();
        total = total + weighted;
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
        let weighted = kl.mul_scalar(config.rel_weight);
        relational_total = relational_total + weighted.clone();
        total = total + weighted;
    }

    VisionDistillationLossTerms {
        total,
        patch: patch_total,
        cls: cls_total,
        relational: relational_total,
    }
}

pub fn weighted_patch_mse_loss<B: Backend>(
    targets: &[WeightedPatchDistillTarget<B>],
) -> Tensor<B, 1> {
    let Some(first) = targets.first() else {
        panic!("weighted_patch_mse_loss requires at least one target");
    };
    let device = first.student.device();
    let mut students = Vec::with_capacity(targets.len());
    let mut teachers = Vec::with_capacity(targets.len());
    let mut sample_weights = Vec::new();

    for target in targets {
        let [batch, student_tokens, student_dim] = target.student.shape().dims::<3>();
        let [teacher_batch, teacher_tokens, teacher_dim] = target.teacher.shape().dims::<3>();
        assert_eq!(
            batch, teacher_batch,
            "patch distill groups require matching batch size"
        );
        assert_eq!(
            (student_tokens, student_dim),
            (teacher_tokens, teacher_dim),
            "patch distill groups require matching token and feature shapes"
        );
        students.push(target.student.clone());
        teachers.push(target.teacher.clone());
        let per_sample_weight = target.weight / batch.max(1) as f32;
        sample_weights.extend(std::iter::repeat_n(per_sample_weight, batch));
    }

    let student = feature_layer_norm(Tensor::cat(students, 0));
    let teacher = feature_layer_norm(Tensor::cat(teachers, 0).detach());
    let total_batch = sample_weights.len();
    let per_sample = (student - teacher)
        .powf_scalar(2.0)
        .mean_dim(2)
        .mean_dim(1)
        .reshape([total_batch]);
    let weights =
        Tensor::<B, 1>::from_data(TensorData::new(sample_weights, [total_batch]), &device);
    per_sample.mul(weights).sum().reshape([1])
}

pub fn weighted_cls_mse_loss<B: Backend>(targets: &[WeightedClsDistillTarget<B>]) -> Tensor<B, 1> {
    let Some(first) = targets.first() else {
        panic!("weighted_cls_mse_loss requires at least one target");
    };
    let device = first.student.device();
    let mut students = Vec::with_capacity(targets.len());
    let mut teachers = Vec::with_capacity(targets.len());
    let mut sample_weights = Vec::new();

    for target in targets {
        let [batch, student_dim] = target.student.shape().dims::<2>();
        let [teacher_batch, teacher_dim] = target.teacher.shape().dims::<2>();
        assert_eq!(
            batch, teacher_batch,
            "cls distill groups require matching batch size"
        );
        assert_eq!(
            student_dim, teacher_dim,
            "cls distill groups require matching feature shapes"
        );
        students.push(target.student.clone());
        teachers.push(target.teacher.clone());
        let per_sample_weight = target.weight / batch.max(1) as f32;
        sample_weights.extend(std::iter::repeat_n(per_sample_weight, batch));
    }

    let student = feature_layer_norm(Tensor::cat(students, 0));
    let teacher = feature_layer_norm(Tensor::cat(teachers, 0).detach());
    let total_batch = sample_weights.len();
    let per_sample = (student - teacher)
        .powf_scalar(2.0)
        .mean_dim(1)
        .reshape([total_batch]);
    let weights =
        Tensor::<B, 1>::from_data(TensorData::new(sample_weights, [total_batch]), &device);
    per_sample.mul(weights).sum().reshape([1])
}

pub fn weighted_cls_cosine_loss<B: Backend>(
    targets: &[WeightedClsDistillTarget<B>],
) -> Tensor<B, 1> {
    let Some(first) = targets.first() else {
        panic!("weighted_cls_cosine_loss requires at least one target");
    };
    let device = first.student.device();
    let mut students = Vec::with_capacity(targets.len());
    let mut teachers = Vec::with_capacity(targets.len());
    let mut sample_weights = Vec::new();

    for target in targets {
        let [batch, student_dim] = target.student.shape().dims::<2>();
        let [teacher_batch, teacher_dim] = target.teacher.shape().dims::<2>();
        assert_eq!(
            batch, teacher_batch,
            "cls distill groups require matching batch size"
        );
        assert_eq!(
            student_dim, teacher_dim,
            "cls distill groups require matching feature shapes"
        );
        students.push(target.student.clone());
        teachers.push(target.teacher.clone());
        let per_sample_weight = target.weight / batch.max(1) as f32;
        sample_weights.extend(std::iter::repeat_n(per_sample_weight, batch));
    }

    let student = l2_normalize(Tensor::cat(students, 0));
    let teacher = l2_normalize(Tensor::cat(teachers, 0).detach());
    let total_batch = sample_weights.len();
    let per_sample = student
        .mul(teacher)
        .sum_dim(1)
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .reshape([total_batch]);
    let weights =
        Tensor::<B, 1>::from_data(TensorData::new(sample_weights, [total_batch]), &device);
    per_sample.mul(weights).sum().reshape([1])
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

    #[test]
    fn grouped_weighted_patch_and_cls_losses_match_individual_accumulation() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let student_patch_a =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let teacher_patch_a =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let student_patch_b =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let teacher_patch_b =
            Tensor::<Backend, 3>::random([2, 4, 8], burn::tensor::Distribution::Default, &device);
        let student_cls_a =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);
        let teacher_cls_a =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);
        let student_cls_b =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);
        let teacher_cls_b =
            Tensor::<Backend, 2>::random([2, 8], burn::tensor::Distribution::Default, &device);

        let patch_targets = vec![
            WeightedPatchDistillTarget {
                student: student_patch_a.clone(),
                teacher: teacher_patch_a.clone(),
                weight: 0.75,
            },
            WeightedPatchDistillTarget {
                student: student_patch_b.clone(),
                teacher: teacher_patch_b.clone(),
                weight: 0.25,
            },
        ];
        let cls_targets = vec![
            WeightedClsDistillTarget {
                student: student_cls_a.clone(),
                teacher: teacher_cls_a.clone(),
                weight: 0.75,
            },
            WeightedClsDistillTarget {
                student: student_cls_b.clone(),
                teacher: teacher_cls_b.clone(),
                weight: 0.25,
            },
        ];

        let patch_grouped = weighted_patch_mse_loss(&patch_targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("grouped patch")[0];
        let cls_mse_grouped = weighted_cls_mse_loss(&cls_targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("grouped cls mse")[0];
        let cls_cos_grouped = weighted_cls_cosine_loss(&cls_targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("grouped cls cosine")[0];

        let cfg_patch = VisionDistillationLossConfig {
            patch_mse_weight: 1.0,
            cls_mse_weight: 0.0,
            cls_cosine_weight: 0.0,
            rel_weight: 0.0,
            rel_tau: 0.07,
            rel_sample_tokens: None,
        };
        let cfg_cls_mse = VisionDistillationLossConfig {
            patch_mse_weight: 0.0,
            cls_mse_weight: 1.0,
            cls_cosine_weight: 0.0,
            rel_weight: 0.0,
            rel_tau: 0.07,
            rel_sample_tokens: None,
        };
        let cfg_cls_cos = VisionDistillationLossConfig {
            patch_mse_weight: 0.0,
            cls_mse_weight: 0.0,
            cls_cosine_weight: 1.0,
            rel_weight: 0.0,
            rel_tau: 0.07,
            rel_sample_tokens: None,
        };

        let patch_expected = 0.75
            * vision_distillation_loss_terms(
                student_patch_a,
                teacher_patch_a,
                student_cls_a.clone(),
                teacher_cls_a.clone(),
                &cfg_patch,
            )
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("patch a")[0]
            + 0.25
                * vision_distillation_loss_terms(
                    student_patch_b,
                    teacher_patch_b,
                    student_cls_b.clone(),
                    teacher_cls_b.clone(),
                    &cfg_patch,
                )
                .total
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("patch b")[0];
        let cls_mse_expected = 0.75
            * vision_distillation_loss_terms(
                Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                student_cls_a.clone(),
                teacher_cls_a.clone(),
                &cfg_cls_mse,
            )
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cls mse a")[0]
            + 0.25
                * vision_distillation_loss_terms(
                    Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                    Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                    student_cls_b.clone(),
                    teacher_cls_b.clone(),
                    &cfg_cls_mse,
                )
                .total
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("cls mse b")[0];
        let cls_cos_expected = 0.75
            * vision_distillation_loss_terms(
                Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                student_cls_a,
                teacher_cls_a,
                &cfg_cls_cos,
            )
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cls cos a")[0]
            + 0.25
                * vision_distillation_loss_terms(
                    Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                    Tensor::<Backend, 3>::zeros([2, 1, 8], &device),
                    student_cls_b,
                    teacher_cls_b,
                    &cfg_cls_cos,
                )
                .total
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("cls cos b")[0];

        assert!((patch_grouped - patch_expected).abs() < 1e-5);
        assert!((cls_mse_grouped - cls_mse_expected).abs() < 1e-5);
        assert!((cls_cos_grouped - cls_cos_expected).abs() < 1e-5);
    }
}
