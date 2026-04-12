use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DragonNormKind {
    #[default]
    LayerNorm,
    #[serde(alias = "rmsnorm")]
    RmsNorm,
    #[serde(alias = "dyt")]
    DynamicTanh,
    Derf,
}

impl core::fmt::Display for DragonNormKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for DragonNormKind {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for DragonNormKind {
    type InnerModule = DragonNormKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for DragonNormKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for DragonNormKind {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DragonNormConfig {
    #[serde(default)]
    pub kind: DragonNormKind,
    #[serde(default = "default_norm_epsilon")]
    pub eps: f32,
    #[serde(default)]
    pub alpha_init: Option<f32>,
    #[serde(default)]
    pub shift_init: Option<f32>,
}

impl Default for DragonNormConfig {
    fn default() -> Self {
        Self {
            kind: DragonNormKind::default(),
            eps: default_norm_epsilon(),
            alpha_init: None,
            shift_init: None,
        }
    }
}

const fn default_norm_epsilon() -> f32 {
    1e-5
}

const fn default_dyt_alpha_init() -> f32 {
    0.5
}

const fn default_derf_alpha_init() -> f32 {
    0.886_226_95
}

const fn default_norm_shift_init() -> f32 {
    0.0
}

impl DragonNormConfig {
    pub fn resolved_alpha_init(&self) -> f32 {
        self.alpha_init.unwrap_or(match self.kind {
            DragonNormKind::LayerNorm | DragonNormKind::RmsNorm => 1.0,
            DragonNormKind::DynamicTanh => default_dyt_alpha_init(),
            DragonNormKind::Derf => default_derf_alpha_init(),
        })
    }

    pub fn resolved_shift_init(&self) -> f32 {
        self.shift_init.unwrap_or(default_norm_shift_init())
    }
}

impl<B: Backend> Module<B> for DragonNormConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for DragonNormConfig {
    type InnerModule = DragonNormConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for DragonNormConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "kind={}, eps={}, alpha_init={}, shift_init={}",
            self.kind,
            self.eps,
            self.resolved_alpha_init(),
            self.resolved_shift_init()
        );
        content
            .set_top_level_type("DragonNormConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for DragonNormConfig {}

#[derive(Module, Debug)]
pub struct DragonNorm<B: Backend> {
    kind: DragonNormKind,
    #[module(skip)]
    eps: f32,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    alpha: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

impl<B: Backend> DragonNorm<B> {
    fn param_rms<const D: usize>(tensor: Tensor<B, D>) -> f32 {
        let values = tensor
            .powf_scalar(2.0)
            .mean()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("dragon norm rms scalar");
        values.first().copied().unwrap_or(0.0).sqrt()
    }

    fn blend_param<const D: usize>(
        source: Tensor<B, D>,
        fresh: Tensor<B, D>,
        alpha: f32,
    ) -> Tensor<B, D> {
        let alpha = alpha.clamp(0.0, 1.0);
        (fresh.mul_scalar(1.0 - alpha) + source.mul_scalar(alpha)).detach()
    }

    fn match_fresh_rms<const D: usize>(source: Tensor<B, D>, fresh: Tensor<B, D>) -> Tensor<B, D> {
        let source_rms = Self::param_rms(source.clone());
        let fresh_rms = Self::param_rms(fresh);
        if source_rms <= 1.0e-8 || !source_rms.is_finite() || !fresh_rms.is_finite() {
            return source;
        }
        source.mul_scalar(fresh_rms / source_rms).detach()
    }

    pub fn new(config: &DragonNormConfig, width: usize, device: &B::Device) -> Self {
        let width = width.max(1);
        let alpha_init = config.resolved_alpha_init();
        let shift_init = config.resolved_shift_init();
        Self {
            kind: config.kind,
            eps: config.eps.max(1e-8),
            gamma: Param::from_tensor(Tensor::<B, 1>::ones([width], device)),
            beta: Param::from_tensor(Tensor::<B, 1>::zeros([width], device)),
            alpha: Param::from_tensor(Tensor::<B, 1>::ones([1], device).mul_scalar(alpha_init)),
            shift: Param::from_tensor(Tensor::<B, 1>::ones([1], device).mul_scalar(shift_init)),
        }
    }

    pub fn blended_with(&self, fresh: &Self, alpha: f32) -> Self {
        Self {
            kind: self.kind,
            eps: self.eps,
            gamma: Param::from_tensor(Self::blend_param(
                self.gamma.val(),
                fresh.gamma.val(),
                alpha,
            )),
            beta: Param::from_tensor(Self::blend_param(self.beta.val(), fresh.beta.val(), alpha)),
            alpha: Param::from_tensor(Self::blend_param(
                self.alpha.val(),
                fresh.alpha.val(),
                alpha,
            )),
            shift: Param::from_tensor(Self::blend_param(
                self.shift.val(),
                fresh.shift.val(),
                alpha,
            )),
        }
    }

    pub fn matched_fresh_rms(&self, fresh: &Self) -> Self {
        Self {
            kind: self.kind,
            eps: self.eps,
            gamma: Param::from_tensor(Self::match_fresh_rms(self.gamma.val(), fresh.gamma.val())),
            beta: Param::from_tensor(Self::match_fresh_rms(self.beta.val(), fresh.beta.val())),
            alpha: Param::from_tensor(Self::match_fresh_rms(self.alpha.val(), fresh.alpha.val())),
            shift: Param::from_tensor(Self::match_fresh_rms(self.shift.val(), fresh.shift.val())),
        }
    }

    pub fn kind(&self) -> DragonNormKind {
        self.kind
    }

    pub fn forward<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let gamma = self.param_view::<D>(self.gamma.val());
        let beta = self.param_view::<D>(self.beta.val());

        let output = match self.kind {
            DragonNormKind::LayerNorm => {
                let (var, mean) = tensor.clone().var_mean_bias(D - 1);
                tensor.sub(mean).div(var.add_scalar(self.eps).sqrt())
            }
            DragonNormKind::RmsNorm => {
                let (var, mean) = tensor.clone().var_mean_bias(D - 1);
                let rms = var.add(mean.powf_scalar(2.0)).add_scalar(self.eps).sqrt();
                tensor.div(rms)
            }
            DragonNormKind::DynamicTanh => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                tensor.mul(alpha).tanh()
            }
            DragonNormKind::Derf => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let shift = self.scalar_param_view::<D>(self.shift.val());
                tensor.mul(alpha).add(shift).erf()
            }
        };

        output.mul(gamma).add(beta)
    }

    fn param_view<const D: usize>(&self, param: Tensor<B, 1>) -> Tensor<B, D> {
        let [width] = param.shape().dims::<1>();
        let mut shape = [1; D];
        shape[D - 1] = width;
        param.reshape(shape)
    }

    fn scalar_param_view<const D: usize>(&self, param: Tensor<B, 1>) -> Tensor<B, D> {
        let shape = [1; D];
        param.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    fn device() -> <Backend as burn::tensor::backend::Backend>::Device {
        <Backend as burn::tensor::backend::Backend>::Device::default()
    }

    #[test]
    fn layer_norm_zero_centers_rows() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::LayerNorm,
                ..Default::default()
            },
            2,
            &device,
        );
        let x = Tensor::<Backend, 2>::from_data(TensorData::new(vec![1.0, 3.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            (data[0] + 1.0).abs() < 1e-4,
            "expected approx -1, got {}",
            data[0]
        );
        assert!(
            (data[1] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[1]
        );
    }

    #[test]
    fn rms_norm_preserves_direction_without_mean_centering() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::RmsNorm,
                ..Default::default()
            },
            2,
            &device,
        );
        let x = Tensor::<Backend, 2>::from_data(TensorData::new(vec![3.0, 3.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            (data[0] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[0]
        );
        assert!(
            (data[1] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[1]
        );
    }

    #[test]
    fn dynamic_tanh_is_bounded() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::DynamicTanh,
                ..Default::default()
            },
            2,
            &device,
        );
        let x =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![-10.0, 10.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            data[0] > -1.01 && data[0] < -0.9,
            "expected bounded negative output, got {}",
            data[0]
        );
        assert!(
            data[1] < 1.01 && data[1] > 0.9,
            "expected bounded positive output, got {}",
            data[1]
        );
    }

    #[test]
    fn derf_is_bounded() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::Derf,
                ..Default::default()
            },
            2,
            &device,
        );
        let x =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![-10.0, 10.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            data[0] > -1.01 && data[0] < -0.9,
            "expected bounded negative output, got {}",
            data[0]
        );
        assert!(
            data[1] < 1.01 && data[1] > 0.9,
            "expected bounded positive output, got {}",
            data[1]
        );
    }

    #[test]
    fn dyt_and_derf_use_scalar_alpha_and_shift_parameters() {
        let device = device();
        let dyt = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::DynamicTanh,
                ..Default::default()
            },
            8,
            &device,
        );
        let derf = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::Derf,
                ..Default::default()
            },
            8,
            &device,
        );

        assert_eq!(dyt.alpha.val().shape().dims::<1>(), [1]);
        assert_eq!(dyt.shift.val().shape().dims::<1>(), [1]);
        assert_eq!(derf.alpha.val().shape().dims::<1>(), [1]);
        assert_eq!(derf.shift.val().shape().dims::<1>(), [1]);
    }

    #[test]
    fn norm_kind_specific_defaults_resolve_expected_alpha() {
        let dyt = DragonNormConfig {
            kind: DragonNormKind::DynamicTanh,
            ..Default::default()
        };
        let derf = DragonNormConfig {
            kind: DragonNormKind::Derf,
            ..Default::default()
        };

        assert!((dyt.resolved_alpha_init() - 0.5).abs() < 1e-6);
        assert!((derf.resolved_alpha_init() - 0.886_226_95).abs() < 1e-6);
    }
}
