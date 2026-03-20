use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Debug, Clone)]
pub struct ManifoldHyperConnectionCoefficients<B: Backend> {
    pub residual_weights: Tensor<B, 2>,
    pub branch_input_weights: Tensor<B, 2>,
    pub branch_output_weights: Option<Tensor<B, 2>>,
}

#[derive(Debug, Clone)]
pub struct ManifoldHyperConnectionWidthOutput<B: Backend> {
    pub branch_input: Tensor<B, 4>,
    pub residuals_out: Tensor<B, 4>,
    pub coefficients: ManifoldHyperConnectionCoefficients<B>,
}

impl<B: Backend> ManifoldHyperConnectionWidthOutput<B> {
    pub fn into_legacy(self) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2>>) {
        (
            self.branch_input,
            self.residuals_out,
            self.coefficients.branch_output_weights,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldHyperConnectionStreamCoefficients<B: Backend> {
    pub residual_weights: Tensor<B, 4>,
    pub branch_input_weights: Tensor<B, 3>,
    pub branch_output_weights: Option<Tensor<B, 3>>,
}

#[derive(Debug, Clone)]
pub struct ManifoldHyperConnectionStreamOutput<B: Backend> {
    pub branch_input: Tensor<B, 4>,
    pub residuals_out: Tensor<B, 4>,
    pub coefficients: ManifoldHyperConnectionStreamCoefficients<B>,
}
