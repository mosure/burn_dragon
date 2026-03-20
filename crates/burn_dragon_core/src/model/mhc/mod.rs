mod coefficients;
mod config;
mod reference;

pub use coefficients::{
    ManifoldHyperConnectionCoefficients, ManifoldHyperConnectionStreamCoefficients,
    ManifoldHyperConnectionStreamOutput, ManifoldHyperConnectionWidthOutput,
};
pub use config::{ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionsConfig};
pub use reference::ManifoldHyperConnections;

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub fn mhc_split<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2>>) {
    mhc_split_with_coefficients(mhc, residuals, None)
}

pub fn mhc_split_with_coefficients<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
    coefficients: Option<&ManifoldHyperConnectionCoefficients<B>>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2>>) {
    if let Some(mhc) = mhc {
        if let Some(coefficients) = coefficients {
            mhc.width_connection_with_coefficients(residuals, coefficients)
                .into_legacy()
        } else {
            mhc.width_connection(residuals).into_legacy()
        }
    } else {
        (residuals.clone(), residuals, None)
    }
}

pub fn mhc_merge<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    branch_output: Tensor<B, 4>,
    residuals: Tensor<B, 4>,
    beta: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    mhc_merge_with_coefficients(mhc, branch_output, residuals, None, beta)
}

pub fn mhc_merge_with_coefficients<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    branch_output: Tensor<B, 4>,
    residuals: Tensor<B, 4>,
    coefficients: Option<&ManifoldHyperConnectionCoefficients<B>>,
    beta: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    if let Some(mhc) = mhc {
        if let Some(coefficients) = coefficients {
            mhc.depth_connection_with_coefficients(branch_output, residuals, coefficients)
        } else {
            mhc.depth_connection(branch_output, residuals, beta)
        }
    } else {
        branch_output
    }
}

pub fn mhc_passthrough<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
) -> Tensor<B, 4> {
    mhc_passthrough_with_coefficients(mhc, residuals, None)
}

pub fn mhc_passthrough_with_coefficients<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
    coefficients: Option<&ManifoldHyperConnectionCoefficients<B>>,
) -> Tensor<B, 4> {
    if let Some(mhc) = mhc {
        if let Some(coefficients) = coefficients {
            let output = mhc.width_connection_with_coefficients(residuals, coefficients);
            mhc.depth_connection_with_coefficients(
                output.branch_input,
                output.residuals_out,
                coefficients,
            )
        } else {
            mhc.passthrough(residuals)
        }
    } else {
        residuals
    }
}

#[cfg(test)]
mod tests;
