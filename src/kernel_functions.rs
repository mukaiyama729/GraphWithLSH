use ndarray::{Array1};
use pyo3::prelude::*;
use pyo3::types::{PyList};
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult};

use crate::compute;
use compute::kernels::{AlphaDecayKernel, AlphaKernel, GaussianKernel};

#[pyfunction]
pub fn alpha_decay(input: &PyList, knn_dist1: f32, knn_dist2: &PyList, alpha: i32) -> PyResult<Vec<f32>> {
    let input = input.extract::<Vec<f32>>()?;
    let knn_dist2 = knn_dist2.extract::<Vec<f32>>()?;
    let input = Array1::from(input);
    let knn_dist2 = Array1::from(knn_dist2);
    let kernel = AlphaDecayKernel::new(alpha);
    let result = kernel.apply(&input, &knn_dist1, &knn_dist2);
    match result {
        Ok(arr) => Ok(arr.to_vec()),
        Err(_) => Err(PyValueError::new_err("Error in kernel application")),
    }
}

#[pyfunction]
pub fn gaussian_kernel(input: &PyList, knn_dist1: f32, knn_dist2: &PyList, sigma: f32) -> PyResult<Vec<f32>> {
    let input = input.extract::<Vec<f32>>()?;
    let knn_dist2 = knn_dist2.extract::<Vec<f32>>()?;
    let input = Array1::from(input);
    let knn_dist2 = Array1::from(knn_dist2);
    let kernel = GaussianKernel::new(sigma);
    let result = kernel.apply(&input, &knn_dist1, &knn_dist2);
    match result {
        Ok(arr) => Ok(arr.to_vec()),
        Err(_) => Err(PyValueError::new_err("Error in kernel application")),
    }
}
