use ndarray::{Array1, Array2, arr2};
use pyo3::prelude::*;
use pyo3::types::{PyList};
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

pub mod compute;
pub mod metrics;
pub mod kernel_functions;

use compute::kernels::{AlphaDecayKernel, AlphaKernel};
use metrics::DistanceClass;
use kernel_functions::{alpha_decay, gaussian_kernel};

#[pymodule]
fn rust_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alpha_decay, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_kernel, m)?)?;
    m.add_class::<DistanceClass>()?;
    Ok(())
}
