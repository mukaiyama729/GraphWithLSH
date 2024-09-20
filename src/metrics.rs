use ndarray::{Array1, Array2, arr2};
use pyo3::prelude::*;
use pyo3::types::{PyList};
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use crate::compute;
use compute::kernels::{AlphaDecayKernel, AlphaKernel};
use compute::similarity::{CosineDistanceCalc, EuclideanDistanceCalc, ManhattanDistanceCalc};
use compute::traits::distance::Distance;

fn vec_to_array2(v: Vec<Vec<f32>>) -> PyResult<Array2<f32>> {
    let rows = v.len();
    let cols = v[0].len();

    // Vec<Vec<f32>> を 1次元の Vec<f32> に変換
    let flattened: Vec<f32> = v.into_iter().flatten().collect();

    // Array2<f32> に変換
    Array2::from_shape_vec((rows, cols), flattened)
        .map_err(|_| PyValueError::new_err("Error converting to Array2"))
}

fn process_list_of_lists(input: &PyList) -> PyResult<Array2<f32>> {
    let mut result: Vec<Vec<f32>> = Vec::new();

    // inputはリストのリストなので、各要素がリスト
    for item in input.iter() {
        let sublist = item.downcast::<PyList>()?; // 各要素もPyListとして扱う
        let vec: Vec<f32> = sublist.extract()?; // 各サブリストをVec<f32>に変換
        result.push(vec);
    }

    // Vec<Vec<f32>> を Array2<f32> に変換して返す
    vec_to_array2(result)
}

#[pyclass]
pub struct DistanceClass {}

#[pymethods]
impl DistanceClass {
    #[new]
    fn new() -> Self {
        DistanceClass {}
    }

    pub fn cosine_distance_matrix(&self, vectors: &PyList) -> PyResult<Vec<Vec<f32>>>  {
        let vectors = process_list_of_lists(vectors)?; // Array2<f32> を受け取る
        let cosine_distance = CosineDistanceCalc::CosineDistance{};
        let result = cosine_distance.distance_matrix(&vectors);
        match result {
            Ok(arr) => Ok(arr.outer_iter().map(|v| v.to_vec()).collect()),
            Err(_) => Err(PyValueError::new_err("Error in cosine distance calculation")),
        }
    }

    pub fn cosine_distance_matrix_simd(&self, vectors: &PyList) -> PyResult<Vec<Vec<f32>>> {
        let vectors = process_list_of_lists(vectors)?; // Array2<f32> を受け取る
        let cosine_distance = CosineDistanceCalc::CosineDistanceSIMD{};
        let result = cosine_distance.distance_matrix(&vectors);
        match result {
            Ok(arr) => Ok(arr.outer_iter().map(|v| v.to_vec()).collect()),
            Err(_) => Err(PyValueError::new_err("Error in cosine distance calculation")),
        }
    }

    pub fn euclidean_distance_matrix(&self, vectors: &PyList) -> PyResult<Vec<Vec<f32>>> {
        let vectors = process_list_of_lists(vectors)?; // Array2<f32> を受け取る
        let euclidean_distance = EuclideanDistanceCalc::EuclideanDistance {};
        let result = euclidean_distance.distance_matrix(&vectors);
        match result {
            Ok(arr) => Ok(arr.outer_iter().map(|v| v.to_vec()).collect()),
            Err(_) => Err(PyValueError::new_err("Error in euclidean distance calculation")),
        }
    }

    pub fn euclidean_distance_matrix_simd(&self, vectors: &PyList) -> PyResult<Vec<Vec<f32>>> {
        let vectors = process_list_of_lists(vectors)?; // Array2<f32> を受け取る
        let euclidean_distance = EuclideanDistanceCalc::EuclideanDistanceSIMD {};
        let result = euclidean_distance.distance_matrix(&vectors);
        match result {
            Ok(arr) => Ok(arr.outer_iter().map(|v| v.to_vec()).collect()),
            Err(_) => Err(PyValueError::new_err("Error in euclidean distance calculation")),
        }
    }
}
