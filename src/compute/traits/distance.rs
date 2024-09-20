use ndarray::Array2;
use std::fmt::Error;

pub trait Distance {
    fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error>;
}
