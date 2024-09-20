use std::fmt::Error;
use ndarray::Array1;

pub trait AlphaKernel {
    fn apply(&self, input: &Array1<f32>, knn_dist1: &f32, knn_dist2:  &Array1<f32>) -> Result<Array1<f32>, Error>;
}

pub struct AlphaDecayKernel {
    pub alpha: i32,
}

impl AlphaKernel for AlphaDecayKernel {
    fn apply(&self, input: &Array1<f32>, knn_dist1: &f32, knn_dist2: &Array1<f32>) -> Result<Array1<f32>, Error> {
        let noise = 0.0000001;
        let first_args = (input / (*knn_dist1 + noise));
        let first_args = first_args.mapv(|x| (-x.powi(self.alpha)).exp());
        let second_args = input / &(knn_dist2.mapv(|x| x + noise));
        let second_args = second_args.mapv(|x| (-x.powi(self.alpha)).exp());
        return Ok((first_args + second_args) / 2.0);
    }
}

impl AlphaDecayKernel {
    pub fn new(alpha: i32) -> Self {
        AlphaDecayKernel {
            alpha: alpha,
        }
    }
}

pub struct GaussianKernel {
    sigma: f32,
}

impl GaussianKernel {
    pub fn new(sigma: f32) -> Self {
        GaussianKernel {
            sigma: sigma,
        }
    }

    pub fn apply(&self, input: &Array1<f32>, knn_dist1: &f32, knn_dist2: &Array1<f32>) -> Result<Array1<f32>, Error> {
        let noise = 0.0000001;
        let first_args = (input / (*knn_dist1 + noise)).mapv(|x| -x.powi(2) / (2.0 * self.sigma.powi(2))).mapv(|x| x.exp());
        let second_args = input / &(knn_dist2.mapv(|x| x + noise));
        let second_args = second_args.mapv(|x| -x.powi(2) / (2.0 * self.sigma.powi(2))).mapv(|x| x.exp());
        return Ok((first_args + second_args) / 2.0);
    }
}
