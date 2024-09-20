pub mod CosineDistanceCalc {
    use std::fmt::Error;
    use ndarray::{Array2, Axis};
    use wide::f32x4;
    use crate::compute::traits::distance::Distance;
    use pyo3::prelude::*;

    pub struct CosineDistance {}

    impl Distance for CosineDistance {
        fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
            let dist = 1.0 - vectors.dot(&vectors.t());
            return Ok(dist);
        }
    }
    pub struct CosineDistanceSIMD {}

    impl Distance for CosineDistanceSIMD {
        fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
            let (n, d) = vectors.dim();
            let mut distance_matrix = Array2::zeros((n, n));
            for i in 0..n {
                for j in i..n {
                    let vec1 = vectors.row(i).to_vec();
                    let vec2 = vectors.row(j).to_vec();
                    let dist = self.cosine_dist(&vec1, &vec2);
                    distance_matrix[[i, j]] = dist;
                    distance_matrix[[j, i]] = dist;
                }
            }
            return Ok(distance_matrix);
        }
    }

    impl CosineDistanceSIMD {
        fn cosine_dist(&self, vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
            let dim = vec1.len();
            let mut dot_product = f32x4::splat(0.0);
            let mut norm1  = f32x4::splat(0.0);
            let mut norm2  = f32x4::splat(0.0);

            let mut rest_dot_product = 0.0;
            let mut rest_norm1 = 0.0;
            let mut rest_norm2 = 0.0;

            let chunk = dim / 4;

            for k in 0..chunk {
                let vec1_part = f32x4::from(&vec1[k * 4..]);
                let vec2_part = f32x4::from(&vec2[k * 4..]);

                dot_product += vec1_part * vec2_part;
                norm1 += vec1_part * vec1_part;
                norm2 += vec2_part * vec2_part;
            }

            for k in chunk * 4..dim {
                rest_dot_product += vec1[k] * vec2[k];
                rest_norm1 += vec1[k] * vec1[k];
                rest_norm2 += vec2[k] * vec2[k];
            }
            let dot_product = self.sum_f32x4(&dot_product) + rest_dot_product;
            let norm1 = self.sum_f32x4(&norm1) + rest_norm1;
            let norm2 = self.sum_f32x4(&norm2) + rest_norm2;

            return 1.0 - dot_product / (norm1.sqrt() * norm2.sqrt());
        }

        fn sum_f32x4(&self, vec: &f32x4) -> f32 {
            let arr: [f32; 4] = vec.to_array();
            arr.iter().sum()
        }
    }
}


pub mod EuclideanDistanceCalc {
    use std::fmt::Error;
    use ndarray::{Array2, Axis};
    use wide::f32x4;
    use crate::compute::traits::distance::Distance;
    pub struct EuclideanDistance {}

    impl Distance for EuclideanDistance {
        fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
            let (n, d) = vectors.dim();

            // a: (n, 1, d) -> ブロードキャストの準備 (n, n, d) にブロードキャストされることを期待
            let a = vectors.view().insert_axis(Axis(1));
            let a_broadcast = a.broadcast((n, n, d)).expect("Failed to broadcast a");

            // b: (1, n, d) -> ブロードキャストの準備 (n, n, d) にブロードキャストされることを期待
            let b = vectors.view().insert_axis(Axis(0));
            let b_broadcast = b.broadcast((n, n, d)).expect("Failed to broadcast b");

            // 差分を計算: (n, n, d)
            let diffs = &a_broadcast - &b_broadcast;
            let squared_diffs = diffs.mapv(|x| x.powi(2));
            let sum_squared_diffs = squared_diffs.sum_axis(Axis(2));
            let euclidean_dists = sum_squared_diffs.mapv(|x| x.sqrt());
            Ok(euclidean_dists)
        }
    }

    pub struct EuclideanDistanceSIMD {}

    impl Distance for EuclideanDistanceSIMD {
        fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
            let (n, d) = vectors.dim();
            let mut distance_matrix = Array2::zeros((n, n));

            for i in 0..n {
                for j in i..n {
                    let vec1 = vectors.row(i).to_vec();
                    let vec2 = vectors.row(j).to_vec();
                    let dist = self.euclidean_dist(&vec1, &vec2);
                    distance_matrix[[i, j]] = dist;
                    distance_matrix[[j, i]] = dist;
                }
            }
            return Ok(distance_matrix);
        }
    }

    impl EuclideanDistanceSIMD {
        fn euclidean_dist(&self, vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
            let dim = vec1.len();
            let mut squared_diffs = f32x4::splat(0.0);
            let mut rest_squared_diffs = 0.0;
            let chunk = dim / 4;
            for k in 0..chunk {
                let vec1_part = f32x4::from(&vec1[k*4..(k+1)*4]);
                let vec2_part = f32x4::from(&vec2[k*4..(k+1)*4]);
                let diff = vec1_part - vec2_part;
                squared_diffs += diff * diff;
            }

            for k in chunk * 4..dim {
                let diff = vec1[k] - vec2[k];
                rest_squared_diffs += diff * diff;
            }
            let squared_diffs = self.sum_f32x4(&squared_diffs) + rest_squared_diffs;
            return squared_diffs.sqrt();
        }

        fn sum_f32x4(&self, vec: &f32x4) -> f32 {
            let arr: [f32; 4] = vec.to_array();
            arr.iter().sum()
        }
    }
}

pub mod ManhattanDistanceCalc {
    use std::fmt::Error;
    use ndarray::{Array2, Axis};
    use wide::f32x4;
    use crate::compute::traits::distance::Distance;
    pub struct ManhattanDistance {}

    impl Distance for ManhattanDistance {
        fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
            let (n, d) = vectors.dim();

            // expect to broadcast (n, 1, d) to (n, n, d)
            let a = vectors.view().insert_axis(Axis(1));
            let a_broadcast = a.broadcast((n, n, d)).expect("Failed to broadcast a");

            // expect to broadcast (1, n, d) to (n, n, d)
            let b = vectors.view().insert_axis(Axis(0));
            let b_broadcast = b.broadcast((n, n, d)).expect("Failed to broadcast b");

            // calculate difference between vectors a, b: (n, n, d)
            let diffs = &a_broadcast - &b_broadcast;
            let abs_diffs = diffs.mapv(|x| x.abs());
            let manhattan_dists = abs_diffs.sum_axis(Axis(2));
            Ok(manhattan_dists)
        }
    }
}


pub mod ChebyshevDistanceCalc {
    use std::fmt::Error;
    use ndarray::{Array2, Axis};
    use wide::f32x4;
    use crate::compute::traits::distance::Distance;
    pub struct ChebyshevDistance {}

    //impl Distance for ChebyshevDistance {
    //    fn distance_matrix(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, Error> {
    //        // ベクトルの数(n)と次元(d)を取得
    //        let (n, d) = vectors.dim();
    //        // a: (n, 1, d)
    //        let a = vectors.view().insert_axis(Axis(1));
    //        // b: (1, n, d)
    //        let b = vectors.view().insert_axis(Axis(0));
    //        // 差分を計算: (n, n, d)
    //        let diffs = &a - &b;
    //        let abs_diffs = diffs.mapv(|x| x.abs());
    //        let chebyshev_dists = abs_diffs.map_axis(Axis(2), |x| x.max());
    //        Ok(chebyshev_dists)
    //    }
    //}
}

