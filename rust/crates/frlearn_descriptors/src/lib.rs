//! Descriptor and novelty detection models for fuzzy-rough workflows.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod nnd;

pub use nnd::{NND, NNDModel};

#[cfg(feature = "svm")]
pub mod svm {
    use frlearn_core::{FrError, FrResult, Matrix};

    #[derive(Debug, Clone, Copy, Default)]
    pub struct SvmDescriptor;

    impl SvmDescriptor {
        pub fn fit(&self, _x_inlier: &Matrix) -> FrResult<()> {
            Err(FrError::InvalidInput(
                "svm backend is feature-gated placeholder".to_string(),
            ))
        }
    }
}

#[cfg(feature = "eif")]
pub mod eif {
    use frlearn_core::{FrError, FrResult, Matrix};

    #[derive(Debug, Clone, Copy, Default)]
    pub struct EifDescriptor;

    impl EifDescriptor {
        pub fn fit(&self, _x_inlier: &Matrix) -> FrResult<()> {
            Err(FrError::InvalidInput(
                "eif backend is feature-gated placeholder".to_string(),
            ))
        }
    }
}

#[cfg(feature = "sae")]
pub mod sae {
    use frlearn_core::{FrError, FrResult, Matrix};

    #[derive(Debug, Clone, Copy, Default)]
    pub struct SaeDescriptor;

    impl SaeDescriptor {
        pub fn fit(&self, _x_inlier: &Matrix) -> FrResult<()> {
            Err(FrError::InvalidInput(
                "sae backend is feature-gated placeholder".to_string(),
            ))
        }
    }
}
