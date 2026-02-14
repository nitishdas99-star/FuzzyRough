use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FrError {
    #[error("input matrix is empty")]
    EmptyInput,
    #[error("label length mismatch: expected {expected}, found {found}")]
    LabelLengthMismatch { expected: usize, found: usize },
    #[error("model is not fitted")]
    NotFitted,
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
