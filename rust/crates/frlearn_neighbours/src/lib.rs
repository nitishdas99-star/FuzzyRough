//! Neighbour-based classifiers built on fuzzy-rough principles.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod frnn;
pub mod nn;

pub use frnn::{FRNN, FRNNModel};
pub use nn::{NN, NNModel};
