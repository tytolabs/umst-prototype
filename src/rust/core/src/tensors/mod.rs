// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT

pub mod functor;
pub mod geometry;
pub mod hyper_graph_tensor;
pub mod mix;
// pub mod sparse; // Disabled (Might be safe but not needed)

pub use geometry::GeometryData;
pub use hyper_graph_tensor::{HyperGraphTensor, TensorConstraint, TensorNode};
pub use mix::{MaterialType, MixTensor};

// pub use sparse::SparseTensor;
