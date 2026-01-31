// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::GeometryData;

/// The Unified Hierarchical Tensor State (UMST)
///
/// A Hypergraph where nodes represent physical entities (Voxels, Joints, SDFs)
/// and edges represent constraints (Rheology, Collision).
///
/// Corresponds to Figure 1 in the documentation.
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct HyperGraphTensor {
    // We wrap the petgraph structure using 'u32' index for WASM compatibility
    #[serde(skip)]
    pub(crate) graph: Graph<TensorNode, TensorConstraint, Directed, u32>,
    root_id: u32,
}

/// Nodes within the Unified Tensor
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TensorNode {
    /// Material Node: Holds rheological properties (Yield Stress, Viscosity)
    Material {
        id: String,
        yield_stress: f32, // Pascals
        viscosity: f32,    // Pa*s
        density: f32,      // kg/m3
        cost: f32,
        carbon: f32,
    },
    /// Geometric Node: Holds spatial representation (SDF/Voxel)
    Geometry {
        id: String,
        sdf_data: Vec<u8>,
        bounds: [f32; 6],
    },
    /// Kinematic Node: Holds robot/actuator state
    Kinematic {
        id: String,
        joints: Vec<f32>,
        tcp_pose: [f32; 7],
    },
    /// Trajectory Node: Represents a toolpath (Slicer Output)
    Trajectory {
        id: String,
        waypoints: Vec<[f32; 5]>, // [x, y, z, extrusion, speed]
    },
    /// Structural Node: Grouping mechanism
    Group { id: String },
}

/// Edges representing Constraints/Relationships
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TensorConstraint {
    /// Material dictates Kinematic limit (Viscosity -> Speed Limit)
    RheologyConstraint { factor: f32 },
    /// Geometry defines collision boundary
    CollisionConstraint,
    /// Parent-Child hierarchy
    Hierarchical,
    /// Explicit Assignment (e.g., Geometry G assigned to Material M)
    Assignment,
}

#[derive(Serialize)]
struct RenderNode {
    idx: u32,
    id: String,
    label: String,
    node_type: String,
    radius: f32,
}

#[derive(Serialize)]
struct RenderEdge {
    source: String,
    target: String,
    constraint: String,
}

#[derive(Serialize)]
struct RenderGraphState {
    nodes: Vec<RenderNode>,
    edges: Vec<RenderEdge>,
}

#[wasm_bindgen]
impl HyperGraphTensor {
    /// Initialize a new, empty Unified Tensor
    #[wasm_bindgen(constructor)]
    pub fn new() -> HyperGraphTensor {
        let mut graph = Graph::new();
        let root = graph.add_node(TensorNode::Group {
            id: "Project_Root".to_string(),
        });

        HyperGraphTensor {
            graph,
            root_id: root.index() as u32,
        }
    }

    /// Add a Material Node
    pub fn add_material(
        &mut self,
        id: String,
        yield_stress: f32,
        viscosity: f32,
        cost: f32,
        carbon: f32,
    ) -> u32 {
        let node = TensorNode::Material {
            id,
            yield_stress,
            viscosity,
            density: 2300.0, // Default concrete density
            cost,
            carbon,
        };
        let idx = self.graph.add_node(node);
        self.graph.add_edge(
            NodeIndex::new(self.root_id as usize),
            idx,
            TensorConstraint::Hierarchical,
        );
        idx.index() as u32
    }

    /// Add a Geometry Node (Spatial Data)
    pub fn add_geometry(&mut self, id: String, _sdf_data: &[u8], bounds: &[f32]) -> u32 {
        let node = TensorNode::Geometry {
            id,
            sdf_data: _sdf_data.to_vec(),
            bounds: bounds.try_into().unwrap_or([0.0; 6]),
        };
        let idx = self.graph.add_node(node);
        self.graph.add_edge(
            NodeIndex::new(self.root_id as usize),
            idx,
            TensorConstraint::Hierarchical,
        );
        idx.index() as u32
    }

    /// Assign a specific Material to a Geometry Node
    pub fn assign_material(&mut self, geo_id: u32, mat_id: u32) {
        self.graph.add_edge(
            NodeIndex::new(geo_id as usize),
            NodeIndex::new(mat_id as usize),
            TensorConstraint::Assignment,
        );
    }

    /// Run Topological Sort to detect cycles (Oracle Validation)
    pub fn validate(&self) -> bool {
        petgraph::algo::toposort(&self.graph, None).is_ok()
    }

    /// Get Graph State for Visualization
    pub fn get_graph_state(&self) -> JsValue {
        let nodes: Vec<RenderNode> = self
            .graph
            .raw_nodes()
            .iter()
            .enumerate()
            .map(|(idx, n)| {
                let (id, label, node_type) = match &n.weight {
                    TensorNode::Group { id } => (id.clone(), id.clone(), "Group".to_string()),
                    TensorNode::Material { id, .. } => {
                        (id.clone(), id.clone(), "Material".to_string())
                    }
                    TensorNode::Geometry { id, .. } => {
                        (id.clone(), id.clone(), "Geometry".to_string())
                    }
                    TensorNode::Kinematic { id, .. } => {
                        (id.clone(), id.clone(), "Kinematic".to_string())
                    }
                    TensorNode::Trajectory { id, .. } => {
                        (id.clone(), id.clone(), "Trajectory".to_string())
                    }
                };
                RenderNode {
                    idx: idx as u32,
                    id,
                    label,
                    node_type,
                    radius: 20.0,
                }
            })
            .collect();

        let edges: Vec<RenderEdge> = self
            .graph
            .raw_edges()
            .iter()
            .filter_map(|e| {
                let source = self.graph.node_weight(e.source())?;
                let target = self.graph.node_weight(e.target())?;
                // Simplified ID extraction for brevity
                let get_id = |n: &TensorNode| match n {
                    TensorNode::Group { id } => id.clone(),
                    TensorNode::Material { id, .. } => id.clone(),
                    TensorNode::Geometry { id, .. } => id.clone(),
                    TensorNode::Kinematic { id, .. } => id.clone(),
                    TensorNode::Trajectory { id, .. } => id.clone(),
                };
                Some(RenderEdge {
                    source: get_id(source),
                    target: get_id(target),
                    constraint: "Constraint".to_string(),
                })
            })
            .collect();

        serde_wasm_bindgen::to_value(&RenderGraphState { nodes, edges }).unwrap_or(JsValue::NULL)
    }

    /// Extract geometry data from all geometry nodes in the hypergraph
    pub fn extract_geometry_data(&self) -> Vec<GeometryData> {
        self.graph
            .raw_nodes()
            .iter()
            .filter_map(|node| {
                if let TensorNode::Geometry { sdf_data, bounds, .. } = &node.weight {
                    Some(GeometryData::from_sdf_data(sdf_data, bounds))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get aggregated geometry properties (average across all geometry nodes)
    pub fn get_aggregate_geometry(&self) -> GeometryData {
        let geometries = self.extract_geometry_data();
        if geometries.is_empty() {
            return GeometryData::default();
        }

        let mut aggregated = GeometryData {
            surface_area: geometries.iter().map(|g| g.surface_area).sum::<f32>() / geometries.len() as f32,
            volume: geometries.iter().map(|g| g.volume).sum::<f32>() / geometries.len() as f32,
            bounding_volume: geometries.iter().map(|g| g.bounding_volume).sum::<f32>() / geometries.len() as f32,
            sav_ratio: geometries.iter().map(|g| g.sav_ratio).sum::<f32>() / geometries.len() as f32,
            complexity: geometries.iter().map(|g| g.complexity).sum::<f32>() / geometries.len() as f32,
            mean_curvature: geometries.iter().map(|g| g.mean_curvature).sum::<f32>() / geometries.len() as f32,
            gaussian_curvature: geometries.iter().map(|g| g.gaussian_curvature).sum::<f32>() / geometries.len() as f32,
        };

        // Recalculate derived properties
        if aggregated.volume > 0.0 {
            aggregated.sav_ratio = aggregated.surface_area / aggregated.volume;
        }

        aggregated
    }
}
