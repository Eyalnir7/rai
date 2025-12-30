#pragma once

#include "HeteroGraph.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>

typedef rai::Array<double> arr;

// Edge types with their data (rolled out from pair_edges and sink_edges)
struct EdgeData {
    bool directed;
    std::vector<std::pair<rai::String, rai::String>> edges;
};

struct SinkEdgeData {
    bool directed;
    std::vector<std::vector<rai::String>> edges; // Each edge is a vector: [sink, connected1, connected2, ...]
};

struct IntermediateHeteroData {
    // Node types with their data (rolled out from relevant_nodes)
    struct NodeTypeData {
        std::vector<rai::String> names;
        std::vector<torch::Tensor> features;
        std::vector<int> times;
    };
    
    NodeTypeData ssBox_nodes;
    NodeTypeData place_frame_nodes;
    NodeTypeData object_nodes;
    NodeTypeData ssCylinder_nodes;
    NodeTypeData pick_nodes;
    NodeTypeData place_nodes;
    
    EdgeData close_edges;
    EdgeData time_edges;
    SinkEdgeData pick_edges;
    SinkEdgeData place_edges;
    
    // Constructor
    IntermediateHeteroData() {
        // Initialize edge properties
        close_edges.directed = false;
        time_edges.directed = true;
        pick_edges.directed = false;
        place_edges.directed = false;
    }
};

// Main function to convert rai::Configuration to HeteroGraph with task plan
HeteroGraph convertToHeteroGraph(const IntermediateHeteroData& interm);

// Return intermediate data structure analogous to Python get_hetero_data_input
IntermediateHeteroData get_hetero_data_input(
    rai::Configuration& C,
    StringAA task_plan,
    torch::Device device,
    int action_number = -1);




