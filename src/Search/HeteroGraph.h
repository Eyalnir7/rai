#pragma once

#include <string>
#include <unordered_map>
#include <torch/torch.h>

struct HeteroGraph {
    // Node features dictionary: node_type -> feature tensor
    std::unordered_map<std::string, torch::Tensor> x_dict;
    
    // Node times dictionary: node_type -> times tensor
    std::unordered_map<std::string, torch::Tensor> times_dict;
    
    // Node actives dictionary: node_type -> actives tensor
    std::unordered_map<std::string, torch::Tensor> actives_dict;
    
    // Edge indices dictionary: edge_type_string -> edge_index tensor
    // Edge type strings are formatted as "src___edge_name___dst"
    std::unordered_map<std::string, torch::Tensor> edge_index_dict;
    
    // Number of constraint nodes
    int num_pick_nodes;
    int num_place_nodes;
    
    // Constructor
    HeteroGraph() : num_pick_nodes(0), num_place_nodes(0) {}
    
    HeteroGraph(
        const std::unordered_map<std::string, torch::Tensor>& x_dict,
        const std::unordered_map<std::string, torch::Tensor>& times_dict,
        const std::unordered_map<std::string, torch::Tensor>& actives_dict,
        const std::unordered_map<std::string, torch::Tensor>& edge_index_dict,
        int num_pick_nodes,
        int num_place_nodes
    ) : x_dict(x_dict), 
        times_dict(times_dict),
        actives_dict(actives_dict),
        edge_index_dict(edge_index_dict),
        num_pick_nodes(num_pick_nodes),
        num_place_nodes(num_place_nodes) {}
};