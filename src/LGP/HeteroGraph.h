#pragma once

#include <string>
#include <unordered_map>
#pragma push_macro("LOG")
#pragma push_macro("CHECK")
#undef LOG
#undef CHECK

#include <torch/torch.h>

// Restore RAI macros after torch includes
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

struct HeteroGraph {
    // Node features dictionary: node_type -> feature tensor
    std::unordered_map<std::string, torch::Tensor> x_dict;
    
    // Node times dictionary: node_type -> times tensor
    std::unordered_map<std::string, torch::Tensor> times_dict;
    
    // Batch dictionary: node_type -> batch tensor (for graph pooling)
    std::unordered_map<std::string, torch::Tensor> batch_dict;
    
    // Edge indices dictionary: edge_type_string -> edge_index tensor
    // Edge type strings are formatted as "src___edge_name___dst"
    std::unordered_map<std::string, torch::Tensor> edge_index_dict;
    
    // Constructor
    HeteroGraph() {}
    
    HeteroGraph(
        const std::unordered_map<std::string, torch::Tensor>& x_dict,
        const std::unordered_map<std::string, torch::Tensor>& times_dict,
        const std::unordered_map<std::string, torch::Tensor>& batch_dict,
        const std::unordered_map<std::string, torch::Tensor>& edge_index_dict
    ) : x_dict(x_dict), 
        times_dict(times_dict),
        batch_dict(batch_dict),
        edge_index_dict(edge_index_dict) {}
};