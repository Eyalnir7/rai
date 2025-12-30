#include "ModelPredictor.h"
#include <iostream>
#include <chrono>

ModelPredictor::ModelPredictor(const std::string& model_path, torch::Device device)
    : device_(device), model_loaded_(false) {
    try {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        model_loaded_ = true;
        std::cout << "Model loaded successfully from: " << model_path << std::endl;
        std::cout << "Using device: " << (device_.is_cpu() ? "CPU" : "CUDA") << std::endl;
        
        // Warm-up pass to trigger JIT compilation
        std::cout << "Warming up model..." << std::endl;
        warmUp();
        std::cout << "Model ready for inference." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model from " << model_path << std::endl;
        std::cerr << e.what() << std::endl;
        model_loaded_ = false;
    }
}

torch::Tensor ModelPredictor::predict(rai::Configuration& C, const StringAA& task_plan) {
    if (!model_loaded_) {
        std::cerr << "Model is not loaded. Cannot make predictions." << std::endl;
        return torch::Tensor();
    }
    
    // Convert configuration and task plan to intermediate heterogeneous data
    IntermediateHeteroData hetero_data = get_hetero_data_input(C, task_plan, device_);
    
    // Convert to HeteroGraph
    HeteroGraph g = convertToHeteroGraph(hetero_data);
    
    // Run the model
    torch::Tensor output = runModelForward(g);
    
    return output;
}

void ModelPredictor::warmUp() {
    // Create dummy inputs with typical sizes
    torch::Dict<std::string, torch::Tensor> x_dict, times_dict, actives_dict, edge_index_dict;
    
    // Add some dummy tensors (adjust sizes based on typical input)
    x_dict.insert("pick", torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("place", torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("ssBox", torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("object", torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    
    times_dict.insert("pick", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("place", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("ssBox", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("object", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    
    actives_dict.insert("pick", torch::zeros({4}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
    actives_dict.insert("place", torch::zeros({4}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
    actives_dict.insert("ssBox", torch::zeros({4}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
    actives_dict.insert("object", torch::zeros({4}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
    
    edge_index_dict.insert("ssBox___pick_edge___pick", torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    edge_index_dict.insert("object___place_edge___place", torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_dict);
    inputs.push_back(times_dict);
    inputs.push_back(actives_dict);
    inputs.push_back(edge_index_dict);
    inputs.push_back((int64_t)4);
    inputs.push_back((int64_t)4);
    
    // Run a few warm-up passes to trigger JIT compilation
    for (int i = 0; i < 3; i++) {
        module_.forward(inputs);
    }
}

torch::Tensor ModelPredictor::runModelForward(const HeteroGraph& g) {
    // Convert unordered_map → torch::Dict<string, Tensor>
    torch::Dict<std::string, torch::Tensor> x_dict, times_dict, actives_dict, edge_index_dict;

    for (const auto& kv : g.x_dict)
        x_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.times_dict)
        times_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.actives_dict)
        actives_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.edge_index_dict)
        edge_index_dict.insert(kv.first, kv.second);

    // Prepare inputs vector in correct order for exportableModel.py
    // forward(x_dict, times_dict, actives_dict, edge_index_dict, num_pick_nodes, num_place_nodes)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_dict);
    inputs.push_back(times_dict);
    inputs.push_back(actives_dict);
    inputs.push_back(edge_index_dict);
    inputs.push_back((int64_t)g.num_pick_nodes);
    inputs.push_back((int64_t)g.num_place_nodes);

    // Forward pass with timing
    auto start = std::chrono::high_resolution_clock::now();
    torch::IValue output = module_.forward(inputs);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Forward pass took " << elapsed.count() << " seconds." << std::endl;
    
    return output.toTensor();
}
