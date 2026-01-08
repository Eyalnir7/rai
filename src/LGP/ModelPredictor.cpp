#include "ModelPredictor.h"
#include <iostream>
#include <chrono>

ModelPredictor::ModelPredictor(const std::string& model_path, torch::Device device)
    : device_(device), model_loaded_(false) {
    try {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();  // Set to evaluation mode
        
        // Disable gradient tracking and profiling to prevent memory accumulation
        torch::jit::getProfilingMode() = false;
        torch::jit::getExecutorMode() = false;
        
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

ModelPredictor::~ModelPredictor() {
    // Clear any cached execution plans
    if (model_loaded_) {
        // Force cleanup of internal caches
        module_ = torch::jit::Module();
    }
}

torch::Tensor ModelPredictor::predict(rai::Configuration& C, const StringAA& task_plan, int action_num) {
    if (!model_loaded_) {
        std::cerr << "Model is not loaded. Cannot make predictions." << std::endl;
        return torch::Tensor();
    }

    c10::InferenceMode guard;
    
    torch::Tensor output;
    {
        // Use NoGradGuard to prevent gradient tracking
        torch::NoGradGuard no_grad;
        
        // Convert configuration and task plan to intermediate heterogeneous data
        IntermediateHeteroData hetero_data = get_hetero_data_input(C, task_plan, device_, action_num);
        
        // Convert to HeteroGraph
        HeteroGraph g = convertToHeteroGraph(hetero_data);
        
        // Run the model
        output = runModelForward(g);
        
        // Clone and detach to fully break from any computation graph
        output = output;
    }
    
    return output;
}

void ModelPredictor::warmUp() {
    torch::NoGradGuard no_grad;
    
    // Create dummy inputs with typical sizes
    torch::Dict<std::string, torch::Tensor> x_dict, times_dict, batch_dict, edge_index_dict;
    
    // Add some dummy tensors (adjust sizes based on typical input)
    // x_dict.insert("pick", torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    // x_dict.insert("place", torch::zeros({4, 10}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("ssBox", torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("object", torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("place_frame", torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    x_dict.insert("ssCylinder", torch::zeros({4, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    
    times_dict.insert("pick", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("place", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("ssBox", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("object", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("place_frame", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    times_dict.insert("ssCylinder", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    
    batch_dict.insert("pick", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    batch_dict.insert("place", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    batch_dict.insert("ssBox", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    batch_dict.insert("object", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    batch_dict.insert("place_frame", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    batch_dict.insert("ssCylinder", torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    
    edge_index_dict.insert("ssBox___pick_edge___pick", torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    edge_index_dict.insert("object___place_edge___place", torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_dict);
    inputs.push_back(times_dict);
    inputs.push_back(edge_index_dict);
    inputs.push_back(batch_dict);
    
    // Run a few warm-up passes to trigger JIT compilation
    for (int i = 0; i < 3; i++) {
        auto output = module_.forward(inputs);
    }
}

torch::Tensor ModelPredictor::runModelForward(const HeteroGraph& g) {
    // Convert unordered_map → torch::Dict<string, Tensor>
    torch::Dict<std::string, torch::Tensor> x_dict, times_dict, edge_index_dict, batch_dict;

    for (const auto& kv : g.x_dict)
        x_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.times_dict)
        times_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.edge_index_dict)
        edge_index_dict.insert(kv.first, kv.second);

    for (const auto& kv : g.batch_dict)
        batch_dict.insert(kv.first, kv.second);

    // Prepare inputs vector in correct order for bestScriptedModel.py
    // forward(x_dict, times_dict, edge_index_dict, batch_dict)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_dict);
    inputs.push_back(times_dict);
    inputs.push_back(edge_index_dict);
    inputs.push_back(batch_dict);

    // Forward pass with timing
    auto start = std::chrono::high_resolution_clock::now();
    torch::IValue output = module_.forward(inputs);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if(rai::getParameter<int>("GNN/verbose", 1) > 0) std::cout << "Forward pass took " << elapsed.count() << " seconds." << std::endl;
    
    // Clone and detach to prevent accumulation
    torch::Tensor result = output.toTensor();
    
    return result;
}
