#pragma once

// Save RAI macros before including torch
#pragma push_macro("LOG")
#pragma push_macro("CHECK")
#undef LOG
#undef CHECK

#include <torch/torch.h>
#include <torch/script.h>

// Restore RAI macros after torch includes
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

#include <LGP/LGP_TAMP_Abstraction.h>
#include <LGP/HeteroGraph.h>
#include <LGP/HeteroGraphConverter.h>

/**
 * @brief Class for loading and running a scripted PyTorch model for constraint prediction
 * 
 * This class encapsulates the functionality of loading a TorchScript model and running
 * predictions on RAI configurations with task plans.
 */
class ModelPredictor {
public:
    /**
     * @brief Construct a new ModelPredictor
     * 
     * @param model_path Path to the scripted PyTorch model (.pt file)
     * @param device Torch device to use (default: CPU)
     */
    ModelPredictor(const std::string& model_path, torch::Device device = torch::kCPU);    
    /**
     * @brief Destructor - cleans up model resources
     */
    ~ModelPredictor();
    
    /**
     * @brief Run prediction on a configuration and task plan
     * 
     * @param C The RAI configuration
     * @param task_plan The sequence of actions
     * @return torch::Tensor The model's prediction output
     */
    torch::Tensor predict(rai::Configuration& C, const StringAA& task_plan, int action_num=-1);
    
    /**
     * @brief Check if the model is loaded
     * 
     * @return true if model is loaded, false otherwise
     */
    bool isLoaded() const { return model_loaded_; }
    
    /**
     * @brief Get the device being used
     * 
     * @return torch::Device The current device
     */
    torch::Device getDevice() const { return device_; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool model_loaded_;
    
    /**
     * @brief Run the model forward pass with the HeteroGraph
     * 
     * @param g The heterogeneous graph input
     * @return torch::Tensor The model output
     */
    torch::Tensor runModelForward(const HeteroGraph& g);
    
    /**
     * @brief Warm up the model with dummy forward passes to trigger JIT compilation
     */
    void warmUp();
};
