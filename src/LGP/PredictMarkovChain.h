#pragma once

#include <Search/BanditProcess.h>
#include <LGP/DataManger.h>
#include <Kin/kin.h>
#include <memory>
#pragma push_macro("LOG")
#pragma push_macro("CHECK")
#undef LOG
#undef CHECK

#include <torch/torch.h>

// Restore RAI macros after torch includes
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

// Forward declaration (ModelPredictor is in global namespace)
class ModelPredictor;

namespace rai {

// NodePredictor: Manages prediction models and provides unified prediction interface
struct NodePredictor {
  String predictionType;  // "GT", "myopicGT", "GNN", or "none"
  String solver;          // "GITTINS" or other
  torch::Device device;   // Device for GNN models
  
  // GNN models (only initialized if predictionType == "GNN" && solver == "GITTINS")
  std::shared_ptr<ModelPredictor> model_feasibility_lgp;
  std::shared_ptr<ModelPredictor> model_feasibility_waypoints;
  std::shared_ptr<ModelPredictor> model_qr_feas_lgp;
  std::shared_ptr<ModelPredictor> model_qr_feas_rrt;
  std::shared_ptr<ModelPredictor> model_qr_feas_waypoints;
  std::shared_ptr<ModelPredictor> model_qr_infeas_lgp;
  std::shared_ptr<ModelPredictor> model_qr_infeas_waypoints;
  
  NodePredictor(const String& _predictionType, const String& _solver, const String& _device="cpu", const String& modelDir = "");
  
  // Prediction methods - automatically dispatch to correct implementation
  BanditProcess predict_waypoints(int planID, Configuration& C, StringAA taskPlan);
  BanditProcess predict_rrt(int planID, int numAction, Configuration& C, StringAA taskPlan);
  BanditProcess predict_lgp(int planID, Configuration& C, StringAA taskPlan);
  
private:
  void initializeGNNModels(const std::string& model_dir);
  
  // Helper function to convert GNN predictions to MarkovChain
  MarkovChain convert_tensors_to_markov_chain(
      torch::Tensor& feasibility,
      torch::Tensor& feas_quantiles_tensor,
      torch::Tensor& infeas_quantiles_tensor);
  
  // Test-based prediction method (generates random MarkovChains)
  Array<MarkovChain> test_predict_waypoints_chains(Configuration& C, StringAA taskPlan);
  
  // GNN-based prediction methods (private - return MarkovChain arrays)
  Array<MarkovChain> GNN_predict_waypoints_chains(Configuration& C, StringAA taskPlan);
  Array<MarkovChain> GNN_predict_rrt_chains(Configuration& C, StringAA taskPlan, int actionNum);
  Array<MarkovChain> GNN_predict_lgp_chains(Configuration& C, StringAA taskPlan);
};

// Helper function: Convert quantile predictions to MarkovChain
MarkovChain get_markov_chain_from_quantiles(
    const std::vector<double>& feas_quantiles,
    const std::vector<double>& infeas_quantiles,
    const std::vector<double>& quantile_levels,
    double avgFeas
);

// Ground Truth Bandit Process functions
BanditProcess GT_BP_LGP(int planID);
BanditProcess GT_BP_RRT(int planID, int numAction);
BanditProcess GT_BP_Waypoints(int planID);
BanditProcess myopic_GT_BP_Waypoints(int planID);
BanditProcess myopic_GT_BP_RRT(int planID, int numAction);

} // namespace rai