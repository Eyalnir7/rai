#pragma once

#include <Search/BanditProcess.h>
#include <LGP/DataManger.h>
#include <Kin/kin.h>
#include <memory>

// Forward declaration (ModelPredictor is in global namespace)
class ModelPredictor;

namespace rai {

// NodePredictor: Manages prediction models and provides unified prediction interface
struct NodePredictor {
  String predictionType;  // "GT", "myopicGT", "GNN", or "none"
  String solver;          // "GITTINS" or other
  
  // GNN models (only initialized if predictionType == "GNN" && solver == "GITTINS")
  std::shared_ptr<ModelPredictor> model_feasibility_lgp;
  std::shared_ptr<ModelPredictor> model_feasibility_waypoints;
  std::shared_ptr<ModelPredictor> model_qr_feas_lgp;
  std::shared_ptr<ModelPredictor> model_qr_feas_rrt;
  std::shared_ptr<ModelPredictor> model_qr_feas_waypoints;
  std::shared_ptr<ModelPredictor> model_qr_infeas_lgp;
  std::shared_ptr<ModelPredictor> model_qr_infeas_waypoints;
  
  NodePredictor(const String& _predictionType, const String& _solver, const String& modelDir = "");
  
  // Prediction methods - automatically dispatch to correct implementation
  BanditProcess predict_waypoints(int planID, Configuration& C, StringAA taskPlan);
  BanditProcess predict_rrt(int planID, int numAction, Configuration& C, StringAA taskPlan);
  BanditProcess predict_lgp(int planID, Configuration& C, StringAA taskPlan);
  
private:
  void initializeGNNModels(const std::string& model_dir);
  
  // GNN-based prediction methods (private - only used internally)
  BanditProcess GNN_predict_waypoints(Configuration& C, StringAA taskPlan);
  BanditProcess GNN_predict_rrt(Configuration& C, StringAA taskPlan, int actionNum);
  BanditProcess GNN_predict_lgp(Configuration& C, StringAA taskPlan);
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