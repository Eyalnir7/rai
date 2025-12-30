#include "PredictMarkovChain.h"
#include "ModelPredictor.h"
#include <Search/MarkovChain.h>
#include <torch/torch.h>

namespace rai {

//===========================================================================
// NodePredictor Implementation
//===========================================================================

NodePredictor::NodePredictor(const String& _predictionType, const String& _solver, const String& modelDir)
  : predictionType(_predictionType), solver(_solver) {
  
  // Only initialize GNN models if using GNN prediction with GITTINS solver
  if(predictionType == "GNN" && solver == "GITTINS" && modelDir.N > 0) {
    initializeGNNModels(modelDir.p);
  }
}

void NodePredictor::initializeGNNModels(const std::string& model_dir) {
  std::cout << "Initializing GNN models from directory: " << model_dir << std::endl;
  
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
  
  try {
    model_feasibility_lgp = std::make_shared<ModelPredictor>(model_dir + "model_FEASIBILITY_LGP.pt", device);
    model_feasibility_waypoints = std::make_shared<ModelPredictor>(model_dir + "model_FEASIBILITY_WAYPOINTS.pt", device);
    model_qr_feas_lgp = std::make_shared<ModelPredictor>(model_dir + "model_QUANTILE_REGRESSION_FEAS_LGP.pt", device);
    model_qr_feas_rrt = std::make_shared<ModelPredictor>(model_dir + "model_QUANTILE_REGRESSION_FEAS_RRT.pt", device);
    model_qr_feas_waypoints = std::make_shared<ModelPredictor>(model_dir + "model_QUANTILE_REGRESSION_FEAS_WAYPOINTS.pt", device);
    model_qr_infeas_lgp = std::make_shared<ModelPredictor>(model_dir + "model_QUANTILE_REGRESSION_INFEAS_LGP.pt", device);
    model_qr_infeas_waypoints = std::make_shared<ModelPredictor>(model_dir + "model_QUANTILE_REGRESSION_INFEAS_WAYPOINTS.pt", device);
    
    std::cout << "All GNN models loaded successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Error loading GNN models: " << e.what() << std::endl;
    // Reset all pointers on failure
    model_feasibility_lgp.reset();
    model_feasibility_waypoints.reset();
    model_qr_feas_lgp.reset();
    model_qr_feas_rrt.reset();
    model_qr_feas_waypoints.reset();
    model_qr_infeas_lgp.reset();
    model_qr_infeas_waypoints.reset();
  }
}

BanditProcess NodePredictor::predict_waypoints(int planID, Configuration& C, StringAA taskPlan) {
  if(predictionType == "GT") {
    return GT_BP_Waypoints(planID);
  } else if(predictionType == "myopicGT") {
    return myopic_GT_BP_Waypoints(planID);
  } else if(predictionType == "GNN") {
    return GNN_predict_waypoints(C, taskPlan);
  } else {
    // Default: return empty BanditProcess for "none" or unknown types
    Array<MarkovChain> chains;
    return BanditProcess(std::move(chains));
  }
}

BanditProcess NodePredictor::predict_rrt(int planID, int numAction, Configuration& C, StringAA taskPlan) {
  if(predictionType == "GT") {
    return GT_BP_RRT(planID, numAction);
  } else if(predictionType == "myopicGT") {
    return myopic_GT_BP_RRT(planID, numAction);
  } else if(predictionType == "GNN") {
    return GNN_predict_rrt(C, taskPlan, numAction);
  } else {
    // Default: return empty BanditProcess for "none" or unknown types
    Array<MarkovChain> chains;
    return BanditProcess(std::move(chains));
  }
}

BanditProcess NodePredictor::predict_lgp(int planID, Configuration& C, StringAA taskPlan) {
  if(predictionType == "GT") {
    return GT_BP_LGP(planID);
  } else if(predictionType == "myopicGT") {
    return GT_BP_LGP(planID);  // myopicGT uses same for LGP
  } else if(predictionType == "GNN") {
    return GNN_predict_lgp(C, taskPlan);
  } else {
    // Default: return empty BanditProcess for "none" or unknown types
    Array<MarkovChain> chains;
    return BanditProcess(std::move(chains));
  }
}

//===========================================================================
// Helper function: Convert quantile predictions to MarkovChain
//===========================================================================

/**
 * @brief Construct a MarkovChain from predicted quantiles and average feasibility
 * 
 * This function implements the logic from get_chain_probs_from_quantile_values in Python.
 * It takes predicted quantiles for feasible and infeasible outcomes, along with the
 * quantile levels and average feasibility, and constructs transition probabilities
 * for a MarkovChain.
 * 
 * @param feas_quantiles Vector of time quantiles for feasible outcomes (sorted)
 * @param infeas_quantiles Vector of time quantiles for infeasible outcomes (sorted)
 * @param quantile_levels Vector of quantile levels in (0,1], e.g., [0.5, 0.9]
 * @param avgFeas Average feasibility probability in [0,1]
 * @return MarkovChain constructed from the quantile predictions
 */
MarkovChain get_markov_chain_from_quantiles(
    const std::vector<double>& feas_quantiles,
    const std::vector<double>& infeas_quantiles,
    const std::vector<double>& quantile_levels,
    double avgFeas
) {
    // Merge and sort all unique quantile times (rounded to ints)
    std::set<int> quantiles_set;
    for (double q : feas_quantiles) {
        quantiles_set.insert(static_cast<int>(std::round(q)));
    }
    for (double q : infeas_quantiles) {
        quantiles_set.insert(static_cast<int>(std::round(q)));
    }
    std::vector<int> quantiles(quantiles_set.begin(), quantiles_set.end());
    
    std::vector<double> feas_probs;
    std::vector<double> infeas_probs;
    std::vector<int> times;
    
    for (int q : quantiles) {
        double feas_prob = 0.0;
        double infeas_prob = 0.0;
        
        // Check if q matches any feas_quantile (rounded)
        auto feas_it = std::find_if(feas_quantiles.begin(), feas_quantiles.end(),
                                     [q](double val) { return static_cast<int>(std::round(val)) == q; });
        
        if (feas_it != feas_quantiles.end()) {
            int quantile_idx = std::distance(feas_quantiles.begin(), feas_it);
            double current_level = quantile_levels[quantile_idx];
            
            // Find previous quantile level (or 0 if first)
            double prev_level = (quantile_idx > 0) ? quantile_levels[quantile_idx - 1] : 0.0;
            double delta_level = current_level - prev_level;
            
            // Find corresponding infeasible quantile level
            auto infeas_upper = std::upper_bound(infeas_quantiles.begin(), infeas_quantiles.end(), q);
            int infeas_quantile_index = std::distance(infeas_quantiles.begin(), infeas_upper);
            double infeas_level = (infeas_quantile_index < quantile_levels.size()) 
                                  ? quantile_levels[infeas_quantile_index] : 1.0;
            
            // Compute feasible transition probability
            double denominator = 1.0 - avgFeas * current_level - (1.0 - avgFeas) * infeas_level 
                               + avgFeas * delta_level;
            if (std::abs(denominator) > 1e-10) {
                feas_prob = avgFeas * delta_level / denominator;
            }
        }
        
        // Check if q matches any infeas_quantile (rounded)
        auto infeas_it = std::find_if(infeas_quantiles.begin(), infeas_quantiles.end(),
                                       [q](double val) { return static_cast<int>(std::round(val)) == q; });
        
        if (infeas_it != infeas_quantiles.end()) {
            int quantile_idx = std::distance(infeas_quantiles.begin(), infeas_it);
            double current_level = quantile_levels[quantile_idx];
            
            // Find previous quantile level (or 0 if first)
            double prev_level = (quantile_idx > 0) ? quantile_levels[quantile_idx - 1] : 0.0;
            double delta_level = current_level - prev_level;
            
            // Find corresponding feasible quantile level
            auto feas_upper = std::upper_bound(feas_quantiles.begin(), feas_quantiles.end(), q);
            int feas_quantile_index = std::distance(feas_quantiles.begin(), feas_upper);
            double feas_level = (feas_quantile_index < quantile_levels.size()) 
                              ? quantile_levels[feas_quantile_index] : 1.0;
            
            // Compute infeasible transition probability
            double denominator = 1.0 - avgFeas * feas_level - (1.0 - avgFeas) * current_level 
                               + (1.0 - avgFeas) * delta_level;
            if (std::abs(denominator) > 1e-10) {
                infeas_prob = (1.0 - avgFeas) * delta_level / denominator;
            }
        }
        
        // Only add non-zero transitions
        if (feas_prob > 0.0) {
            feas_probs.push_back(feas_prob);
            times.push_back(q);
        }
        if (infeas_prob > 0.0) {
            infeas_probs.push_back(infeas_prob);
            times.push_back(q);
        }
    }
    
    // Separate times for done and fail transitions
    std::vector<int> done_times;
    std::vector<int> fail_times;
    
    for (size_t i = 0; i < feas_probs.size(); ++i) {
        done_times.push_back(times[i]);
    }
    for (size_t i = 0; i < infeas_probs.size(); ++i) {
        fail_times.push_back(times[feas_probs.size() + i]);
    }
    
    // Construct and return MarkovChain
    return MarkovChain(feas_probs, done_times, infeas_probs, fail_times, BanditType::LINE);
}

//===========================================================================
// Ground Truth Bandit Process functions
//===========================================================================

BanditProcess GT_BP_LGP(int planID) {
    // Get LGP transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    TransitionData data = dataMgr.getLGPTransitions(planID);
    
    // TODO: Convert TransitionData to Array<MarkovChain>
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::LGPPathNode;
    
    return bp;
}

BanditProcess GT_BP_RRT(int planID, int numAction) {
    // Get RRT transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto [data, planLength] = dataMgr.getRRTTransitions(planID, numAction);
    
    // TODO: Convert TransitionData to Array<MarkovChain>
    Array<MarkovChain> chains;
    MarkovChain mc(
            data.done_transitions,
            data.done_times,
            data.fail_transitions,
            data.fail_times,
            BanditType::LINE
        );
    chains.append(mc);
    for(int i = numAction+1; i < planLength; ++i) {
        auto [data_i, planLength_i] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc_i(
            data_i.done_transitions,
            data_i.done_times,
            data_i.fail_transitions,
            data_i.fail_times,
            BanditType::LINE
        );
        chains.append(mc_i);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.append(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::RRTNode;
    
    return bp;
}

BanditProcess GT_BP_Waypoints(int planID) {
    // Get waypoint transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto data = dataMgr.getWaypointTransitions(planID);
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);

    // Get planLength from first RRT call
    auto [data_0, planLength] = dataMgr.getRRTTransitions(planID, 0);
    for(int i = 0; i < planLength; ++i) {
        auto [data_i, planLength_i] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc_i(
            data_i.done_transitions,
            data_i.done_times,
            data_i.fail_transitions,
            data_i.fail_times,
            BanditType::LINE
        );
        chains.append(mc_i);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.append(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::WaypointsNode;
    
    return bp;
}

BanditProcess myopic_GT_BP_Waypoints(int planID) {
    // Get waypoint transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto data = dataMgr.getWaypointTransitions(planID);
    
    // Only consider the waypoints chain itself (myopic)
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // Create and configure BanditProcess
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::WaypointsNode;
    
    return bp;
}

BanditProcess myopic_GT_BP_RRT(int planID, int numAction) {
    // Get RRT transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto [data, planLength] = dataMgr.getRRTTransitions(planID, numAction);
    
    // Only consider the RRT chain itself (myopic)
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // Create and configure BanditProcess
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::RRTNode;
    
    return bp;
}

BanditProcess NodePredictor::GNN_predict_waypoints(Configuration& C, StringAA taskPlan){
    torch::Tensor feasibility = model_feasibility_waypoints->predict(C, taskPlan);
    torch::Tensor feas_quantiles = model_qr_feas_waypoints->predict(C, taskPlan);
    torch::Tensor infeas_quantiles = model_qr_infeas_waypoints->predict(C, taskPlan);



    Array<MarkovChain> chains;
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::WaypointsNode;
    return bp;
}

BanditProcess NodePredictor::GNN_predict_rrt(Configuration& C, StringAA taskPlan, int actionNum){
    // TODO: Implement GNN-based prediction for RRT
    // Use model_qr_feas_rrt
    Array<MarkovChain> chains;
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::RRTNode;
    return bp;
}

BanditProcess NodePredictor::GNN_predict_lgp(Configuration& C, StringAA taskPlan){
    // TODO: Implement GNN-based prediction for LGP
    // Use model_feasibility_lgp, model_qr_feas_lgp, and model_qr_infeas_lgp
    Array<MarkovChain> chains;
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::LGPPathNode;
    return bp;
}

} // namespace rai