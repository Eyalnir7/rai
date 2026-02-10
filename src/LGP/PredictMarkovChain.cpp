#include "PredictMarkovChain.h"
#include "ModelPredictor.h"
#include <Search/MarkovChain.h>
#include <torch/torch.h>
#include <random>
#include <filesystem>

namespace rai {

//===========================================================================
// Helper function to find model file by pattern
//===========================================================================

std::string findModelFile(const std::string& model_dir, const std::string& pattern) {
  try {
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (filename.find(pattern) == 0) {  // Check if filename starts with pattern
          return entry.path().string();
        }
      }
    }
  } catch (const std::exception& e) {
    std::cout << "Error searching for model file in " << model_dir << ": " << e.what() << std::endl;
  }
  
  // If not found, return the original path for backward compatibility
  std::cout << "Warning: Could not find model file matching '" << pattern << "' in " << model_dir << std::endl;
  return model_dir + pattern + ".pt";
}

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
    model_feasibility_lgp = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_FEASIBILITY_LGP"), device);
    model_feasibility_waypoints = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_FEASIBILITY_WAYPOINTS"), device);
    model_qr_feas_lgp = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_QUANTILE_REGRESSION_FEAS_LGP"), device);
    model_qr_feas_rrt = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_QUANTILE_REGRESSION_FEAS_RRT"), device);
    model_qr_feas_waypoints = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_QUANTILE_REGRESSION_FEAS_WAYPOINTS"), device);
    model_qr_infeas_lgp = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_QUANTILE_REGRESSION_INFEAS_LGP"), device);
    model_qr_infeas_waypoints = std::make_shared<ModelPredictor>(findModelFile(model_dir, "model_QUANTILE_REGRESSION_INFEAS_WAYPOINTS"), device);
    
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
  } else if(predictionType == "test"){
        Array<MarkovChain> chains = test_predict_waypoints_chains(C, taskPlan);
        BanditProcess bp(std::move(chains));
        bp.nodeType = NodeType::WaypointsNode;
        return bp;
    }
    else if(predictionType == "GNN") {
    Array<MarkovChain> chains = GNN_predict_waypoints_chains(C, taskPlan);
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::WaypointsNode;
    return bp;
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
    Array<MarkovChain> chains = GNN_predict_rrt_chains(C, taskPlan, numAction);
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::RRTNode;
    return bp;
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
    Array<MarkovChain> chains = GNN_predict_lgp_chains(C, taskPlan);
    BanditProcess bp(std::move(chains));
    bp.nodeType = NodeType::LGPPathNode;
    return bp;
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
 * This function implements the logic from get_chain_probs_from_quantile_values_new_format in Python.
 * It takes predicted quantiles for feasible and infeasible outcomes, along with the
 * quantile levels and average feasibility, and constructs transition probabilities
 * for a MarkovChain using a running sum approach.
 * 
 * @param feas_quantiles Vector of time quantiles for feasible outcomes (integers)
 * @param infeas_quantiles Vector of time quantiles for infeasible outcomes (integers)
 * @param quantile_levels Vector of quantile levels in (0,1], e.g., [0.5, 0.9]
 * @param avgFeas Average feasibility probability in [0,1]
 * @return MarkovChain constructed from the quantile predictions
 */
MarkovChain get_markov_chain_from_quantiles(
    const std::vector<int>& feas_quantiles,
    const std::vector<int>& infeas_quantiles,
    const std::vector<double>& quantile_levels,
    double avgFeas
) {
    int verbose = rai::getParameter<int>("GNN/verbose", 0);
    if(verbose > 1){
      std::cout << "average feasibility: " << avgFeas << std::endl;
      //print the inputs
      std::cout << "Feasible Quantiles: [";
      for (size_t i = 0; i < feas_quantiles.size(); ++i) {
          std::cout << feas_quantiles[i];
          if (i < feas_quantiles.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "Infeasible Quantiles: [";
      for (size_t i = 0; i < infeas_quantiles.size(); ++i) {
          std::cout << infeas_quantiles[i];
          if (i < infeas_quantiles.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "Quantile Levels: [";
      for (size_t i = 0; i < quantile_levels.size(); ++i) {
          std::cout << quantile_levels[i];
          if (i < quantile_levels.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    }


    // Get unique sorted quantiles from both arrays
    std::set<int> quantiles_set;
    for (int q : feas_quantiles) {
        quantiles_set.insert(q);
    }
    for (int q : infeas_quantiles) {
        quantiles_set.insert(q);
    }
    std::vector<int> unique_quantiles(quantiles_set.begin(), quantiles_set.end());
    
    // Create boolean arrays indicating membership in original arrays
    std::vector<bool> in_feas_quantile;
    std::vector<bool> in_infeas_quantile;
    
    for (int Aq : unique_quantiles) {
        bool is_in_feas = std::find(feas_quantiles.begin(), feas_quantiles.end(), Aq) != feas_quantiles.end();
        in_feas_quantile.push_back(is_in_feas);
        
        bool is_in_infeas = std::find(infeas_quantiles.begin(), infeas_quantiles.end(), Aq) != infeas_quantiles.end();
        in_infeas_quantile.push_back(is_in_infeas);
    }
    
    // Compute transition probabilities using running sum approach
    std::vector<double> done_trans;
    std::vector<double> fail_trans;
    double sum_done = 0.0;
    double sum_fail = 0.0;
    double next_transition = 1.0;
    int done_index = 0;
    int fail_index = 0;
    int last_quantile = 0;
    double current_done_transition = 0.0;
    double current_fail_transition = 0.0;
    
    for (size_t i = 0; i < unique_quantiles.size(); ++i) {
        int Aq = unique_quantiles[i];
        current_done_transition = 0.0;
        current_fail_transition = 0.0;
        
        if (in_feas_quantile[i]) {
            // Handle repeated quantiles by removing previous transition
            if (Aq == last_quantile && !done_trans.empty()) {
                done_trans.pop_back();
            }
            double qi = quantile_levels[done_index];
            current_done_transition = (avgFeas * qi - sum_done) / next_transition;
            current_done_transition = std::max(0.0, std::min(1.0, current_done_transition));
            sum_done += current_done_transition;
            done_index++;
            done_trans.push_back(current_done_transition);
        }
        
        if (in_infeas_quantile[i]) {
            // Handle repeated quantiles by removing previous transition
            if (Aq == last_quantile && !fail_trans.empty()) {
                fail_trans.pop_back();
            }
            double qi = quantile_levels[fail_index];
            current_fail_transition = ((1.0 - avgFeas) * qi - sum_fail) / next_transition;
            current_fail_transition = std::max(0.0, std::min(1.0, current_fail_transition));
            sum_fail += current_fail_transition;
            fail_index++;
            fail_trans.push_back(current_fail_transition);
        }
        
        next_transition = next_transition * (1.0 - current_done_transition - current_fail_transition);
        last_quantile = Aq;
    }
    
    // Make feas_quantiles and infeas_quantiles unique and sorted for output
    std::set<int> feas_set(feas_quantiles.begin(), feas_quantiles.end());
    std::vector<int> unique_feas_quantiles(feas_set.begin(), feas_set.end());
    
    std::set<int> infeas_set(infeas_quantiles.begin(), infeas_quantiles.end());
    std::vector<int> unique_infeas_quantiles(infeas_set.begin(), infeas_set.end());
    
    // Add remaining probability to last fail transition
    if (!fail_trans.empty()) {
        if(in_feas_quantile[unique_quantiles.size()-1] && !in_infeas_quantile[unique_quantiles.size()-1]){
            fail_trans.push_back((1.0 - done_trans.back()));
            unique_infeas_quantiles.push_back(unique_quantiles[unique_quantiles.size()-1]);
        }
        else if(!in_feas_quantile[unique_quantiles.size()-1] && in_infeas_quantile[unique_quantiles.size()-1]){
            fail_trans.back() = 1;
        }
        else if(in_feas_quantile[unique_quantiles.size()-1] && in_infeas_quantile[unique_quantiles.size()-1]){
          fail_trans.back() = 1-done_trans.back();
        }
    }
    else{
      done_trans.back() = 1;
    }
    
    // Construct and return MarkovChain
    if(verbose > 1){
      std::cout << "=============================================================================== Constructed MarkovChain =============================================================================== " << endl;
      
      cout << "done_transitions_: [";
      for (size_t i = 0; i < done_trans.size(); ++i) {
          cout << done_trans[i];
          if (i < done_trans.size() - 1) cout << ", ";
      }
      cout << "]" << std::endl;
      
      cout << "done_times_: [";
      for (size_t i = 0; i < unique_feas_quantiles.size(); ++i) {
          cout << unique_feas_quantiles[i];
          if (i < unique_feas_quantiles.size() - 1) cout << ", ";
      }
      cout << "]" << std::endl;
      
      cout << "fail_transitions_: [";
      for (size_t i = 0; i < fail_trans.size(); ++i) {
          cout << fail_trans[i];
          if (i < fail_trans.size() - 1) cout << ", ";
      }
      cout << "]" << std::endl;
      
      cout << "fail_times_: [";
      for (size_t i = 0; i < unique_infeas_quantiles.size(); ++i) {
          cout << unique_infeas_quantiles[i];
          if (i < unique_infeas_quantiles.size() - 1) cout << ", ";
      }
      cout << "]" << std::endl;
      
      cout << "========================================" << std::endl;
    }
    auto mc = MarkovChain(done_trans, unique_feas_quantiles, fail_trans, unique_infeas_quantiles, BanditType::LINE);
    return mc;
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

MarkovChain NodePredictor::convert_tensors_to_markov_chain(
    torch::Tensor& feasibility,
    torch::Tensor& feas_quantiles_tensor,
    torch::Tensor& infeas_quantiles_tensor) {
    
    // Squeeze tensors to remove batch dimension if present
    feasibility = feasibility.squeeze();
    feas_quantiles_tensor = feas_quantiles_tensor.squeeze();
    infeas_quantiles_tensor = infeas_quantiles_tensor.squeeze();
    
    // Convert feasibility tensor (size 1) to double
    double avgFeas = feasibility.item<double>();
    
    // Convert quantile tensors (size 5) to std::vector<int> by rounding up
    std::vector<int> feas_quantiles_vec;
    std::vector<int> infeas_quantiles_vec;
    
    auto feas_accessor = feas_quantiles_tensor.accessor<float, 1>();
    for (int i = 0; i < feas_accessor.size(0); ++i) {
        feas_quantiles_vec.push_back(static_cast<int>(std::ceil(feas_accessor[i])));
    }
    
    auto infeas_accessor = infeas_quantiles_tensor.accessor<float, 1>();
    for (int i = 0; i < infeas_accessor.size(0); ++i) {
        infeas_quantiles_vec.push_back(static_cast<int>(std::ceil(infeas_accessor[i])));
    }
    
    // Define quantile levels (assuming 5 quantiles)
    std::vector<double> quantile_levels = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    // Get MarkovChain from quantiles
    return get_markov_chain_from_quantiles(feas_quantiles_vec, infeas_quantiles_vec, quantile_levels, avgFeas);
}

Array<MarkovChain> NodePredictor::test_predict_waypoints_chains(Configuration& C, StringAA taskPlan){
    // Generate random predictions similar to GNN structure
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> feas_dist(0.3, 0.9);  // Random feasibility between 0.3 and 0.9
    std::uniform_int_distribution<> time_dist(5, 50);      // Random times between 5 and 50
    
    Array<MarkovChain> result;
    
    // 1. Generate waypoints chain
    double waypoints_feas = feas_dist(gen);
    std::vector<int> waypoints_feas_quantiles;
    std::vector<int> waypoints_infeas_quantiles;
    
    // Generate 5 random sorted feasible quantiles
    for(int i = 0; i < 5; ++i) {
        waypoints_feas_quantiles.push_back(time_dist(gen) + i * 10);
    }
    std::sort(waypoints_feas_quantiles.begin(), waypoints_feas_quantiles.end());
    
    // Generate 5 random sorted infeasible quantiles
    for(int i = 0; i < 5; ++i) {
        waypoints_infeas_quantiles.push_back(time_dist(gen) + i * 10);
    }
    std::sort(waypoints_infeas_quantiles.begin(), waypoints_infeas_quantiles.end());
    
    MarkovChain waypoints_mc = get_markov_chain_from_quantiles(
        waypoints_feas_quantiles, 
        waypoints_infeas_quantiles, 
        {0.1, 0.3, 0.5, 0.7, 0.9}, 
        waypoints_feas
    );
    result.append(waypoints_mc);
    
    // 2. Generate RRT chains (one per action)
    int planLength = taskPlan.N;
    for(int i = 0; i < planLength; ++i) {
        std::vector<int> rrt_feas_quantiles;
        
        // Generate 5 random sorted feasible quantiles for RRT
        for(int j = 0; j < 5; ++j) {
            rrt_feas_quantiles.push_back(time_dist(gen) + j * 8);
        }
        std::sort(rrt_feas_quantiles.begin(), rrt_feas_quantiles.end());
        rrt_feas_quantiles.push_back(200);  // Add max time
        
        MarkovChain rrt_mc = get_markov_chain_from_quantiles(
            rrt_feas_quantiles, 
            {}, 
            {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}, 
            1.0
        );
        result.append(rrt_mc);
    }
    
    // 3. Generate LGP chain
    double lgp_feas = feas_dist(gen);
    std::vector<int> lgp_feas_quantiles;
    std::vector<int> lgp_infeas_quantiles;
    
    // Generate 5 random sorted feasible quantiles
    for(int i = 0; i < 5; ++i) {
        lgp_feas_quantiles.push_back(time_dist(gen) + i * 15);
    }
    std::sort(lgp_feas_quantiles.begin(), lgp_feas_quantiles.end());
    
    // Generate 5 random sorted infeasible quantiles
    for(int i = 0; i < 5; ++i) {
        lgp_infeas_quantiles.push_back(time_dist(gen) + i * 15);
    }
    std::sort(lgp_infeas_quantiles.begin(), lgp_infeas_quantiles.end());
    
    MarkovChain lgp_mc = get_markov_chain_from_quantiles(
        lgp_feas_quantiles, 
        lgp_infeas_quantiles, 
        {0.1, 0.3, 0.5, 0.7, 0.9}, 
        lgp_feas
    );
    result.append(lgp_mc);
    
    return result;
}

Array<MarkovChain> NodePredictor::GNN_predict_waypoints_chains(Configuration& C, StringAA taskPlan){
    // std::cout << "\n=== GNN_predict_waypoints_chains ===" << std::endl;
    
    torch::NoGradGuard no_grad;
    
    torch::Tensor feasibility = model_feasibility_waypoints->predict(C, taskPlan).detach();
    torch::Tensor feas_quantiles = model_qr_feas_waypoints->predict(C, taskPlan).detach();
    torch::Tensor infeas_quantiles = model_qr_infeas_waypoints->predict(C, taskPlan).detach();
    
    // Apply sigmoid to feasibility
    feasibility = torch::sigmoid(feasibility);
    
    // Apply softplus to quantiles (except first entry)
    feas_quantiles = feas_quantiles.squeeze();
    infeas_quantiles = infeas_quantiles.squeeze();
    if (feas_quantiles.size(0) > 1) {
        feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    if (infeas_quantiles.size(0) > 1) {
        infeas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(infeas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    
    // std::cout << "Waypoints predictions:" << std::endl;
    // std::cout << "  feasibility shape: " << feasibility.sizes() << ", value: " << feasibility << std::endl;
    // std::cout << "  feas_quantiles shape: " << feas_quantiles.sizes() << ", values: " << feas_quantiles << std::endl;
    // std::cout << "  infeas_quantiles shape: " << infeas_quantiles.sizes() << ", values: " << infeas_quantiles << std::endl;

    Array<MarkovChain> result;
    result.append(convert_tensors_to_markov_chain(feasibility, feas_quantiles, infeas_quantiles));
    int planLength = taskPlan.N;
    Array<MarkovChain> rrtChains;;
    for(int i = 0; i < planLength; ++i) {
        torch::Tensor rrt_feas_quantiles = model_qr_feas_rrt->predict(C, taskPlan, i).detach();
        rrt_feas_quantiles = rrt_feas_quantiles.squeeze();  // Remove batch dimension
        
        // Apply softplus to all entries except the first
        if (rrt_feas_quantiles.size(0) > 1) {
            rrt_feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
                torch::nn::functional::softplus(rrt_feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}));
        }
        
        // std::cout << "RRT action " << i << " feas_quantiles shape: " << rrt_feas_quantiles.sizes() << ", values: " << rrt_feas_quantiles << std::endl;
        std::vector<int> feas_quantiles_vec;
        auto feas_accessor = rrt_feas_quantiles.accessor<float, 1>();
        for (int i = 0; i < feas_accessor.size(0); ++i) {
            feas_quantiles_vec.push_back(static_cast<int>(std::ceil(feas_accessor[i])));
        }
        feas_quantiles_vec.push_back(200);
        MarkovChain rrtWaypointsMC = get_markov_chain_from_quantiles(feas_quantiles_vec, {}, std::vector<double>{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}, 1.0);
        result.append(rrtWaypointsMC);
    }
    torch::Tensor lgp_feasibility = model_feasibility_lgp->predict(C, taskPlan).detach();
    torch::Tensor lgp_feas_quantiles = model_qr_feas_lgp->predict(C, taskPlan).detach();
    torch::Tensor lgp_infeas_quantiles = model_qr_infeas_lgp->predict(C, taskPlan).detach();
    
    // Apply sigmoid to feasibility
    lgp_feasibility = torch::sigmoid(lgp_feasibility);
    
    // Apply softplus to quantiles (except first entry)
    lgp_feas_quantiles = lgp_feas_quantiles.squeeze();
    lgp_infeas_quantiles = lgp_infeas_quantiles.squeeze();
    if (lgp_feas_quantiles.size(0) > 1) {
        lgp_feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(lgp_feas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    if (lgp_infeas_quantiles.size(0) > 1) {
        lgp_infeas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(lgp_infeas_quantiles.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    
    // std::cout << "LGP predictions:" << std::endl;
    // std::cout << "  feasibility shape: " << lgp_feasibility.sizes() << ", value: " << lgp_feasibility << std::endl;
    // std::cout << "  feas_quantiles shape: " << lgp_feas_quantiles.sizes() << ", values: " << lgp_feas_quantiles << std::endl;
    // std::cout << "  infeas_quantiles shape: " << lgp_infeas_quantiles.sizes() << ", values: " << lgp_infeas_quantiles << std::endl;
    
    result.append(convert_tensors_to_markov_chain(lgp_feasibility, lgp_feas_quantiles, lgp_infeas_quantiles));
    // std::cout << "=== End GNN_predict_waypoints_chains ===\n" << std::endl;
    return result;
}

Array<MarkovChain> NodePredictor::GNN_predict_rrt_chains(Configuration& C, StringAA taskPlan, int actionNum){
    // TODO: Implement GNN-based prediction for RRT
    // Use model_qr_feas_rrt
    cout << "not implemented yet" << endl;
    Array<MarkovChain> chains;
    return chains;
}

Array<MarkovChain> NodePredictor::GNN_predict_lgp_chains(Configuration& C, StringAA taskPlan){
    // Use model_feasibility_lgp, model_qr_feas_lgp, and model_qr_infeas_lgp
    torch::NoGradGuard no_grad;
    
    torch::Tensor feasibility = model_feasibility_lgp->predict(C, taskPlan).detach();
    torch::Tensor feas_quantiles_tensor = model_qr_feas_lgp->predict(C, taskPlan).detach();
    torch::Tensor infeas_quantiles_tensor = model_qr_infeas_lgp->predict(C, taskPlan).detach();
    
    // Apply sigmoid to feasibility
    feasibility = torch::sigmoid(feasibility);
    
    // Apply softplus to quantiles (except first entry)
    feas_quantiles_tensor = feas_quantiles_tensor.squeeze();
    infeas_quantiles_tensor = infeas_quantiles_tensor.squeeze();
    if (feas_quantiles_tensor.size(0) > 1) {
        feas_quantiles_tensor.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(feas_quantiles_tensor.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    if (infeas_quantiles_tensor.size(0) > 1) {
        infeas_quantiles_tensor.index({torch::indexing::Slice(1, torch::indexing::None)}) = 
            torch::nn::functional::softplus(infeas_quantiles_tensor.index({torch::indexing::Slice(1, torch::indexing::None)}));
    }
    
    // Convert tensors to MarkovChain
    MarkovChain mc = convert_tensors_to_markov_chain(feasibility, feas_quantiles_tensor, infeas_quantiles_tensor);
    
    Array<MarkovChain> chains;
    chains.append(mc);
    return chains;
}

} // namespace rai