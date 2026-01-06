#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

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
void get_markov_chain_from_quantiles(
    const std::vector<int>& feas_quantiles,
    const std::vector<int>& infeas_quantiles,
    const std::vector<double>& quantile_levels,
    double avgFeas
) {
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
    std::cout << "=============================================================================== Constructed MarkovChain =============================================================================== " << std::endl;
    
    std::cout << "done_transitions_: [";
    for (size_t i = 0; i < done_trans.size(); ++i) {
        std::cout << done_trans[i];
        if (i < done_trans.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "done_times_: [";
    for (size_t i = 0; i < unique_feas_quantiles.size(); ++i) {
        std::cout << unique_feas_quantiles[i];
        if (i < unique_feas_quantiles.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "fail_transitions_: [";
    for (size_t i = 0; i < fail_trans.size(); ++i) {
        std::cout << fail_trans[i];
        if (i < fail_trans.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "fail_times_: [";
    for (size_t i = 0; i < unique_infeas_quantiles.size(); ++i) {
        std::cout << unique_infeas_quantiles[i];
        if (i < unique_infeas_quantiles.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "========================================" << std::endl;
    return;
}

int main() {
    // Test case 1: Simple example
    std::cout << "Test 1: Basic quantiles\n";
    std::vector<int> feas1 = {69, 38, 36, 46, 92};
    std::vector<int> infeas1 = {134, 59, 60, 99, 206};
    std::vector<double> levels1 = {0.100, 0.300, 0.500, 0.700, 0.900};
    double avgFeas1 = 0.948;
    get_markov_chain_from_quantiles(feas1, infeas1, levels1, avgFeas1);
    
    return 0;
}