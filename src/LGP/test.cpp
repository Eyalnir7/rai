#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

void get_markov_chain_from_quantiles(
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
    
    // Print results
    std::cout << "Done times: ";
    for (int t : done_times) std::cout << t << " ";
    std::cout << "\nDone probs: ";
    for (double p : feas_probs) std::cout << p << " ";
    std::cout << "\n\nFail times: ";
    for (int t : fail_times) std::cout << t << " ";
    std::cout << "\nFail probs: ";
    for (double p : infeas_probs) std::cout << p << " ";
    std::cout << "\n\n";
}

int main() {
    // Test case 1: Simple example
    std::cout << "Test 1: Basic quantiles\n";
    std::vector<double> feas1 = {1, 3, 5, 7};
    std::vector<double> infeas1 = {2, 4, 6, 8};
    std::vector<double> levels1 = {0.5, 1};
    double avgFeas1 = 0.6;
    get_markov_chain_from_quantiles(feas1, infeas1, levels1, avgFeas1);
    
    // Test case 2: Same quantiles
    std::cout << "Test 2: Overlapping quantiles\n";
    std::vector<double> feas2 = {10.0, 20.0};
    std::vector<double> infeas2 = {10.0, 20.0};
    std::vector<double> levels2 = {0.5, 0.9};
    double avgFeas2 = 0.5;
    get_markov_chain_from_quantiles(feas2, infeas2, levels2, avgFeas2);
    
    // Test case 3: Float quantiles that round to same int
    std::cout << "Test 3: Float quantiles\n";
    std::vector<double> feas3 = {10.3, 20.7};
    std::vector<double> infeas3 = {15.1, 25.9};
    std::vector<double> levels3 = {0.33, 0.67};
    double avgFeas3 = 0.7;
    get_markov_chain_from_quantiles(feas3, infeas3, levels3, avgFeas3);
    
    return 0;
}