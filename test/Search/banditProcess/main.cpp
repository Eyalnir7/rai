// testBanditProcess.cpp
#include <Search/MarkovChain.h>
#include <Search/BanditProcess.h>
#include <Core/array.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

void test_line_bandit() {
    std::cout << "\n=== Test 1: LINE Bandit ===" << std::endl;
    
    // done_transitions = np.array([0.5])
    // done_times = np.array([1])
    // fail_transitions = np.array([0.5])
    // fail_times = np.array([1])
    std::vector<double> done_transitions = {0.5};
    std::vector<int> done_times = {1};
    std::vector<double> fail_transitions = {0.5};
    std::vector<int> fail_times = {1};
    
    MarkovChain markov_chain(done_transitions, done_times, fail_transitions, fail_times, BanditType::LINE);
    
    rai::Array<MarkovChain> chains(1);
    chains(0) = markov_chain;
    
    rai::BanditProcess bandit_process(chains);
    
    auto [stopping_time, gi] = bandit_process.compute_gittins_index(0);
    
    std::cout << "Gittins index: " << gi << ", Stopping time: " << stopping_time << std::endl;
    
    // Expected: (beta**2)/(2*(1+beta)*(1-beta)+beta**2)
    double beta = rai::BanditProcess::beta;  // Should be 0.99 from MarkovChain::beta
    double expected = (beta*beta) / (2*(1+beta)*(1-beta) + beta*beta);
    std::cout << "Expected: " << expected << std::endl;
}

void test_loop_bandit() {
    std::cout << "\n=== Test 2: LOOP Bandit ===" << std::endl;
    
    // done_transitions = np.array([0.6,0.6])
    // done_times = np.array([0,2])
    // fail_transitions = np.array([0.4])
    // fail_times = np.array([1])
    std::vector<double> done_transitions = {0.6, 0.6};
    std::vector<int> done_times = {0, 2};
    std::vector<double> fail_transitions = {0.4};
    std::vector<int> fail_times = {1};
    
    MarkovChain markov_chain(done_transitions, done_times, fail_transitions, fail_times, BanditType::LOOP);
    
    rai::Array<MarkovChain> chains(1);
    chains(0) = markov_chain;
    
    rai::BanditProcess bandit_process(chains);
    
    auto [stopping_time, gi] = bandit_process.compute_gittins_index(0);
    
    std::cout << "Gittins index: " << gi << ", Stopping time: " << stopping_time << std::endl;
    
    // Expected values from Python code
    double b = rai::BanditProcess::beta;  // Should be 0.99
    // V1 = (0.6*b + 0.4*0.6**2*b**3)/(1- 0.4**2*b**2 - 0.6*0.4**2*b**3) # this is if the stopping time is 2
    double V1 = (0.6*b + 0.4*0.36*b*b*b) / (1.0 - 0.16*b*b - 0.6*0.16*b*b*b);
    std::cout << "V1 (if stopping time is 2): " << V1 << std::endl;
    
    // Expected if stopping time is 0
    double V0 = 0.6*b / (1.0 - 0.4*b);
    std::cout << "V0 (if stopping time is 0): " << V0 << std::endl;
}

int main() {
    try {
        test_line_bandit();
        test_loop_bandit();
        
        std::cout << "\n=== All tests completed ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}