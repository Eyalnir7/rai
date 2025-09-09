#include <Search/ProjectionBanditProcess.h> 
#include <Search/PlanDataManager.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <Core/array.h>

namespace rai {

    inline double oneStepAnalysisLine(
        const PlanDataManager::ComputeData& computeData,
        double beta,
        double nextLevelGI)
    {
        // ...existing validation code...
        Array<double> succ_probs = computeData.succ_probs;
        Array<double> fail_probs = computeData.fail_probs;
        Array<double> next_probs = computeData.next_probs;
        Array<int> t = rai::convert<int>(computeData.t);

        const size_t n = succ_probs.N;
        // nextLevelGI /= (1 - beta);

        // Use rai arrays instead of Eigen
        arr A(n, n);  // Matrix A
        arr b(n);     // Vector b
        arr c(n);     // Vector c
        A.setZero();
        b.setZero();
        c.setZero();

        for (size_t i = 0; i < n; ++i) {
            
            if (i < n - 1) {
                double effective_beta = std::pow(beta, t(i+1) - t(i));
                A(i, i+1) = next_probs(i) * effective_beta;
            }

            b(i) = succ_probs(i) * beta * nextLevelGI;
            c(i) = b(i) + 1;
        }

        // Build (I - A)
        arr IminusA(n, n);
        IminusA.setId();  // Identity matrix
        IminusA -= A;     // I - A

        arr x = inverse(IminusA) * b;

        return x(0)/c(0);
    }

    // returns the numerator of the gittins index expression. To get the gittins index, multiply by (1 - beta)
    inline double oneStepAnalysisLoop(
        const PlanDataManager::ComputeData& computeData,
        double beta,
        double nextLevelGI)
    {
        // ...existing validation code...
        Array<double> succ_probs = computeData.succ_probs;
        Array<double> fail_probs = computeData.fail_probs;
        Array<double> next_probs = computeData.next_probs;
        Array<int> t = rai::convert<int>(computeData.t);

        const size_t n = succ_probs.N;
        // nextLevelGI /= (1 - beta);

        // Use rai arrays instead of Eigen
        arr A(n, n);  // Matrix A
        arr b(n);     // Vector b
        A.setZero();
        b.setZero();

        for (size_t i = 0; i < n; ++i) {
            A(i, 0) = fail_probs(i) * beta;
            
            if (i < n - 1) {
                double effective_beta = std::pow(beta, t(i+1) - t(i));
                A(i, i+1) = next_probs(i) * effective_beta;
            }

            b(i) = succ_probs(i) * beta * nextLevelGI;
        }

        // Build (I - A)
        arr IminusA(n, n);
        IminusA.setId();  // Identity matrix
        IminusA -= A;     // I - A

               // print the A matrix
        // std::cout << "A matrix:" << std::endl;
        // for (size_t i = 0; i < n; ++i) {
        //     for (size_t j = 0; j < n; ++j) {
        //         std::cout << IminusA(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // double det = determinant(IminusA);
        // std::cout << "Matrix determinant: " << det << std::endl;

        // if (std::abs(det) < 1e-12) {
        //     std::cout << "Warning: Matrix is nearly singular!" << std::endl;
        //     // Handle the singular case
        //     // return 0.0; // or some default value
        // }

        arr x = inverse(IminusA) * b;

        return x(0);
    }

    double geometric_sum(double beta, int t1, int t2) {
        if (t2 <= t1) {
            return 0.0;
        }
        if (beta == 1.0) {
            return static_cast<double>(t2 - t1);
        }
        return std::pow(beta, t1) * (1.0 - std::pow(beta, t2 - t1)) / (1.0 - beta);
    }

    std::pair<double, int> single_chain_gittins_index(
    const PlanDataManager::ComputeData& computeData,
    double beta)
    {
        Array<double> succ_probs = computeData.succ_probs;
        Array<double> fail_probs = computeData.fail_probs;
        Array<double> next_probs = computeData.next_probs;
        Array<int> t = rai::convert<int>(computeData.t);
        if (succ_probs.N == 0 || fail_probs.N == 0 || next_probs.N == 0 || t.N == 0) {
            throw std::invalid_argument("Input vectors cannot be empty");
        }

        if (succ_probs.N != fail_probs.N ||
            succ_probs.N != next_probs.N ||
            succ_probs.N != t.N) {
            throw std::invalid_argument("All input vectors must have the same size");
        }

        int current_t = t(0);
        double sum_P = 0.0;
        double sum_Q = geometric_sum(beta, 0, current_t);
        double product_g = 1.0;
        double P = succ_probs(0);
        double F = fail_probs(0);
        double max_index = -std::numeric_limits<double>::infinity();
        int argmax_index = 0;
        int previous_t = current_t;
        
        // i is the number of nodes in the continuation set except for the node DONE
        for (size_t n = 0; n < succ_probs.N; ++n) {
            // get the index of n'th entry
            previous_t = current_t;
            double numerator = (1.0 - beta) * sum_P + P * std::pow(beta, current_t);
            double denominator = (1.0 - beta) * sum_Q + P * std::pow(beta, current_t);

            double v_n = numerator / denominator;
            if (v_n > max_index) {
                max_index = v_n;
                argmax_index = current_t;
            }

            if (n + 1 < succ_probs.N) {
                current_t = t(n + 1);
            }
            sum_P += P * geometric_sum(beta, previous_t, current_t);
            sum_Q += (1.0 - F) * geometric_sum(beta, previous_t, current_t);
            product_g *= next_probs(n);
            if (n + 1 < succ_probs.N) {
                P = P + product_g * succ_probs(n + 1);
                F = F + product_g * fail_probs(n + 1);
            }
        }

        double gittins_index = max_index;
        int optimal_stopping_time = argmax_index;
        // std::cout << "Single chain Gittins index: " << gittins_index << " optimal stopping time single chain: " << optimal_stopping_time << std::endl;
        return std::make_pair(gittins_index, optimal_stopping_time);
    }

    ProjectionBanditProcess::ProjectionBanditProcess(const TaskPlan taskPlan, rai::NodeType nodeType) 
        : nodeType(nodeType) {
        empty = false;
        update_probs(taskPlan, nodeType);
    }

    void ProjectionBanditProcess::updatePlanData(const TaskPlan taskPlan, rai::NodeType nodeType) {
        if (!planDataManager.isInitialized()) {
            planDataManager.loadPlansFile("plansAround.g"); 
        }

        planDataManager.getProjectedPlanData(taskPlan, nodeType, planData);
    }

    void ProjectionBanditProcess::update_probs(const TaskPlan taskPlan, rai::NodeType nodeType) {

        if(planData.N == 0){
            updatePlanData(taskPlan, nodeType);
        }
        // iterate throught the planData array backwards
        // std::cout << "Updating probabilities for " << taskPlan << std::endl;
        double nextGI = 1/(1 - beta);
        for (int i = planData.N - 1; i >= 0; --i) {
            const PlanDataManager::ComputeData& data = planData(i);
            auto [gittins_index, optimal_stopping_time] = single_chain_gittins_index(data, beta);
            const PlanDataManager::ComputeData currentData = data.sliceByThreshold(optimal_stopping_time);
            nextGI = oneStepAnalysisLoop(currentData, beta, nextGI);
            // 0.176978
            // arr succ_probs = {0,0,0,0,0,0,0,0.5,0.1};
            // arr fail_probs = {0.5,0,0,0,0,0,0,0,0.9};
            // arr next_probs = {0.5,1,1,1,1,1,1,0.5,0};
            // arr t = {1,2,3,4,5,6,7,8,9};
            // double beta = 0.75;
            // PlanDataManager::ComputeData test = PlanDataManager::ComputeData(t, succ_probs, fail_probs, next_probs);
            // cout << "test================================="<< oneStepAnalysis(test, beta, 1) << std::endl;
            // std::cout << "Step " << i << " Gittins index (unnormalized): " << nextGI << " optimal stopping time: " << optimal_stopping_time << std::endl;
            // nextGI /= (1 - beta);
            // std::cout << "Gittins index: " << nextGI << " optimal stopping time: " << optimal_stopping_time <<" beta: " << beta << std::endl;
            if(i==1) nextLevelGI = nextGI;
            if(i==0){
                currentGI = nextGI * (1 - beta);
                optimalStoppingTime = optimal_stopping_time;
                computeData = &data;
            }
        }
    }

    double ProjectionBanditProcess::compute_gittins_index(int state) const {
        if (state <= optimalStoppingTime) {
            return currentGI;
        }
        return 0.0;
    }

} // namespace rai